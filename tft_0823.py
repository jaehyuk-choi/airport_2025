# -*- coding: utf-8 -*-
"""
TFT for Jeju (2023~2025) + Seollal evaluation
- Input : output/jeju_preprocessed.csv
- Output: runs/ (체크포인트/로그), preds/ (예측 csv)
"""

import os, random, warnings
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from sklearn.metrics import mean_absolute_percentage_error as sk_mape
from pytorch_forecasting.data.encoders import NaNLabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)
os.makedirs("runs", exist_ok=True)
os.makedirs("preds", exist_ok=True)

# -----------------------------
# 0) 재현성
# -----------------------------
SEED = 42
def set_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed, workers=True)
set_seed(SEED)

# -----------------------------
# 1) 환경/하이퍼파라미터
# -----------------------------
DATA_PATH = "output/jeju_preprocessed.csv"  # 전처리 산출물
TARGET = "Vehicle_Max"
GROUP_COL = "airport"
TIME_COL = "time_idx"

# 설날 평가 시나리오
SEOLLAL_START = pd.Timestamp("2025-01-24")
SEOLLAL_END   = pd.Timestamp("2025-02-02")
RUN_DATE      = pd.Timestamp("2025-01-10")   # 모델 실행일(14일 전)
MAX_ENCODER_LENGTH = 180                      # 최근 2개월 lookback
MAX_PRED_LENGTH    = 24                       # 24일 디코더(마지막 10일 평가)
BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-3
EARLY_STOP_PATIENCE = 6
HIDDEN_SIZE = 128
ATTN_HEADS = 4
DROP_OUT = 0.15
QUANTILES = [0.025, 0.5, 0.975]   # 신뢰구간용

# 학습/검증 분할(23~25 내에서)
# - train: 시작 ~ 2024-09-30
# - val  : 2024-01-01 ~ 2025-01-09(=RUN_DATE-1)
TRAIN_END = pd.Timestamp("2023-12-31")
VAL_END   = RUN_DATE - pd.Timedelta(days=1)

# -----------------------------
# 2) 데이터 로드
# -----------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
if GROUP_COL not in df.columns:
    df[GROUP_COL] = "jeju"
assert TARGET in df.columns, f"'{TARGET}' 없음"
assert TIME_COL in df.columns, f"'{TIME_COL}' 없음"

# 날짜별로 나눌 수 있도록 인덱스 보조
df["date"] = df["datetime"].dt.date


# ① 카테고리형으로 캐스팅 (문자열로)
for c in ["dow", "airport"]:
    if c in df.columns:
        df[c] = df[c].astype(str)     # 또는 df[c] = df[c].astype("category")

# ② time_idx는 실수 아닌 정수형 보장(혹시 몰라 안전)
df["time_idx"] = df["time_idx"].astype(int)

# ③ (선택) 명시적 인코더 지정
categorical_encoders = {
    "airport": NaNLabelEncoder(add_nan=False),
    "dow": NaNLabelEncoder(add_nan=False),
    "Holiday_Type": NaNLabelEncoder(add_nan=True),   # 추가
    "Holiday": NaNLabelEncoder(add_nan=True),
}


# === [NEW] 설 특화 피처들: Rel_Seollal / bridge / block length / weight ===
df["year"] = df["datetime"].dt.year

# (a) 설날 상대일(Rel_Seollal)
# Holiday_Type에 'Seollal' 있거나, Holiday(문자열/라벨)에 '설' 포함 시 설 블록으로 간주
has_seollal_type = "Holiday_Type" in df.columns
has_holiday_str  = "Holiday" in df.columns and df["Holiday"].dtype == object

is_seollal = df["Holiday_Type"].fillna("").str.contains("설날", case=False)

anchors = (
    df.loc[is_seollal, ["year","datetime"]]
      .groupby("year")["datetime"].median()
      .rename("seollal_anchor")
)

df = df.merge(anchors, on="year", how="left")
df["Rel_Seollal"] = (df["datetime"] - df["seollal_anchor"]).dt.days
df["Rel_Seollal_clip"] = df["Rel_Seollal"].clip(-14, 14)

# (b) 브릿지 / 블록 길이
# is_holiday 정규화(정수 0/1). 없으면 Holiday_Type 존재로 대체
if "Holiday" in df.columns and df["Holiday"].dtype != object:
    df["is_holiday"] = df["Holiday"].fillna(0).astype(int)
else:
    df["is_holiday"] = (df["Holiday_Type"].notna() if "Holiday_Type" in df.columns else 0).astype(int)

# is_weekend이 없으면 요일로 계산(일=6 기준)
if "is_weekend" not in df.columns:
    # dow가 0~6(월~일)로 들어왔다고 가정, 형식 다르면 필요에 맞게 매핑
    df["dow"] = df["dow"].astype(int)
    df["is_weekend"] = (df["dow"].isin([5, 6])).astype(int)
else:
    df["is_weekend"] = df["is_weekend"].astype(int)

# 연휴/주말 합친 off 여부
df["is_off"] = ((df["is_holiday"] > 0) | (df["is_weekend"] > 0)).astype(int)

# off 구간의 연속 블록 식별
# (값 변화 지점마다 그룹 id 증가)
df = df.sort_values("datetime").reset_index(drop=True)
df["off_grp_id"] = (df["is_off"] != df["is_off"].shift()).cumsum()

# off 블록 길이(휴무 블록일 때만 길이, 아니면 0)
df["off_block_len"] = df.groupby("off_grp_id")["is_off"].transform("sum") * df["is_off"]

# 블록이 '주말'과 '공휴일'을 모두 포함하면 bridge=1
blk_agg = df.groupby("off_grp_id").agg(
    blk_has_off=("is_off", "max"),
    blk_has_wknd=("is_weekend", "max"),
    blk_has_hol=("is_holiday", "max"),
).reset_index()
blk_agg["block_is_bridge"] = ((blk_agg["blk_has_off"] == 1) & (blk_agg["blk_has_wknd"] == 1) & (blk_agg["blk_has_hol"] == 1)).astype(int)
df = df.merge(blk_agg[["off_grp_id", "block_is_bridge"]], on="off_grp_id", how="left")
df["is_bridge"] = df["block_is_bridge"].fillna(0).astype(int)
df["is_bridge"] = df["is_bridge"].astype(str)

# (c) 설 주변 가중치(학습용)
near_seollal = df["Rel_Seollal"].abs().le(7).fillna(False)
df["weight"] = 1.0 + 3.5 * near_seollal.astype(float)


# -----------------------------
# 3) known / unknown 변수 구성
#    - known: 미래에 확정적으로 알 수 있는 캘린더/명절 계열
#    - unknown: 과거 관측만 이용(인코더 전용), 디코더에서는 제공하지 않음
# -----------------------------
# # 동적으로 존재하는 컬럼만 활용
# def has(*cols): return [c for c in cols if c in df.columns]

# static_categoricals = has(GROUP_COL)
# static_reals        = []  # 없음

# # 캘린더/명절: known reals/categoricals
# known_reals = has(
#     TIME_COL, "Day_sin", "Day_cos", "month_sin", "month_cos", "Arrival_Num", "Departure_Num", "LCC_ratio", "DOM_Portion", "도민_Arr", "도민_Dep" 
#     "is_weekend"
# )
# known_categoricals = has("dow", 'Holiday', 'Holiday_Type')  

# # unknown reals (인코더 전용; 디코더는 자동으로 비워짐)
# unknown_reals = has(
#     TARGET, 
#     "entry_count", "exit_count",
#     "TempAvg_C", "Precip_mm", "Late", 
# )
# # target은 unknown_reals에 포함되어야 PF에서 정상 작동
# if TARGET not in unknown_reals:
#     unknown_reals = [TARGET] + unknown_reals

# print("\n[변수 구성 요약]")
# print(" - static_categoricals:", static_categoricals)
# print(" - static_reals       :", static_reals)
# print(" - known_categoricals :", known_categoricals)
# print(" - known_reals        :", known_reals)
# print(" - unknown_reals(enc) :", unknown_reals)

def has(*cols): return [c for c in cols if c in df.columns]

static_categoricals = has(GROUP_COL)
static_reals        = []  # 없으면 빈 리스트

# 캘린더/주기/설 특화: known
KNOWN_REAL_WHITELIST = [
    TIME_COL, "Day_sin", "Day_cos", "month_sin", "month_cos",
    "is_weekend", "Rel_Seollal_clip",
    "Arrival_Num", "Departure_Num", "LCC_ratio", "DOM_Portion",
    "도민_Arr", "도민_Dep",
]
known_reals = has(*KNOWN_REAL_WHITELIST)

known_categoricals = has("dow", "Holiday", "Holiday_Type")

# 측정/실측 계열은 unknown(인코더 전용)
unknown_reals = has(
    TARGET,
    "entry_count", "exit_count",
    "TempAvg_C", "Precip_mm", "Late"
)
if TARGET not in unknown_reals:
    unknown_reals = [TARGET] + unknown_reals

print("\n[변수 구성 요약]")
print(" - static_categoricals:", static_categoricals)
print(" - static_reals       :", static_reals)
print(" - known_categoricals :", known_categoricals)
print(" - known_reals        :", known_reals)
print(" - unknown_reals(enc) :", unknown_reals)

# -----------------------------
# 4) 기간 분할
# -----------------------------
train_mask = df["datetime"] <= TRAIN_END
val_mask   = (df["datetime"] > TRAIN_END) & (df["datetime"] <= VAL_END)
pred_mask  = df["datetime"] <= SEOLLAL_END  # 예측은 RUN_DATE부터 자동 생성

df_train = df.loc[train_mask].copy()
df_val   = df.loc[val_mask].copy()
df_pred  = df.loc[pred_mask].copy()

print("\n[기간 요약]")
print(f" - train: {df_train['datetime'].min().date()} ~ {df_train['datetime'].max().date()} (n={len(df_train):,})")
print(f" - val  : {df_val['datetime'].min().date()} ~ {df_val['datetime'].max().date()} (n={len(df_val):,})")
print(f" - pred : .. ~ {df_pred['datetime'].max().date()} (n={len(df_pred):,})")
run_idx = int(df.loc[df["datetime"]==RUN_DATE, TIME_COL].iloc[0])
print(f" - RUN_DATE={RUN_DATE.date()} (time_idx={run_idx}) / 예측길이={MAX_PRED_LENGTH}일")

# -----------------------------
# 5) TimeSeriesDataSet 정의
# -----------------------------
# target_normalizer: 공항별 스케일 정규화
target_normalizer = GroupNormalizer(
    groups=[GROUP_COL], transformation="softplus"  # 비음수 타깃 안정화
)

training = TimeSeriesDataSet(
    df_train,
    time_idx=TIME_COL,
    target=TARGET,
    group_ids=[GROUP_COL],
    max_encoder_length=MAX_ENCODER_LENGTH,
    max_prediction_length=MAX_PRED_LENGTH,
    static_categoricals=static_categoricals,
    static_reals=static_reals,
    time_varying_known_categoricals=known_categoricals,
    time_varying_known_reals=known_reals,
    time_varying_unknown_reals=unknown_reals,
    target_normalizer=target_normalizer,
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
    categorical_encoders=categorical_encoders,
    weight="weight",
)

validation = TimeSeriesDataSet.from_dataset(
    training, df_val, predict=False, stop_randomization=True
)

# RUN_DATE 기준으로 예측 세트를 생성(predict=True)
# - from_dataset을 쓰면 스케일/인코더 일관성이 유지됨
# - 이후 결과에서 RUN_DATE로 시작하는 디코더만 골라 평가
prediction_dataset = TimeSeriesDataSet.from_dataset(
    training, df_pred, predict=True, stop_randomization=True
)

# -----------------------------
# 6) DataLoader
# -----------------------------
num_workers = 0  # 노트북/서버 상황 따라 조정
train_loader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=num_workers)
val_loader   = validation.to_dataloader(train=False, batch_size=BATCH_SIZE*2, num_workers=num_workers)
pred_loader  = prediction_dataset.to_dataloader(train=False, batch_size=BATCH_SIZE*2, num_workers=num_workers)

# -----------------------------
# 7) 모델 정의
# -----------------------------
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=LR,
    hidden_size=HIDDEN_SIZE,
    attention_head_size=ATTN_HEADS,
    dropout=DROP_OUT,
    loss=QuantileLoss(quantiles=QUANTILES),
    output_size=len(QUANTILES),
    reduce_on_plateau_patience=3,
)

print(f"\n모델 파라미터 수: {tft.size()/1e6:.2f}M")

# -----------------------------
# 8) 콜백/로거
# -----------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=EARLY_STOP_PATIENCE,
    mode="min"
)
ckpt = ModelCheckpoint(
    dirpath="runs",
    filename="tft-{epoch:02d}-{val_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_top_k=1
)
logger = TensorBoardLogger("runs", name="tft_jeju")

# -----------------------------
# 9) 학습
# -----------------------------
accelerator = "gpu" if torch.cuda.is_available() else "cpu"
trainer = Trainer(
    max_epochs=EPOCHS,
    accelerator=accelerator,
    devices=1,
    enable_checkpointing=True,
    callbacks=[early_stop, ckpt, TQDMProgressBar(refresh_rate=50)],
    logger=logger,
    gradient_clip_val=0.1,
)

trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
best_path = ckpt.best_model_path or ckpt.last_model_path
print(f"\n[체크포인트] {best_path}")
if best_path:
    tft = TemporalFusionTransformer.load_from_checkpoint(best_path)

# -----------------------------
# 10) 검증셋 성능(중앙값 예측)
# -----------------------------
val_pred = tft.predict(val_loader).detach().cpu().numpy()  # (N, pred_len)
val_true = torch.cat([y[0] for x, y in iter(val_loader)], dim=0).detach().cpu().numpy()

# 각 샘플의 디코더 첫 시점만 비교하는 일반적 관행(여기선 전체 horizon 평균 MAPE도 같이)
val_pred_med = val_pred[:, :].reshape(-1)
val_true_flat= val_true[:, :].reshape(-1)
val_mape_all = sk_mape(val_true_flat, np.maximum(1e-6, val_pred_med))
print(f"\n[Validation] 전체 horizon MAPE: {val_mape_all*100:.2f}%")

# -----------------------------
# 11) RUN_DATE 기반 24일 예측 생성
# -----------------------------
# raw 모드로 예측하여 디코더 time_idx를 회수
raw_preds = tft.predict(pred_loader, mode="raw", return_x=True)
pred_tensor = raw_preds.output[0].detach().cpu()          # shape: [B, pred_len, n_quantiles]
dec_timeidx = raw_preds.x["decoder_time_idx"].detach().cpu()  # [B, pred_len]
groups      = raw_preds.x["groups"].detach().cpu()    # [B, 1] (airport 인덱스)

# 중앙값(0.5 quantile)만 추출
q50_idx = QUANTILES.index(0.5)
pred50 = pred_tensor[..., q50_idx]  # [B, pred_len]

# 배치 단위 -> DataFrame 구성
records = []
min_date = df["datetime"].min()
idx2date = (
    df[["time_idx","datetime"]]
      .drop_duplicates()
      .set_index("time_idx")["datetime"]
)

for i in range(pred50.size(0)):
    for j in range(pred50.size(1)):
        ti = int(dec_timeidx[i, j].item())
        dt = pd.Timestamp(idx2date.loc[ti]).date()
        records.append({
            "decoder_start_time_idx": int(dec_timeidx[i, 0].item()),
            "date": pd.Timestamp(dt),
            "pred50": float(pred50[i, j].item())
        })
pred_df = pd.DataFrame(records)

# RUN_DATE로 시작하는 디코더만 남김
run_start_idx = int(df.loc[df["datetime"]==RUN_DATE, TIME_COL].iloc[0])
pred_run = pred_df[pred_df["decoder_start_time_idx"] == run_start_idx].copy()

# 실제값 merge 및 평가
truth = df.loc[(df["datetime"] >= RUN_DATE) & (df["datetime"] <= SEOLLAL_END),
               ["datetime", TARGET]].rename(columns={"datetime":"date"})
merged = pred_run.merge(truth, on="date", how="left")

# 전체 24일 성능
mape_24 = sk_mape(merged[TARGET].values, np.maximum(1e-6, merged["pred50"].values))
print(f"\n[RUN_DATE 예측] 24일 전체 MAPE: {mape_24*100:.2f}%  (기간: {RUN_DATE.date()}~{SEOLLAL_END.date()})")

# 설 연휴 10일 성능 (2025-01-24 ~ 2025-02-02)
mask_holiday = (merged["date"] >= SEOLLAL_START) & (merged["date"] <= SEOLLAL_END)
merged_h = merged.loc[mask_holiday].copy()
mape_h = sk_mape(merged_h[TARGET].values, np.maximum(1e-6, merged_h["pred50"].values))
print(f"[설 연휴 10일] MAPE: {mape_h*100:.2f}%  (기간: {SEOLLAL_START.date()}~{SEOLLAL_END.date()})")

# 저장
out_all = merged.sort_values("date")
out_hol = merged_h.sort_values("date")
out_all.to_csv("preds/jeju_run24_all.csv", index=False, encoding="utf-8-sig")
out_hol.to_csv("preds/jeju_run24_seollal.csv", index=False, encoding="utf-8-sig")
print("\n[저장] preds/jeju_run24_all.csv, preds/jeju_run24_seollal.csv")

import matplotlib.pyplot as plt

# -----------------------------
# 예측 quantiles 추출 (이미 pred_tensor, dec_timeidx 있음)
# -----------------------------
q025_idx = QUANTILES.index(0.025)
q50_idx  = QUANTILES.index(0.5)
q975_idx = QUANTILES.index(0.975)

pred025 = pred_tensor[..., q025_idx]
pred50  = pred_tensor[..., q50_idx]
pred975 = pred_tensor[..., q975_idx]

# DataFrame 구성
records = []
idx2date = (
    df[["time_idx", "datetime"]]
      .drop_duplicates()
      .set_index("time_idx")["datetime"]
)

min_date = df["datetime"].min()
for i in range(pred50.size(0)):
    start_idx = int(dec_timeidx[i, 0].item())
    for j in range(pred50.size(1)):
        ti = int(dec_timeidx[i, j].item())
        dt = pd.Timestamp(idx2date.loc[ti])  # ✅ 안전한 날짜 복원

        records.append({
            "decoder_start_time_idx": start_idx,
            "date": dt,                               # 그대로 Timestamp 저장
            "pred025": float(pred025[i, j].item()),
            "pred50":  float(pred50[i, j].item()),
            "pred975": float(pred975[i, j].item()),
        })

pred_all = pd.DataFrame(records)

# RUN_DATE 기준 윈도우만 추출
pred_run = pred_all[pred_all["decoder_start_time_idx"] == run_start_idx].copy()

# 실제값 병합
truth = df.loc[(df["datetime"] >= RUN_DATE) & (df["datetime"] <= SEOLLAL_END),
               ["datetime", TARGET]].rename(columns={"datetime":"date"})
merged = pred_run.merge(truth, on="date", how="left")

# 설날 구간만 필터링
merged_h = merged[(merged["date"] >= SEOLLAL_START) & (merged["date"] <= SEOLLAL_END)]

# -----------------------------
# 그래프 그리기
# -----------------------------
import os

os.makedirs("figs", exist_ok=True)  # 저장 폴더

plt.figure(figsize=(10,5))
plt.plot(merged_h["date"], merged_h["Vehicle_Max"], label="실제값", marker="o", color="black")
plt.plot(merged_h["date"], merged_h["pred50"], label="예측 (중앙값)", marker="s", color="blue")
plt.fill_between(
    merged_h["date"],
    merged_h["pred025"],
    merged_h["pred975"],
    color="blue", alpha=0.2, label="95% 예측구간"
)

plt.axvline(SEOLLAL_START, color="red", linestyle="--", label="설날 시작")
plt.axvline(SEOLLAL_END, color="red", linestyle="--", label="설날 끝")
plt.title("제주공항 주차 수요 - 설날 구간 예측 (95% CI)")
plt.xlabel("날짜")
plt.ylabel("Vehicle_Max")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 저장
save_path = "figs/jeju_seollal_forecast.png"
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"[저장 완료] {save_path}")

from tensorboard.backend.event_processing import event_accumulator

# runs/tft_jeju/version_0/ 폴더 경로
ea = event_accumulator.EventAccumulator("runs/tft_jeju/version_0/")
ea.Reload()

# 이벤트 파일에 저장된 스칼라 키 확인
print(ea.Tags()["scalars"])

# val_loss, train_loss 곡선 뽑기
val_loss = ea.Scalars("val_loss")
train_loss = ea.Scalars("train_loss_epoch")

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot([x.step for x in train_loss], [x.value for x in train_loss], label="Train Loss")
plt.plot([x.step for x in val_loss], [x.value for x in val_loss], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figs/loss_curve.png", dpi=200)
plt.close()
print("[저장 완료] figs/loss_curve.png")


##############
# 훈련된 모델에서 변수 중요도 뽑기
interpretation = tft.interpret_output(raw_preds.output, reduction="sum")

# 변수별 중요도 DataFrame
importance = tft.compute_feature_importance(
    raw_preds.output, raw_preds.x, reduction="sum"
)
print(importance.head(20))

from pytorch_forecasting.models.temporal_fusion_transformer.tuning import plot_interpretation

# 특정 배치(예: 첫 샘플) attention 확인
fig = plot_interpretation(
    interpretation,
    idx=0,                # 배치 내 index
    decoder_len=MAX_PRED_LENGTH
)
fig.savefig("figs/tft_attention.png", dpi=200, bbox_inches="tight")

# 전체 데이터셋 해석
raw_predictions = tft.predict(val_loader, mode="raw", return_x=True)

interpretation = tft.interpret_output(raw_predictions.output, reduction="sum")

# 예측값 분해 시각화
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import visualize_prediction

fig = visualize_prediction(
    tft, raw_predictions.x, raw_predictions.output, idx=0, show_future_observed=True
)
fig.savefig("figs/tft_interpretation_example.png", dpi=200, bbox_inches="tight")

