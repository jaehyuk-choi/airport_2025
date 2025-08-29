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

SEOLLAL_START = pd.Timestamp("2025-01-24")
SEOLLAL_END   = pd.Timestamp("2025-02-02")
RUN_DATE      = pd.Timestamp("2025-01-10")   # 모델 실행일(14일 전)
MAX_ENCODER_LENGTH = 72                      # 최근 2개월 lookback
MAX_PRED_LENGTH    = 24                       # 24일 디코더(마지막 10일 평가)
BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-3
EARLY_STOP_PATIENCE = 6
HIDDEN_SIZE = 128
ATTN_HEADS = 4
DROP_OUT = 0.2
QUANTILES = [0.025, 0.5, 0.975]   # 신뢰구간용

# 학습/검증 분할(23~25 내에서)
# - train: 시작 ~ 2024-09-30
# - val  : 2024-01-01 ~ 2025-01-09(=RUN_DATE-1)
TRAIN_END = pd.Timestamp("2024-09-30")
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
categorical_cols = [
    "dow", "airport", "1_0_Encoding",
    "Holiday", "Holiday_Type",
    "Holiday_Pattern", "Holiday_Pattern_2", "is_weekend"
]

for c in categorical_cols:
    if c in df.columns:
        df[c] = df[c].astype(str)     # 또는 df[c] = df[c].astype("category")

# ② time_idx는 실수 아닌 정수형 보장
df["time_idx"] = df["time_idx"].astype(int)

# ③ (선택) 명시적 인코더 지정
categorical_encoders = {
    "airport": NaNLabelEncoder(add_nan=False),
    "dow": NaNLabelEncoder(add_nan=False),
    "Holiday_Type": NaNLabelEncoder(add_nan=True),   
    "Holiday": NaNLabelEncoder(add_nan=True),
    "1_0_Encoding": NaNLabelEncoder(add_nan=False),
    "Holiday_Pattern": NaNLabelEncoder(add_nan=True),
    "Holiday_Pattern_2": NaNLabelEncoder(add_nan=True)
}


# === [NEW] 설 특화 피처들: Rel_Seollal / bridge / block length / weight ===
df["year"] = df["datetime"].dt.year

# # (c) 설 주변 가중치(학습용)
def build_holiday_weights(df, base_weight=1.0,
                          token_boost=None, boundary_boost=0.5):
    df = df.copy()

    # Holiday_Type 문자열에서 끝 토큰 뽑기 (A1, B2, H 등)
    s = df["Holiday"].astype(str)
    df["htok"] = s.str.extract(r'(A\d+|B\d+|H)$', expand=False)

    # 토큰별 가중치 매핑
    if token_boost is None:
        token_boost = {
            "B5": 2.5, "B4": 2.5, "B3": 2.5, "B2": 2.5, "B1": 2.5,
            "H":  2.5,
            "A1": 2.5, "A2": 2.5, "A3": 2.5, "A4": 2.5, "A5": 2.5,
        }
    df["w_tok"] = df["htok"].map(token_boost).fillna(1.0)

    # # 전날 대비 Holiday_Type이 달라지면 boundary boost
    # df["htok_prev"] = df["htok"].shift(1)
    # df["is_boundary"] = (df["htok"] != df["htok_prev"]).astype(int)
    # df["w_boundary"] = 1.0 + boundary_boost * df["is_boundary"]

    # # 최종 weight
    # df["weight"] = base_weight * df["w_tok"] * df["w_boundary"]
    # return df.drop(columns=["w_tok", "w_boundary", "htok_prev"])

    df["weight"] = base_weight * df["w_tok"]
    return df.drop(columns=["w_tok"])

df = build_holiday_weights(df)

# df["weight"] = 1.0
# mask_seollal = df["Holiday_Type"].astype(str).str.contains("설날", na=False)
# df.loc[mask_seollal, "weight"] = df.loc[mask_seollal, "weight"] + 3.5

def has(*cols): return [c for c in cols if c in df.columns]

static_categoricals = has(GROUP_COL)
static_reals        = []  # 없으면 빈 리스트

# 캘린더/주기/설 특화: known
KNOWN_REAL_LIST = [
    TIME_COL, "Day_sin", "Day_cos",
    "Arrival_Num", "Departure_Num", "LCC_ratio", "DOM_Portion",
    "도민_Arr", "도민_Dep",
]
known_reals = has(*KNOWN_REAL_LIST)

# known_categoricals = has("dow", "Holiday", "Holiday_Type") "Holiday_Pattern"
known_categoricals = has("dow", "1_0_Encoding", "is_weekend", "Holiday_Type", "Holiday")

# 측정/실측 계열은 unknown(인코더 전용)
unknown_reals = has(
    TARGET,
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
    weight="weight"
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
from sklearn.metrics import mean_absolute_error, r2_score

# ---- Full window: RUN_DATE ~ SEOLLAL_END ----
mask_full = (merged["date"] >= RUN_DATE) & (merged["date"] <= SEOLLAL_END)
merged_full = merged.loc[mask_full].copy()

y_true_full = pd.to_numeric(merged_full[TARGET], errors="coerce")
y_pred_full = pd.to_numeric(merged_full["pred50"], errors="coerce")
ok_full = ~(y_true_full.isna() | y_pred_full.isna())

mae_full  = mean_absolute_error(y_true_full[ok_full], y_pred_full[ok_full])
mape_full = sk_mape(y_true_full[ok_full], np.maximum(1e-6, y_pred_full[ok_full]))
r2_full   = r2_score(y_true_full[ok_full], y_pred_full[ok_full])

print(f"[Evaluation | RUN_DATE to Seollal end] "
      f"MAPE: {mape_full*100:.2f}%, MAE: {mae_full:.2f}, R^2: {r2_full:.3f}")

# ---- Seollal window only: SEOLLAL_START ~ SEOLLAL_END ----
y_true_h = pd.to_numeric(merged_h[TARGET], errors="coerce")
y_pred_h = pd.to_numeric(merged_h["pred50"], errors="coerce")
ok_h = ~(y_true_h.isna() | y_pred_h.isna())

mae_h  = mean_absolute_error(y_true_h[ok_h], y_pred_h[ok_h])
mape_h = sk_mape(y_true_h[ok_h], np.maximum(1e-6, y_pred_h[ok_h]))
r2_h   = r2_score(y_true_h[ok_h], y_pred_h[ok_h])

print(f"[Evaluation | Seollal window] "
      f"MAPE: {mape_h*100:.2f}%, MAE: {mae_h:.2f}, R^2: {r2_h:.3f}")

# -----------------------------
# Plot: RUN_DATE ~ SEOLLAL_END
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(merged_full["date"], merged_full["Vehicle_Max"],
         label="Actual", marker="o", color="black")
plt.plot(merged_full["date"], merged_full["pred50"],
         label="Prediction (median)", marker="s", color="blue")
plt.fill_between(
    merged_full["date"],
    merged_full["pred025"],
    merged_full["pred975"],
    color="blue", alpha=0.2, label="95% Prediction Interval"
)

plt.axvline(SEOLLAL_START, color="red", linestyle="--", label="Seollal Start")
plt.axvline(SEOLLAL_END, color="red", linestyle="--", label="Seollal End")
plt.title(f"Jeju Airport Parking Demand Forecast ({RUN_DATE.date()} ~ {SEOLLAL_END.date()})\n"
          f"MAPE: {mape_full*100:.2f}%, MAE: {mae_full:.2f}")
plt.xlabel("Date")
plt.ylabel("Vehicle_Max")
plt.legend()
plt.grid(True)
plt.tight_layout()

save_path = "figs/jeju_forecast_run_to_seollal.png"
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"[Saved] {save_path}")

# import os

# os.makedirs("figs", exist_ok=True)  # 저장 폴더

# plt.figure(figsize=(10,5))
# plt.plot(merged_h["date"], merged_h["Vehicle_Max"], label="실제값", marker="o", color="black")
# plt.plot(merged_h["date"], merged_h["pred50"], label="예측 (중앙값)", marker="s", color="blue")
# plt.fill_between(
#     merged_h["date"],
#     merged_h["pred025"],
#     merged_h["pred975"],
#     color="blue", alpha=0.2, label="95% 예측구간"
# )

# plt.axvline(SEOLLAL_START, color="red", linestyle="--", label="설날 시작")
# plt.axvline(SEOLLAL_END, color="red", linestyle="--", label="설날 끝")
# plt.title("제주공항 주차 수요 - 설날 구간 예측 (95% CI)")
# plt.xlabel("날짜")
# plt.ylabel("Vehicle_Max")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# # 저장
# save_path = "figs/jeju_seollal_forecast.png"
# plt.savefig(save_path, dpi=200, bbox_inches="tight")
# plt.close()
# print(f"[저장 완료] {save_path}")

from tensorboard.backend.event_processing import event_accumulator

# runs/tft_jeju/version_0/ 폴더 경로
ea = event_accumulator.EventAccumulator("runs/tft_jeju/version_0/")
ea.Reload()

# 이벤트 파일에 저장된 스칼라 키 확인
print(ea.Tags()["scalars"])

# val_loss, train_loss 곡선 뽑기
train_loss = ea.Scalars("train_loss_epoch")
val_loss   = ea.Scalars("val_loss")

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
print("Available keys:", interpretation.keys())
print("\n[변수 중요도 요약]")
print("Static variables:", interpretation["static_variables"])
print("Encoder variables:", interpretation["encoder_variables"])
print("Decoder variables:", interpretation["decoder_variables"])

# -----------------------------
# 변수 중요도 추출 (Encoder/Decoder)
# -----------------------------
import pandas as pd
import matplotlib.pyplot as plt

# 해석 결과 텐서
encoder_imp = interpretation["encoder_variables"].detach().cpu().numpy()
decoder_imp = interpretation["decoder_variables"].detach().cpu().numpy()
static_imp  = interpretation["static_variables"].detach().cpu().numpy()

# 모델 내부 변수 이름 사용
encoder_vars = training.encoder_variables
decoder_vars = training.decoder_variables
static_vars  = training.static_variables

# DataFrame 구성
encoder_df = pd.DataFrame({"feature": encoder_vars, "importance": encoder_imp})
decoder_df = pd.DataFrame({"feature": decoder_vars, "importance": decoder_imp})
static_df  = pd.DataFrame({"feature": static_vars, "importance": static_imp})

# 정렬
encoder_df = encoder_df.sort_values("importance", ascending=False)
decoder_df = decoder_df.sort_values("importance", ascending=False)
static_df  = static_df.sort_values("importance", ascending=False)

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(18,6))

encoder_df.head(10).plot(kind="barh", x="feature", y="importance", ax=axes[0], title="Encoder Variables")
decoder_df.head(10).plot(kind="barh", x="feature", y="importance", ax=axes[1], title="Decoder Variables")
static_df.plot(kind="barh", x="feature", y="importance", ax=axes[2], title="Static Variables")

plt.tight_layout()
plt.savefig("figs/feature_importance_groups.png", dpi=200)
plt.close()

print("[저장 완료] figs/feature_importance_groups.png")
