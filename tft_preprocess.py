# -*- coding: utf-8 -*-
"""
Airport - Preprocessing
- _was_imputed 플래그 완전 제거
- 이상치(0/음수/롤링-IQR/MAD) → NaN 마스킹 후 보간
- 고급 보간 (year×month×weekday → month×weekday → month)
- 다중공선성 제거, RF feature selection
- 상세 콘솔 리포트
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# =========================
# 유틸
# =========================
def winsorize(s: pd.Series, lower=0.005, upper=0.995):
    if s.isna().all():
        return s
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lower=lo, upper=hi)

def contiguous_blocks(idx: pd.DatetimeIndex, max_gap_days=1):
    if len(idx) == 0:
        return []
    diffs = idx.to_series().diff().dt.days.fillna(1).astype(int)
    block_ids = (diffs > max_gap_days).cumsum()
    return [sub.index for _, sub in idx.to_series().groupby(block_ids)]

def add_lag_and_roll_safe(df: pd.DataFrame, target="Vehicle_Max",
                          lags=(1,7,14), roll_windows=(7,14)) -> pd.DataFrame:
    out = df.copy()
    if target not in out.columns:
        return out
    for block_idx in contiguous_blocks(out.index, max_gap_days=1):
        sub = out.loc[block_idx, [target]].copy()
        for lag in lags:
            out.loc[block_idx, f"{target}_lag_{lag}"] = sub[target].shift(lag)
        for w in roll_windows:
            out.loc[block_idx, f"{target}_roll_mean_{w}"] = sub[target].shift(1).rolling(w, min_periods=1).mean()
            out.loc[block_idx, f"{target}_roll_std_{w}"]  = sub[target].shift(1).rolling(w, min_periods=1).std()
    return out

def consecutive_nan_runs(s: pd.Series):
    isna = s.isna().astype(int)
    grp = (isna.ne(isna.shift())).cumsum()
    runs = isna.groupby(grp).sum()
    return runs[runs > 0].tolist()


# =========================
# 0) 기본 전처리(보간/이상치 전)
# =========================
def preprocess_basics(df: pd.DataFrame,
                           start_date="2014-01-01",
                           end_date="2025-04-30") -> pd.DataFrame:
    before_cols = df.columns.tolist()
    df.drop(columns=['exit_count', 'entry_count'], errors='ignore', inplace=True)
    # datetime 복원 + 연도 필터
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        df["datetime"] = pd.to_datetime(df[["Year","Month","Day"]], errors="coerce")
    df = df[(df["datetime"] >= pd.Timestamp(start_date)) &
            (df["datetime"] <= pd.Timestamp(end_date))].copy()
    df.sort_values("datetime", inplace=True)

    # 타깃 통일
    if "Vehicle_Max_New" in df.columns and "Vehicle_Max" not in df.columns:
        df["Vehicle_Max"] = df["Vehicle_Max_New"]

    # 휴일 결측 기본값
    for col in ["Holiday","Holiday_Type", "Holiday_Pattern", "Holiday_Pattern_2"]:
        if col in df.columns:
            df[col] = df[col].fillna("N")

    # 삭제(누출/중복/잡음)
    drop_cols = [
        "Year","Month","Day","Vehicle_Max_New",
        "Holiday_Encoding","Is_Holiday",
        # "1_0_Encoding",
        "Unnamed: 28","Unnamed: 27",
        "Tobit_Upper_Limit","Vehicle_Tobit","Vehicle_Tobit_Adjusted",
        "Up_Down",
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

    # 공항/인덱스
    df["airport"] = "jeju"
    df.set_index("datetime", inplace=True)

    # 강수량 결측 0
    if "Precip_mm" in df.columns:
        df["Precip_mm"] = df["Precip_mm"].fillna(0)

    # 캘린더 파생
    if "Day_of_the_Week" in df.columns:
        dow_raw = df["Day_of_the_Week"].astype(float)
        df["Day_sin"] = np.sin(2*np.pi*dow_raw/7.0)
        df["Day_cos"] = np.cos(2*np.pi*dow_raw/7.0)
        df.drop(columns=["Day_of_the_Week"], inplace=True, errors="ignore")
    else:
        dow = df.index.weekday
        df["Day_sin"] = np.sin(2*np.pi*dow/7.0)
        df["Day_cos"] = np.cos(2*np.pi*dow/7.0)

    df["month"] = df.index.month
    df["doy"]   = df.index.dayofyear
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12.0)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12.0)
    df["doy_sin"]   = np.sin(2*np.pi*df["doy"]/365.25)
    df["doy_cos"]   = np.cos(2*np.pi*df["doy"]/365.25)
    df["dow"] = df.index.weekday
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # 타깃 안정화
    if "Vehicle_Max" not in df.columns:
        raise ValueError("타깃 'Vehicle_Max'가 없습니다.")
    df["Vehicle_Max"] = winsorize(df["Vehicle_Max"], 0.005, 0.995).clip(lower=0)

    # time_idx
    min_date = df.index.min()
    df["time_idx"] = (df.index - min_date).days.astype(int)

    # 리포트
    after_cols = df.reset_index().columns.tolist()
    kept   = [c for c in before_cols if c in after_cols]
    dropped= [c for c in before_cols if c not in after_cols]
    added  = [c for c in after_cols if c not in before_cols]

    print("\n[1/6] 컬럼 변화 요약 (23~25 필터 적용)")
    print(f"- 입력 칼럼 수: {len(before_cols)}")
    print(f"- 출력 칼럼 수: {len(after_cols)}  (datetime 포함)")
    print("  · 삭제:", sorted(dropped))
    print("  · 유지:", sorted(kept))
    print("  · 추가:", sorted(added))
    print(f"  · 날짜 범위: {df.index.min().date()} ~ {df.index.max().date()}  (총 {len(df):,}일)")

    return df


# =========================
# 1) 이상치 마스킹 (→ NaN)
# =========================
def detect_outliers_rolling_iqr(s: pd.Series, window=5, k=2.5, past_only=True):
    """
    롤링 IQR 기반 이상치(True = 이상치)
    - past_only=True면 현재값 판단에 '과거'만 사용(shift(1))
    """
    base = s.shift(1) if past_only else s
    q1 = base.rolling(window, min_periods=max(5, window//3)).quantile(0.25)
    q3 = base.rolling(window, min_periods=max(5, window//3)).quantile(0.75)
    iqr = q3 - q1
    low = q1 - k*iqr
    high = q3 + k*iqr
    return (s < low) | (s > high)

def detect_outliers_rolling_mad(s: pd.Series, window=35, z=5.0, past_only=True):
    """
    롤링 Median + MAD 기반 이상치(True = 이상치)
    """
    base = s.shift(1) if past_only else s
    med = base.rolling(window, min_periods=max(5, window//3)).median()
    mad = (base - med).abs().rolling(window, min_periods=max(5, window//3)).median()
    robust_z = 0.6745 * (s - med) / (mad.replace(0, np.nan))
    return robust_z.abs() > z

def mask_outliers(df: pd.DataFrame,
                  zero_outlier_cols=None,
                  iqr_cols=None,
                  mad_cols=None,
                  iqr_window=35, iqr_k=2.5,
                  mad_window=35, mad_z=5.0,
                  past_only=True):
    """
    - zero_outlier_cols: 0/음수 → NaN (counts/strict-positive에 권장)
    - iqr_cols: 롤링 IQR 이상치 → NaN
    - mad_cols: 롤링 MAD 이상치 → NaN
    """
    zero_outlier_cols = zero_outlier_cols or []
    iqr_cols = iqr_cols or []
    mad_cols = mad_cols or []

    print("\n[2/6] 이상치 마스킹 요약")
    for c in zero_outlier_cols:
        if c not in df.columns: 
            continue
        n0 = int((df[c] == 0).sum())
        nn = int((df[c] < 0).sum())
        if n0 or nn:
            df.loc[df[c] <= 0, c] = np.nan
        print(f" - {c}: 0값 {n0}건, 음수 {nn}건 → NaN 전환")

    for c in iqr_cols:
        if c not in df.columns: 
            continue
        mask = detect_outliers_rolling_iqr(df[c].astype(float), window=iqr_window, k=iqr_k, past_only=past_only)
        n = int(mask.sum())
        if n:
        #     df.loc[mask, c] = np.nan
        # print(f" - {c}: 롤링-IQR 이상치 {n}건 → NaN")
        # if n:
            # 🔎 이상치 출력
            outliers = df.loc[mask, c]
            print(f"\n[IQR 이상치] {c}: {n}건")
            print(outliers)  # 처음 20건만 보기
            df.loc[mask, c] = np.nan
        else:
            print(f" - {c}: 이상치 없음")

    for c in mad_cols:
        if c not in df.columns: 
            continue
        mask = detect_outliers_rolling_mad(df[c].astype(float), window=mad_window, z=mad_z, past_only=past_only)
        n = int(mask.sum())
        if n:
        #     df.loc[mask, c] = np.nan
        # print(f" - {c}: 롤링-MAD 이상치 {n}건 → NaN")
            outliers = df.loc[mask, c]
            print(f"\n[MAD 이상치] {c}: {n}건")
            print(outliers)  # 처음 20건만 보기
            df.loc[mask, c] = np.nan
        else:
            print(f" - {c}: 이상치 없음")

    return df


# =========================
# 2) 고급 보간기 (fit/transform)
# =========================
def _rolling_median_fill(s: pd.Series, window=7, max_len=3):
    out = s.copy()
    isna = out.isna()
    if not isna.any():
        return out, 0
    grp = (isna.ne(isna.shift())).cumsum()
    filled = 0
    med = out.shift(1).rolling(window, min_periods=1).median()
    for _, idx in out[isna].groupby(grp[isna]).groups.items():
        if len(idx) <= max_len:
            out.loc[idx] = med.loc[idx]
            filled += len(idx)
    return out, filled

class ImputerAdvanced:
    """
    변수별 규칙 기반 + 다단계 대체:
      (year, month, dow) → (month, dow) → (month)
    """
    def __init__(self, max_linear_gap_temp=7, max_gap_short_counts=3):
        self.max_linear_gap_temp  = max_linear_gap_temp
        self.max_gap_short_counts = max_gap_short_counts
        self.stats_y_m_d = None
        self.stats_m_d   = None
        self.stats_m     = None
        # 변수 타입
        self.smooth_cols = ["TempAvg_C"]
        self.zero_cols   = ["Precip_mm"]  # 결측=0 가정
        self.count_cols  = ["Vehicle_Max","Arrival_Num","Departure_Num",
                            "entry_count","exit_count",
                            "도민_Arr","도민_Dep","도민_입차","도민_출차",
                            "도민_Arrival","도민_Departure"]
        self.ratio_cols  = ["LCC_ratio","FSC_ratio","DOM_Portion","INT_Portion"]
        self.delay_cols  = ["Late"]
        self.inter_cols  = ["Temp_Precip_Interaction"]

    def _sel(self, df, cols):  return [c for c in cols if c in df.columns]

    def fit(self, df: pd.DataFrame):
        tmp = df.copy()
        tmp["year"]  = tmp.index.year
        tmp["month"] = tmp.index.month
        tmp["dow"]   = tmp.index.weekday
        numeric = tmp.select_dtypes(include=[np.number]).columns
        self.stats_y_m_d = tmp.groupby(["year","month","dow"])[list(numeric)].median()
        self.stats_m_d   = tmp.groupby(["month","dow"])[list(numeric)].median()
        self.stats_m     = tmp.groupby(["month"])[list(numeric)].median()
        return self

    def _lookup(self, idx, col):
        for store, key in [(self.stats_y_m_d, (idx.year, idx.month, idx.weekday())),
                           (self.stats_m_d,   (idx.month, idx.weekday())),
                           (self.stats_m,     (idx.month))]:
            try:
                return store.loc[key, col]
            except KeyError:
                continue
        return np.nan

    def transform(self, df: pd.DataFrame, limit_to_index: pd.DatetimeIndex=None) -> pd.DataFrame:
        x = df.copy()
        scope = x.index if limit_to_index is None else x.index[x.index.isin(limit_to_index)]

        # 긴 결측 경고
        print("\n[3/6] 긴 결측 경고(Top 5 by column)")
        for c in x.columns:
            if not pd.api.types.is_numeric_dtype(x[c]): 
                continue
            runs = consecutive_nan_runs(x.loc[scope, c])
            if len(runs) > 0 and max(runs) >= 30:
                runs_sorted = sorted(runs, reverse=True)[:5]
                print(f" - {c}: 최장 {max(runs_sorted)}일 연속 NaN, 상위 {runs_sorted}")

        # 1) zero cols
        for c in self._sel(x, self.zero_cols):
            mask = x[c].isna() & x.index.isin(scope)
            x.loc[mask, c] = 0.0

        # 2) smooth cols (Temp)
        for c in self._sel(x, self.smooth_cols):
            s = x[c].copy()
            s.loc[scope] = s.loc[scope].interpolate(method="time", limit=self.max_linear_gap_temp)
            still = s.isna() & s.index.isin(scope)
            for idx in s.index[still]:
                v = self._lookup(idx, c)
                if not pd.isna(v): s.loc[idx] = v
            x[c] = s

        # 3) count cols
        for c in self._sel(x, self.count_cols):
            s = x[c].copy()
            s_scope = s.copy(); s_scope.loc[~s.index.isin(scope)] = np.nan
            s_new, _ = _rolling_median_fill(s_scope, window=7, max_len=self.max_gap_short_counts)
            sel = s.isna() & (~s_new.isna()); s.loc[sel] = s_new.loc[sel]
            still = s.isna() & s.index.isin(scope)
            for idx in s.index[still]:
                v = self._lookup(idx, c)
                if not pd.isna(v): s.loc[idx] = v
            s = s.clip(lower=0)
            x[c] = s

        # 4) delay cols (Late)
        for c in self._sel(x, self.delay_cols):
            s = x[c].copy()
            still = s.isna() & s.index.isin(scope)
            for idx in s.index[still]:
                v = self._lookup(idx, c)
                if not pd.isna(v): s.loc[idx] = v
            ql, qu = s.quantile(0.005), s.quantile(0.995)
            s = s.clip(ql, qu)
            x[c] = s

        # 5) ratio cols
        for c in self._sel(x, self.ratio_cols):
            s = x[c].copy()
            s.loc[scope] = s.loc[scope].ffill(limit=7).bfill(limit=7)
            still = s.isna() & s.index.isin(scope)
            for idx in s.index[still]:
                v = self._lookup(idx, c)
                if not pd.isna(v): s.loc[idx] = v
            s = s.clip(0.0, 1.0)
            x[c] = s

        # 쌍 정규화
        if all(col in x.columns for col in ["LCC_ratio","FSC_ratio"]):
            ssum = x["LCC_ratio"] + x["FSC_ratio"]
            over1 = (ssum > 1.0) & x.index.isin(scope)
            if over1.any():
                x.loc[over1, ["LCC_ratio","FSC_ratio"]] = x.loc[over1, ["LCC_ratio","FSC_ratio"]].div(ssum[over1], axis=0)
                print("  [정규화] LCC+FSC > 1.0 행 정규화:", int(over1.sum()))
        if all(col in x.columns for col in ["DOM_Portion","INT_Portion"]):
            ssum = x["DOM_Portion"] + x["INT_Portion"]
            over1 = (ssum > 1.0) & x.index.isin(scope)
            if over1.any():
                x.loc[over1, ["DOM_Portion","INT_Portion"]] = x.loc[over1, ["DOM_Portion","INT_Portion"]].div(ssum[over1], axis=0)
                print("  [정규화] DOM+INT > 1.0 행 정규화:", int(over1.sum()))

        # 상호작용 재계산
        if ("Temp_Precip_Interaction" in x.columns) and all(col in x.columns for col in ["TempAvg_C","Precip_mm"]):
            x["Temp_Precip_Interaction"] = x["TempAvg_C"] * x["Precip_mm"]

        # 잔여 NaN 경고
        remain = x.select_dtypes(include=[np.number]).isna().sum()
        remain = remain[remain > 0].sort_values(ascending=False)
        if len(remain) > 0:
            print("\n  [주의] 보간 후에도 NaN 잔존(상위 10개):")
            print(remain.head(10))

        return x


# =========================
# 3) 다중공선성 제거
# =========================
def remove_multicollinearity_numeric(df, target="Vehicle_Max", threshold=0.95):
    """
    1) 스페셜 페어 우선 처리: (LCC,FSC), (DOM,INT) 중 뒤쪽 제거
    2) 숫자형 상관 > threshold → 평균상관 높은 쪽 제거
    """
    to_drop = []

    special_pairs = [("LCC_ratio","FSC_ratio"), ("DOM_Portion","INT_Portion")]
    for a, b in special_pairs:
        if a in df.columns and b in df.columns:
            to_drop.append(b)

    df2 = df.drop(columns=to_drop, errors="ignore")

    num = df2.select_dtypes(include=[np.number]).drop(columns=[target], errors="ignore")
    if num.shape[1] == 0:
        print("\n[4/6] 다중공선성: 숫자형 피처 없음 → 스킵")
        return df2, to_drop

    corr = num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    auto_drop = set()
    pairs = []
    for r in upper.index:
        for c in upper.columns:
            v = upper.loc[r, c]
            if pd.notna(v) and v > threshold:
                pairs.append((r, c, float(v)))
                m_r = upper.loc[r].mean()
                m_c = upper[c].mean()
                drop = r if m_r >= m_c else c
                if drop != target:
                    auto_drop.add(drop)

    print("\n[4/6] 다중공선성 요약")
    if pairs:
        print(f"- 상관 > {threshold} 피처쌍 예시(최대 10개):")
        for (r,c,v) in sorted(pairs, key=lambda x: -x[2])[:10]:
            print(f"  · {r} ↔ {c} = {v:.3f}")
    else:
        print("- 임계 초과 상관쌍 없음")

    if to_drop:
        print(f"- 스페셜 페어 제거: {sorted(to_drop)}")
    if auto_drop:
        print(f"- 상관 기반 자동 제거: {sorted(auto_drop)}")

    df_final = df2.drop(columns=list(auto_drop), errors="ignore")
    return df_final, sorted(to_drop) + sorted(list(auto_drop))


# =========================
# 4) RF 기반 feature selection
# =========================
def rf_feature_filter(df, target="Vehicle_Max",
                      train_end="2025-01-09",
                      importance_thresh=0.002, topn_show=20):
    train = df.loc[df.index <= pd.Timestamp(train_end)].copy()
    num_feats = [c for c in train.select_dtypes(include=[np.number]).columns if c not in [target,"time_idx"]]
    if not num_feats:
        print("\n[5/6] RF 필터: 숫자형 피처 없음 → 스킵")
        return df, pd.Series(dtype=float), []

    X = train[num_feats].fillna(0.0)
    y = train[target].astype(float)

    rf = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    imps = pd.Series(rf.feature_importances_, index=num_feats).sort_values(ascending=False)

    low = imps[imps < importance_thresh].index.tolist()

    print("\n[5/6] RF Feature Importance")
    print("- Top 중요 피처:")
    print(imps.head(topn_show).to_string())
    print(f"- 중요도 < {importance_thresh:.4f} 드롭 대상({len(low)}개): {sorted(low)}")

    df_new = df.drop(columns=low, errors="ignore")
    return df_new, imps, low


# =========================
# 5) 전체 파이프라인
# =========================
def run_preprocess(input_path="output/jeju.csv",
                        output_path="output/jeju_preprocessed.csv",
                        start_date="2014-01-01",
                        end_date="2025-04-30",
                        use_lag_roll=False,
                        impute_only_until=None,         # 예: "2025-01-09" (디코더 NaN 유지)
                        drop_external_pred_cols=True,   # 거의 전부 NaN 컬럼 자동 드롭
                        nan_ratio_drop=0.90,            # 결측비율 ≥ 90%면 드롭
                        corr_threshold=0.95,
                        rf_train_end="2025-01-09",
                        rf_importance_thresh=0.002):
    print("[시작] 제주 전처리 파이프라인 (23~25)")
    raw = pd.read_csv(input_path)
    print(f"- 입력 경로: {input_path}")
    print(f"- 원본 shape: {raw.shape}")

    # 1) 기본 전처리
    df = preprocess_basics(raw, start_date=start_date, end_date=end_date)
    print(f"- 기본 전처리 후 shape: {df.shape}")

    # 2) 이상치 마스킹 (→ NaN)
    zero_cols = [c for c in ["Vehicle_Max","Arrival_Num","Departure_Num","entry_count","exit_count",
                             "도민_Arr","도민_Dep","도민_입차","도민_출차",
                             "도민_Arrival","도민_Departure"] if c in df.columns]
    iqr_cols  = [c for c in ["Arrival_Num","Departure_Num","entry_count","exit_count","Late"] if c in df.columns]
    mad_cols  = [c for c in ["Arrival_Num","Departure_Num","entry_count","exit_count","Late","TempAvg_C"] if c in df.columns]

    df = mask_outliers(df,
                       zero_outlier_cols=zero_cols,        # 0/음수는 NaN 처리
                       iqr_cols=iqr_cols,                  # 롤링-IQR 이상치 NaN
                       mad_cols=mad_cols,                  # 롤링-MAD 이상치 NaN
                       iqr_window=35, iqr_k=2.5,
                       mad_window=35, mad_z=5.0,
                       past_only=True)

    # 3) 보간 전 NaN 요약
    num_cols = df.select_dtypes(include=[np.number]).columns
    na_before = df[num_cols].isna().sum()
    print("\n[3/6] 이상치 마스킹 후(보간 전) 숫자형 NaN 개수(상위 12)")
    print(na_before[na_before > 0].sort_values(ascending=False).head(12))

    # 4) 고급 보간
    imputer = ImputerAdvanced()
    imputer.fit(df)
    scope_index = None
    if impute_only_until is not None:
        until = pd.to_datetime(impute_only_until)
        scope_index = df.index[df.index <= until]
        print(f"- 보간 범위 제한: {impute_only_until} 까지만 (디코더 구간 NaN 유지)")
    df_imp = imputer.transform(df, limit_to_index=scope_index)

    # 5) (옵션) 거의 전부 NaN 컬럼 드롭 (외부 예측치 등)
    dropped_mostly_nan = []
    if drop_external_pred_cols:
        na_ratio = df_imp.isna().mean().sort_values(ascending=False)
        cand = na_ratio[na_ratio >= nan_ratio_drop].index.tolist()
        if cand:
            df_imp = df_imp.drop(columns=cand, errors="ignore")
            dropped_mostly_nan = cand
            print(f"\n[옵션] 결측비율 ≥ {nan_ratio_drop*100:.0f}% 컬럼 드롭: {sorted(cand)}")

    # 6) (옵션) 안전 lag/rolling
    if use_lag_roll:
        df_imp = add_lag_and_roll_safe(df_imp, target="Vehicle_Max", lags=(1,7,14), roll_windows=(7,14))
        print("- 안전 lag/rolling 추가: lag(1,7,14), roll_mean/std(7,14)")

    # 7) 다중공선성 제거
    df_mc, dropped_mc = remove_multicollinearity_numeric(df_imp, target="Vehicle_Max", threshold=corr_threshold)

    # 8) RF feature importance 필터
    df_final, imps, dropped_rf = rf_feature_filter(df_mc, target="Vehicle_Max",
                                                   train_end=rf_train_end,
                                                   importance_thresh=rf_importance_thresh)

    # 9) 최종 NaN 요약
    num_cols_after = df_final.select_dtypes(include=[np.number]).columns
    na_after = df_final[num_cols_after].isna().sum()
    print("\n[6/6] 전처리-최종 숫자형 NaN 개수(상위 12)")
    print(na_after[na_after > 0].sort_values(ascending=False).head(12))

    # 10) 샘플 출력
    show_cols = ["Vehicle_Max","Arrival_Num","Departure_Num","TempAvg_C","Precip_mm",
                 "LCC_ratio","FSC_ratio","DOM_Portion","INT_Portion",
                 "Late","entry_count","exit_count"]
    show_cols = [c for c in show_cols if c in df_final.columns]
    print("\n[샘플] head(5)")
    if show_cols:
        print(df_final.head(5)[show_cols].to_string())
        print("\n[샘플] tail(5)")
        print(df_final.tail(5)[show_cols].to_string())
    else:
        print("표시할 수치형 샘플 컬럼이 없습니다.")

    # 11) 드롭 요약
    print("\n[드롭 요약]")
    if dropped_mostly_nan:
        print(" - 결측비율 기준:", sorted(dropped_mostly_nan))
    if dropped_mc:
        print(" - 다중공선성 기준:", sorted(dropped_mc))
    if dropped_rf:
        print(" - RF 중요도 기준:", sorted(dropped_rf))
    if not (dropped_mostly_nan or dropped_mc or dropped_rf):
        print(" - 드롭된 컬럼 없음")

    print("\n[최종 컬럼 목록]")
    cols = df_final.columns.tolist()
    print(f"- 최종 컬럼 수: {len(cols)}")
    print("· 컬럼들:")
    for i, c in enumerate(cols, 1):
        print(f"  {i:2d}. {c}")
        
    # 12) 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.reset_index().to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n[완료] 저장: {output_path}")
    print("[끝] 제주 전처리 파이프라인 (23~25)")

    return df_final


# =========================
# 실행 예시
# =========================
# if __name__ == "__main__":
#     input_dir = "output"
#     output_dir = "output"

#     # input_dir 안의 csv 파일들 순회
#     for fname in os.listdir(input_dir):
#         if not fname.endswith(".csv"):
#             continue
#         if fname.endswith("_preprocessed.csv"):
#             continue   # 이미 전처리된 건 스킵

#         input_path = os.path.join(input_dir, fname)

#         # 확장자 제거 후 뒤에 _preprocessed 붙이기
#         base, _ = os.path.splitext(fname)
#         output_path = os.path.join(output_dir, f"{base}_preprocessed.csv")

#         print(f"\n=== {fname} 처리 시작 ===")
#         _ = run_preprocess(
#             input_path=input_path,
#             output_path=output_path,
#             start_date="2014-01-01",
#             end_date="2025-04-30",
#             use_lag_roll=False,
#             impute_only_until=None,
#             drop_external_pred_cols=True,
#             nan_ratio_drop=0.90,
#             corr_threshold=0.95,
#             rf_train_end="2025-01-09",
#             rf_importance_thresh=0.002
#         )
#         print(f"=== {fname} 처리 완료 ===\n")
if __name__ == "__main__": 
    _ = run_preprocess(input_path="output/jeju.csv", # ← 필요 시 경로 변경 
                            output_path="output/jeju_preprocessed.csv", 
                            start_date="2014-01-01", 
                            end_date="2025-04-30", 
                            use_lag_roll=False, # True로 켜면 안전 lag/rolling 생성 
                            impute_only_until=None, # 예: "2025-01-09" (디코더 NaN 유지) 
                            drop_external_pred_cols=True, # 거의 전부 NaN인 컬럼 자동 드롭 
                            nan_ratio_drop=0.90, # 90% 이상 NaN이면 드롭 
                            corr_threshold=0.95, # 다중공선성 임계 
                            rf_train_end="2025-01-09", # RF 학습 상한(누출 방지) 
                            rf_importance_thresh=0.002 # RF 중요도 하한(0.2%) 
                            )
