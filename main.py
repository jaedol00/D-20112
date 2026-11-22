import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)

# 페이지 기본 설정
st.set_page_config(
    page_title="다이아몬드 가격 예측 대시보드",
    page_icon="💎",
    layout="wide",
)

plt.rcParams["axes.unicode_minus"] = False

# 도메인 서열 정의 (최하위 등급이 기준 범주)
CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]
CLARITY_ORDER = [
    "I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"
]


@st.cache_data
def load_and_clean_data(file_path_or_buffer):
    # 원본 또는 정제 CSV를 불러와 데이터 정제 후 이력 반환
    raw_df = pd.read_csv(file_path_or_buffer)
    if "Unnamed: 0" in raw_df.columns:
        raw_df = raw_df.drop(columns=["Unnamed: 0"])

    log = {"initial": len(raw_df)}

    # 1. 중복 행 제거
    df = raw_df.drop_duplicates().reset_index(drop=True)
    log["after_dup"] = len(df)
    log["removed_dup"] = log["initial"] - log["after_dup"]

    # 2. 물리적 치수(x, y, z) 0값 제거
    zero_mask = (df["x"] <= 0) | (df["y"] <= 0) | (df["z"] <= 0)
    df = df[~zero_mask].reset_index(drop=True)
    log["after_zero"] = len(df)
    log["removed_zero"] = log["after_dup"] - log["after_zero"]

    # 3. 명백한 물리적 오기입 극단치 제거
    calc_depth = 2 * df["z"] / (df["x"] + df["y"]) * 100
    depth_diff = (df["depth"] - calc_depth).abs()
    outlier_mask = (
        (df["y"] > 15)
        | (df["z"] > 15)
        | (df["z"] < 2.0)
        | (depth_diff > 30)
        | (df["table"] > 90)
    )
    df = df[~outlier_mask].reset_index(drop=True)
    log["final"] = len(df)
    log["removed_outlier"] = log["after_zero"] - log["final"]

    return df, log


@st.cache_resource
def train_ols_model(df: pd.DataFrame):
    # 전처리 및 8:2 층화 분할 후 OLS 선형회귀 모델 학습
    work_df = df.drop(columns=["x", "y", "z"]).copy()
    work_df["log_carat"] = np.log(work_df["carat"])
    work_df["log_price"] = np.log(work_df["price"])

    work_df["cut"] = pd.Categorical(
        work_df["cut"], categories=CUT_ORDER, ordered=True
    )
    work_df["color"] = pd.Categorical(
        work_df["color"], categories=COLOR_ORDER, ordered=True
    )
    work_df["clarity"] = pd.Categorical(
        work_df["clarity"], categories=CLARITY_ORDER, ordered=True
    )

    encoded_df = pd.get_dummies(
        work_df,
        columns=["cut", "color", "clarity"],
        drop_first=True,
        dtype=int,
    )

    exclude_cols = ["carat", "price", "log_price"]
    feature_cols = [
        c for c in encoded_df.columns if c not in exclude_cols
    ]

    # 8:2 가격 10분위 층화 분할
    price_bins = pd.qcut(encoded_df["price"], q=10, labels=False)
    train_df, test_df = train_test_split(
        encoded_df,
        test_size=0.2,
        random_state=42,
        stratify=price_bins,
    )

    X_train = sm.add_constant(
        train_df[feature_cols], has_constant="add"
    )
    y_train_log = train_df["log_price"]

    X_test = sm.add_constant(
        test_df[feature_cols], has_constant="add"
    )
    y_test_log = test_df["log_price"]
    y_test_dollar = test_df["price"]

    ols_model = sm.OLS(y_train_log, X_train).fit()

    # Duan's Smearing 역변환 편향 보정 계수
    train_pred_log = ols_model.predict(X_train)
    resid_train = y_train_log - train_pred_log
    smearing_factor = float(np.mean(np.exp(resid_train)))

    # 테스트 세트 평가
    test_pred_log = ols_model.predict(X_test)
    test_pred_dollar = np.exp(test_pred_log) * smearing_factor

    metrics = {
        "train_n": len(train_df),
        "test_n": len(test_df),
        "r2_log": r2_score(y_test_log, test_pred_log),
        "r2_dollar": r2_score(y_test_dollar, test_pred_dollar),
        "mae": mean_absolute_error(
            y_test_dollar, test_pred_dollar
        ),
        "rmse": root_mean_squared_error(
            y_test_dollar, test_pred_dollar
        ),
        "mape": mean_absolute_percentage_error(
            y_test_dollar, test_pred_dollar
        ) * 100,
        "smearing_factor": smearing_factor,
    }

    eval_data = {
        "y_test_dollar": y_test_dollar.values,
        "test_pred_dollar": test_pred_dollar.values,
        "test_pred_log": test_pred_log.values,
        "resid_log": (y_test_log - test_pred_log).values,
    }

    return ols_model, feature_cols, metrics, eval_data


def predict_single_diamond(
    model,
    feature_cols,
    smearing_factor,
    carat,
    cut,
    color,
    clarity,
    depth,
    table,
):
    # 단일 다이아몬드 입력 스펙의 달러 예측값 및 신뢰 구간 계산
    row = pd.DataFrame(
        {
            "depth": [depth],
            "table": [table],
            "log_carat": [np.log(carat)],
            "cut": pd.Categorical(
                [cut], categories=CUT_ORDER, ordered=True
            ),
            "color": pd.Categorical(
                [color], categories=COLOR_ORDER, ordered=True
            ),
            "clarity": pd.Categorical(
                [clarity], categories=CLARITY_ORDER, ordered=True
            ),
        }
    )
    row_encoded = pd.get_dummies(
        row,
        columns=["cut", "color", "clarity"],
        drop_first=True,
        dtype=int,
    )
    row_encoded = row_encoded.reindex(
        columns=feature_cols, fill_value=0
    )
    row_sm = sm.add_constant(row_encoded, has_constant="add")

    pred_res = model.get_prediction(row_sm).summary_frame(
        alpha=0.05
    )
    pred_log = float(pred_res["mean"].iloc[0])
    obs_ci_low = float(pred_res["obs_ci_lower"].iloc[0])
    obs_ci_high = float(pred
