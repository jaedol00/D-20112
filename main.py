from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="다이아몬드 가격 예측 & 회귀분석 대시보드",
    page_icon="💎",
    layout="wide",
)

CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]
CLARITY_ORDER = [
    "I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"
]


def _prepare_train_test(df: pd.DataFrame):
    # 단일 CSV 입력 시 정제, 변수 선택, 인코딩, 8:2 층화 분할 수행
    df = df.copy()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # 원본 데이터인 경우 정제 수행
    if all(col in df.columns for col in ["x", "y", "z"]):
        df = df.drop_duplicates().reset_index(drop=True)
        mask_pos = (df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)
        df = df[mask_pos].reset_index(drop=True)
        mask_out = (df["y"] <= 20) & (df["z"] <= 10)
        df = df[mask_out].reset_index(drop=True)
        df = df.drop(columns=["x", "y", "z"])

    # depth, table 컬럼이 있으면 제거 (log_carat + 3C 구조 통일)
    for drop_col in ["depth", "table"]:
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])

    # 로그 변환
    if "log_carat" not in df.columns:
        df["log_carat"] = np.log(df["carat"])
    if "log_price" not in df.columns:
        df["log_price"] = np.log(df["price"])

    # 범주형 원-핫 인코딩
    if "cut" in df.columns:
        df["cut"] = pd.Categorical(
            df["cut"], categories=CUT_ORDER, ordered=True
        )
        df["color"] = pd.Categorical(
            df["color"], categories=COLOR_ORDER, ordered=True
        )
        df["clarity"] = pd.Categorical(
            df["clarity"], categories=CLARITY_ORDER, ordered=True
        )
        df = pd.get_dummies(
            df,
            columns=["cut", "color", "clarity"],
            drop_first=True,
            dtype=int,
        )

    # 8:2 가격 10분위 층화 분할
    price_bins = pd.qcut(df["price"], q=10, labels=False)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=price_bins
    )
    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# 2. 데이터 로드 및 모델 학습 (캐싱 적용)
@st.cache_data
def load_data(uploaded_file=None):
    # 1순위: 사이드바 직접 업로드 CSV
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        return _prepare_train_test(raw_df)

    # 2순위: 이미 분할된 train/test CSV 파일 탐색
    pair_candidates = [
        ("train.csv", "test.csv"),
        ("diamonds_train.csv", "diamonds_test.csv"),
    ]
    for tr_file, te_file in pair_candidates:
        if Path(tr_file).exists() and Path(te_file).exists():
            return pd.read_csv(tr_file), pd.read_csv(te_file)

    # 3순위: 단일 CSV 파일 자동 탐색 (정제본 또는 원본)
    single_candidates = [
        "diamonds_cleaned_v2.csv",
        "diamonds_cleaned.csv",
        "diamonds_preprocessed_final.csv",
        "diamonds.csv",
    ]
    for fname in single_candidates:
        if Path(fname).exists():
            raw_df = pd.read_csv(fname)
            return _prepare_train_test(raw_df)

    return None, None


@st.cache_resource
def train_and_evaluate(train_df, test_df):
    exclude = ["carat", "price", "log_price"]
    feature_cols = [
        c for c in train_df.columns if c not in exclude
    ]

    X_train = train_df[feature_cols]
    y_train_log = train_df["log_price"].values
    X_test = test_df[feature_cols]
    y_test_log = test_df["log_price"].values
    y_test_actual = test_df["price"].values

    # 데이터 누수 방지를 위해 Train 세트에서만 스케일러 학습
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5-Fold 교차 검증 기반 RidgeCV 학습
    alphas = np.logspace(-3, 3, 50)
    ridge_cv = RidgeCV(
        alphas=alphas, cv=5, scoring="neg_mean_squared_error"
    )
    ridge_cv.fit(X_train_scaled, y_train_log)

    # 비즈니스 해석(가격 상승 배수 %)용 비표준화 OLS 회귀식
    ols_raw = LinearRegression()
    ols_raw.fit(X_train, y_train_log)

    # 예측 및 달러($) 환산
    pred_train_log = ridge_cv.predict(X_train_scaled)
    pred_test_log = ridge_cv.predict(X_test_scaled)
    pred_train_dollar = np.exp(pred_train_log)
    pred_test_dollar = np.exp(pred_test_log)

    metrics = {
        "best_alpha": float(ridge_cv.alpha_),
        "train_r2": r2_score(
            train_df["price"].values, pred_train_dollar
        ),
        "test_r2": r2_score(y_test_actual, pred_test_dollar),
        "test_r2_log": r2_score(y_test_log, pred_test_log),
        "rmse": root_mean_squared_error(
            y_test_actual, pred_test_dollar
        ),
        "mae": mean_absolute_error(
            y_test_actual, pred_test_dollar
        ),
        "mape": mean_absolute_percentage_error(
            y_test_actual, pred_test_dollar
        ) * 100,
    }

    # 회귀 계수 정리
    coef_df = pd.DataFrame(
        {
            "Feature": feature_cols,
            "Standardized_Coef": ridge_cv.coef_,
            "Raw_Coef": ols_raw.coef_,
        }
    )
    coef_df["Premium_Pct"] = np.where(
        coef_df["Feature"] == "log_carat",
        np.nan
