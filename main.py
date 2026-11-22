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
    page_title="다이아몬드 가격 예측 및 회귀 분석 대시보드",
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
    """원본 또는 정제 CSV를 불러와 데이터 정제 후 이력 반환"""
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
    """전처리 및 8:2 층화 분할 후 OLS 선형회귀 모델 학습"""
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
    """단일 다이아몬드 입력 스펙의 달러($) 예측값 및 신뢰 구간 계산"""
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
    obs_ci_high = float(pred_res["obs_ci_upper"].iloc[0])

    pred_dollar = np.exp(pred_log) * smearing_factor
    ci_low_dollar = np.exp(obs_ci_low) * smearing_factor
    ci_high_dollar = np.exp(obs_ci_high) * smearing_factor

    return pred_dollar, ci_low_dollar, ci_high_dollar


# --- 메인 타이틀 및 데이터 로드 ---
st.title("💎 다이아몬드 가격 예측 및 선형회귀 분석 웹앱")
st.markdown(
    "원본/정제 데이터를 기반으로 **다중공선성 제거(`x,y,z` 제외) · "
    "로그 변환 · 원-핫 인코딩 · Duan 편향 보정**을 적용한 대시보드입니다."
)

# 사이드바: 데이터 소스 선택
with st.sidebar:
    st.header("📂 데이터 소스 설정")
    uploaded_file = st.file_uploader(
        "CSV 파일 직접 업로드 (선택)", type=["csv"]
    )
    st.caption(
        "업로드하지 않으면 기본 데이터(`diamonds_cleaned.csv`)로 구동됩니다."
    )

try:
    data_source = (
        uploaded_file
        if uploaded_file is not None
        else "diamonds_cleaned.csv"
    )
    df_clean, clean_log = load_and_clean_data(data_source)
    ols_model, feature_cols, metrics, eval_data = train_ols_model(
        df_clean
    )
except FileNotFoundError:
    st.error(
        "저장소에 `diamonds_cleaned.csv` 파일이 없습니다. "
        "GitHub에 파일을 올리거나 사이드바에서 CSV를 선택해 주세요."
    )
    st.stop()

# 상단 핵심 지표 요약
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
removed_cnt = clean_log["initial"] - clean_log["final"]
col_m1.metric(
    "최종 분석 데이터",
    f"{clean_log['final']:,} 건",
    f"제거 {removed_cnt}건",
)
col_m2.metric(
    "Test R² (로그 / 달러)",
    f"{metrics['r2_log']:.4f} / {metrics['r2_dollar']:.4f}",
)
col_m3.metric(
    "Test MAE (평균 오차)",
    f"${metrics['mae']:,.2f}",
)
col_m4.metric(
    "Duan 보정 계수",
    f"{metrics['smearing_factor']:.5f}",
)

st.divider()

# 3개 탭 구성
tab1, tab2, tab3 = st.tabs(
    [
        "🔮 1. 다이아몬드 가격 예측기",
        "📊 2. 회귀 계수 및 4C 영향력 분석",
        "🩺 3. 데이터 정제 및 잔차 진단",
    ]
)

# ==========================================
# TAB 1: 실시간 다이아몬드 가격 예측기
# ==========================================
with tab1:
    st.subheader("다이아몬드 4C 및 비율 스펙 입력")
    c1, c2, c3 = st.columns(3)

    with c1:
        carat = st.slider(
            "캐럿 (Carat)",
            min_value=0.20,
            max_value=4.00,
            value=1.00,
            step=0.01,
        )
        cut = st.selectbox(
            "컷 등급 (Cut)", options=CUT_ORDER, index=4
        )

    with c2:
        color = st.selectbox(
            "컬러 등급 (Color: D=최상, J=하)",
            options=COLOR_ORDER,
            index=3,
        )
        clarity = st.selectbox(
            "투명도 등급 (Clarity: IF=최상, I1=하)",
            options=CLARITY_ORDER,
            index=4,
        )

    with c3:
        depth = st.slider(
            "깊이 비율 (Depth %)",
            min_value=50.0,
            max_value=72.0,
            value=61.8,
            step=0.1,
        )
        table = st.slider(
            "테이블 비율 (Table %)",
            min_value=50.0,
            max_value=70.0,
            value=57.0,
            step=0.5,
        )

    pred_dollar, ci_low, ci_high = predict_single_diamond(
        ols_model,
        feature_cols,
        metrics["smearing_factor"],
        carat,
        cut,
        color,
        clarity,
        depth,
        table,
    )

    st.markdown("### 예측 결과")
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric(
        "예상 적정 가격 (Duan 보정)", f"${pred_dollar:,.0f}"
    )
    res_col2.metric(
        "캐럿당 단가 ($ / ct)", f"${pred_dollar / carat:,.0f} / ct"
    )
    res_col3.metric(
        "95% 예측 구간 (개별 시세 범위)",
        f"${ci_low:,.0f} ~${ci_high:,.0f}",
    )

    if carat > 2.5:
        st.warning(
            "⚠️ 2.5ct 초과 대형 다이아몬드는 데이터 상한선($18,823) "
            "제약으로 실제 시세보다 높게 예측될 수 있습니다."
        )

    # 유사 캐럿 구간 내 시장 가격 분포 비교 차트
    low_c = carat - 0.05
    high_c = carat + 0.05
    similar_df = df_clean[
        (df_clean["carat"] >= low_c) & (df_clean["carat"] <= high_c)
    ]
    if len(similar_df) >= 10:
        fig_sim, ax_sim = plt.subplots(figsize=(9, 3.5))
        mkt_label = (
            f"Market ({low_c:.2f}~{high_c:.2f}ct, "
            f"n={len(similar_df):,})"
        )
        ax_sim.hist(
            similar_df["price"],
            bins=35,
            color="#4a7c9b",
            alpha=0.75,
            edgecolor="white",
            label=mkt_label,
        )
        ax_sim.axvline(
            pred_dollar,
            color="#d9534f",
            linestyle="--",
            linewidth=2.2,
            label=f"Predicted: ${pred_dollar:,.0f}",
        )
        med_val = similar_df["price"].median()
        ax_sim.axvline(
            med_val,
            color="#f0ad4e",
            linestyle=":",
            linewidth=2,
            label=f"Market Median: ${med_val:,.0f}",
        )
        chart_title = f"Price Distribution Around {carat:.2f} Carat"
        ax_sim.set_title(
            chart_title,
            fontsize=11,
            fontweight="bold",
        )
        ax_sim.set_xlabel("Price ($)")
        ax_sim.set_ylabel("Count")
        ax_sim.legend()
        plt.tight_layout()
        st.pyplot(fig_sim)


# ==========================================
# TAB 2: 회귀 계수 및 4C 영향력 분석
# ==========================================
with tab2:
    st.subheader(
        "등급별 기준 대비 가격 상승률 (%) — 기준: Fair / J / I1"
    )

    coef_df = pd.DataFrame(
        {
            "Coef": ols_model.params,
            "Std_Err": ols_model.bse,
            "t_stat": ols_model.tvalues,
            "p_value": ols_model.pvalues,
        }
    )
    coef_df["Price_Lift (%)"] = (
        np.exp(coef_df["Coef"]) - 1
    ) * 100
    coef_df["Multiplier (x)"] = np.exp(coef_df["Coef"])

    # 범주형 계수 시각화
    prefixes = ("cut_", "color_", "clarity_")
    cat_coefs = coef_df[
        coef_df.index.str.startswith(prefixes)
    ].copy()
    cat_coefs["Label"] = (
        cat_coefs.index.str.replace("cut_", "Cut: ")
        .str.replace("color_", "Color: ")
        .str.replace("clarity_", "Clarity: ")
    )
    colors_bar = [
        "#2b5c8f"
        if x.startswith("Clarity")
        else ("#3b9ab2" if x.startswith("Color") else "#78b7c5")
        for x in cat_coefs["Label"]
    ]

    fig_coef, ax_coef = plt.subplots(figsize=(10, 5.5))
    bars = ax_coef.barh(
        cat_coefs["Label"],
        cat_coefs["Price_Lift (%)"],
        color=colors_bar,
    )
    ax_coef.set_title(
        "Percentage Price Increase vs. Baseline (Fair / J / I1)",
        fontsize=12,
        fontweight="bold",
    )
    ax_coef.set_xlabel("Price Increase (%)")
    ax_coef.grid(axis="x", linestyle="--", alpha=0.5)
    for bar in bars:
        w = bar.get_width()
        ax_coef.text(
            w + 2,
            bar.get_y() + bar.get_height() / 2,
            f"+{w:.1f}%",
            va="center",
            fontsize=8.5,
        )
    ax_coef.set_xlim(0, cat_coefs["Price_Lift (%)"].max() * 1.15)
    plt.tight_layout()
    st.pyplot(fig_coef)

    st.markdown("#### 전체 OLS 회귀 계수 상세 표")
    fmt_dict = {
        "Coef": "{:.4f}",
        "Std_Err": "{:.4f}",
        "t_stat": "{:.2f}",
        "p_value": "{:.4f}",
        "Price_Lift (%)": "{:+.2f}%",
        "Multiplier (x)": "{:.2f}x",
    }
    st.dataframe(
        coef_df.style.format(fmt_dict),
        use_container_width=True,
    )


# ==========================================
# TAB 3: 데이터 정제 요약 및 잔차 진단
# ==========================================
with tab3:
    st.subheader("1. 데이터 정제(Cleaning) 처리 이력")
    clean_summary = pd.DataFrame(
        [
            {
                "단계": "원본 데이터 로드",
                "제거 건수": 0,
                "잔여 행 수": clean_log["initial"],
                "비고": "초기 입력 데이터",
            },
            {
                "단계": "1차: 완전 중복 제거",
                "제거 건수": clean_log["removed_dup"],
                "잔여 행 수": clean_log["after_dup"],
                "비고": "데이터 누수(Leakage) 방지",
            },
            {
                "단계": "2차: 치수(x,y,z) 0값 제거",
                "제거 건수": clean_log["removed_zero"],
                "잔여 행 수": clean_log["after_zero
