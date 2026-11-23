import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

# 2. 데이터 로드 및 모델 학습 (캐싱 적용)
@st.cache_data
def load_data():
    train_df = pd.read_csv("diamonds_train.csv")
    test_df = pd.read_csv("diamonds_test.csv")
    return train_df, test_df

@st.cache_resource
def train_and_evaluate(train_df, test_df):
    feature_cols = [c for c in train_df.columns if c not in ["carat", "price", "log_price"]]

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
    ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error")
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
        "train_r2": r2_score(train_df["price"].values, pred_train_dollar),
        "test_r2": r2_score(y_test_actual, pred_test_dollar),
        "test_r2_log": r2_score(y_test_log, pred_test_log),
        "rmse": root_mean_squared_error(y_test_actual, pred_test_dollar),
        "mae": mean_absolute_error(y_test_actual, pred_test_dollar),
        "mape": mean_absolute_percentage_error(y_test_actual, pred_test_dollar) * 100,
    }

    # 회귀 계수 정리
    coef_df = pd.DataFrame({
        "Feature": feature_cols,
        "Standardized_Coef": ridge_cv.coef_,
        "Raw_Coef": ols_raw.coef_,
    })
    coef_df["Premium_Pct"] = np.where(
        coef_df["Feature"] == "log_carat",
        np.nan,
        (np.exp(coef_df["Raw_Coef"]) - 1) * 100,
    )

    return ridge_cv, scaler, feature_cols, metrics, coef_df, pred_test_dollar

try:
    train_df, test_df = load_data()
except FileNotFoundError:
    st.error("`train.csv` 또는 `test.csv` 파일을 찾을 수 없습니다. GitHub 저장소 루트 경로에 두 파일을 업로드해 주세요.")
    st.stop()

ridge_model, scaler, feature_cols, metrics, coef_df, pred_test_dollar = train_and_evaluate(train_df, test_df)

# 3. 사이드바: 실시간 가격 예측 시뮬레이터 입력
st.sidebar.header("💎 실시간 다이아몬드 가격 예측")
st.sidebar.markdown("원하는 **4C 스펙**을 선택하면 학습된 선형회귀 모델이 즉시 예상 달러($) 가격을 계산합니다.")

input_carat = st.sidebar.slider("무게 (Carat)", min_value=0.20, max_value=5.00, value=1.00, step=0.01)

cut_options = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
input_cut = st.sidebar.selectbox("컷팅 등급 (Cut)", cut_options, index=4)

color_options = ["J", "I", "H", "G", "F", "E", "D"]
input_color = st.sidebar.selectbox("색상 등급 (Color: J → D 최상급)", color_options, index=5)

clarity_options = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
input_clarity = st.sidebar.selectbox("투명도 등급 (Clarity: I1 → IF 최상급)", clarity_options, index=4)

# 입력값을 모델 피처 벡터(18열)로 변환
input_dict = {col: 0.0 for col in feature_cols}
input_dict["log_carat"] = np.log(input_carat)
if input_cut != "Fair":
    input_dict[f"cut_{input_cut}"] = 1.0
if input_color != "J":
    input_dict[f"color_{input_color}"] = 1.0
if input_clarity != "I1":
    input_dict[f"clarity_{input_clarity}"] = 1.0

input_df = pd.DataFrame([input_dict])[feature_cols]
input_scaled = scaler.transform(input_df)
pred_single_log = ridge_model.predict(input_scaled)[0]
pred_single_dollar = np.exp(pred_single_log)

# ±10.43% (MAPE 기반 참고 범위)
low_est = pred_single_dollar * (1 - metrics["mape"] / 100)
high_est = pred_single_dollar * (1 + metrics["mape"] / 100)

st.sidebar.markdown("---")
st.sidebar.metric(label="예상 시장 가격 (USD)", value=f"${pred_single_dollar:,.2f}")
st.sidebar.caption(f"오차율(MAPE ±{metrics['mape']:.1f}%) 감안 범위: **${low_est:,.0f} ~ ${high_est:,.0f}**")

# 4. 메인 화면 구성
st.title("💎 다이아몬드 가격 예측 및 선형회귀 분석 리포트")
st.markdown(
    "정제된 **53,772건**의 다이아몬드 데이터를 **8:2 층화 분할(`train.csv` / `test.csv`)**하여 "
    "구축한 **로그 변환 Ridge 선형회귀 모델**의 상세 전처리 과정, 회귀 계수 분석 및 실시간 예측 대시보드입니다."
)

# 핵심 성과 지표 (KPI)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("테스트 결정계수 (R²)", f"{metrics['test_r2']:.4f}", f"로그 R²: {metrics['test_r2_log']:.4f}")
kpi2.metric("평균 절대 오차 (MAE)", f"${metrics['mae']:,.2f}")
kpi3.metric("평균 제곱근 오차 (RMSE)", f"${metrics['rmse']:,.2f}")
kpi4.metric("평균 오차율 (MAPE)", f"{metrics['mape']:.2f}%", f"최적 Alpha: {metrics['best_alpha']:.4f}")

st.markdown("---")

# 3개 핵심 분석 탭 구성
tab1, tab2, tab3 = st.tabs([
    "🔮 시뮬레이터 & 등급 비교",
    "🛠️ 상세 데이터 전처리 파이프라인",
    "📈 모델 평가 & 회귀 계수 분석",
])

# --- TAB 1: 시뮬레이터 & 동일 조건 등급별 가격 변화 ---
with tab1:
    st.subheader("선택한 다이아몬드 스펙 요약 및 등급 변화 시뮬레이션")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.info(
            f"**현재 선택 스펙**\n"
            f"- **무게 (Carat):** `{input_carat:.2f} ct`\n"
            f"- **컷팅 (Cut):** `{input_cut}`\n"
            f"- **색상 (Color):** `{input_color}`\n"
            f"- **투명도 (Clarity):** `{input_clarity}`\n\n"
            f"### 예측 가격: **${pred_single_dollar:,.2f}**"
        )
        st.markdown(
            "선형회귀 모델에서 `log(carat)`의 회귀 계수는 **1.8846**입니다. "
            "동일 등급에서 무게가 **2배** 커지면 가격은 약 **3.69배** 상승합니다."
        )
    with col_b:
        sim_rows = []
        for cl in clarity_options:
            tmp = {col: 0.0 for col in feature_cols}
            tmp["log_carat"] = np.log(input_carat)
            if input_cut != "Fair":
                tmp[f"cut_{input_cut}"] = 1.0
            if input_color != "J":
                tmp[f"color_{input_color}"] = 1.0
            if cl != "I1":
                tmp[f"clarity_{cl}"] = 1.0
            p_val = np.exp(ridge_model.predict(scaler.transform(pd.DataFrame([tmp])[feature_cols]))[0])
            sim_rows.append({
                "Clarity 등급": cl,
                "예상 가격 ($)": p_val,
                "선택 여부": "현재 선택" if cl == input_clarity else "다른 등급",
            })

        sim_df = pd.DataFrame(sim_rows)
        fig_sim = px.bar(
            sim_df,
            x="Clarity 등급",
            y="예상 가격 ($)",
            color="선택 여부",
            text_auto=",.0f",
            title=f"{input_carat:.2f}ct / {input_cut} / {input_color} 등급 기준 Clarity별 예상 가격 ($)",
            color_discrete_map={"현재 선택": "#d95f02", "다른 등급": "#2b5c8f"},
        )
        fig_sim.update_layout(showlegend=True, yaxis_title="예상 가격 (USD)")
        st.plotly_chart(fig_sim, use_container_width=True)

# --- TAB 2: 상세 데이터 전처리 파이프라인 ---
with tab2:
    st.subheader("1단계. 데이터 정제 (Data Cleaning): 총 168건 오류·이상치 제거")
    st.markdown(
        "원본 `diamonds.csv`(**53,940행 × 10열**)에서 결측치(`NaN`)는 0건이었으나, "
        "모델 학습을 왜곡하는 중복 데이터와 물리적 오류 값을 3단계에 걸쳐 제거하여 **53,772행**을 확보했습니다."
    )

    clean_summary = pd.DataFrame({
        "정제 단계": [
            "원본 데이터 (Raw Dataset)",
            "① 완전 중복 데이터 제거",
            "② 물리적 치수 0 오류 제거",
            "③ 극단적 치수 오기입 제거",
            "최종 정제 완료 데이터",
        ],
        "적용 조건 (Python 기준)": [
            "diamonds.csv 초기 로드",
            "df.drop_duplicates(keep='first')",
            "(x == 0) | (y == 0) | (z == 0)",
            "(y > 20) | (z > 10)",
            "diamonds_cleaned_v2.csv",
        ],
        "제거 행 수": [
            "-",
            "-146행",
            "-19행",
            "-3행",
            "총 -168행 (0.31% 제거)",
        ],
        "남은 행 수": [
            "53,940행",
            "53,794행",
            "53,775행",
            "53,772행",
            "53,772행",
        ],
        "제거 사유 및 통계적 근거": [
            "10개 컬럼(수치형 7, 범주형 3)으로 구성",
            "동일 데이터가 Train과 Test에 동시에 들어가 성능을 과대평가하는 데이터 누수(Data Leakage) 차단",
            "길이(x), 너비(y), 높이(z) 중 하나라도 0mm인 측정 누락 오류 데이터 제거",
            "y=58.9mm, y=31.8mm, z=31.8mm 등 소수점 오기입으로 회귀선을 크게 기울게 만드는 극단치 제거",
            "깨끗한 물리적·통계적 정합성 확보 완료",
        ],
    })
    st.dataframe(clean_summary, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("2단계. 다중공선성(VIF) 해소 및 핵심 변수 선택 (Feature Selection)")
    st.markdown(
        "무게(`carat`)와 물리적 치수(`x, y, z`)는 모두 다이아몬드 크기를 나타내므로 **심각한 다중공선성(VIF 435~495)**을 유발합니다. "
        "중복 변수를 제거하고 **`log(carat)` + 3C 등급(`cut, color, clarity`)만 사용**하여 모델을 경량화·안정화했습니다."
    )

    col_vif1, col_vif2 = st.columns([1.1, 1])
    with col_vif1:
        vif_table = pd.DataFrame({
            "변수명": ["carat (무게)", "depth (깊이 비율)", "table (상단 너비 비율)", "x (길이 mm)", "y (너비 mm)", "z (높이 mm)"],
            "전체 6개 투입 시 VIF": ["24.61 (위험)", "7.69", "1.17", "490.85 (극심한 중복)", "494.89 (극심한 중복)", "435.26 (극심한 중복)"],
            "x, y, z 제거 후 VIF": ["1.04 (완벽히 독립)", "1.11 (완벽히 독립)", "1.14 (완벽히 독립)", "제거됨", "제거됨", "제거됨"],
            "최종 모델 채택 여부": ["✅ log(carat)으로 변환 채택", "❌ 영향력 미미(-0.001) 제외", "❌ 영향력 미미(-0.001) 제외", "❌ 다중공선성 원인으로 제거", "❌ 다중공선성 원인으로 제거", "❌ 다중공선성 원인으로 제거"],
        })
        st.dataframe(vif_table, use_container_width=True, hide_index=True)

    with col_vif2:
        st.info(
            "**변수 선택 핵심 인사이트**\n"
            "- **다중공선성 해결:** 전체 변수 투입 시 `carat` 계수가 비정상적인 음수(`-0.48`)로 뒤집혔으나, `x, y, z` 제거 후 VIF가 **1.04**로 안정화되었습니다.\n"
            "- **경량화 효과:** 수치형 피처를 6개에서 `log(carat)` **1개**로 줄여도 평균 오차(MAE)는 오히려 **$410 → $405**로 개선되었습니다."
        )

    st.markdown("---")
    st.subheader("3단계. 변수 변환 (Log Transform), 인코딩 및 스케일링")
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.markdown("**① 양변 자연로그 변환 (`log_carat`, `log_price`)**")
        st.markdown(
            "- **곡선 → 직선화:** 기하급수적으로 상승하는 `carat-price` 관계를 직선($r = 0.97$)으로 변환하여 선형회귀 적합도 극대화\n"
            "- **비율(%) 기반 오차 학습:** 저가($500)와 고가($15,000) 모두 동일한 퍼센트 오차 기준으로 공평하게 학습\n"
            "- **복원:** 예측 후 `np.exp()`로 실제 달러($) 환산"
        )
    with col_t2:
        st.markdown("**② 범주형 원-핫 인코딩 (`OneHotEncoder`)**")
        st.markdown(
            "- **GIA 서열 기준 설정:** 각 등급의 최하위(`Fair`, `J`, `I1`)를 기준점(`drop='first'`)으로 설정하여 총 **17개 더미 변수** 생성\n"
            "- **비선형 프리미엄 반영:** 등급 간 가격 상승폭이 일정하지 않은 특성을 개별 회귀 계수로 정밀 학습"
        )
    with col_t3:
        st.markdown("**③ 표준화 스케일링 (`StandardScaler`)**")
        st.markdown(
            "- **규제 페널티 균형:** 연속형(`log_carat`)과 이진 더미(0/1) 변수 간 스케일 차이를 맞추어 Ridge L2 규제가 공평하게 적용되도록 처리\n"
            "- **데이터 누수 방지:** 반드시 `train.csv`에서만 `fit()` 수행 후 `test.csv`에 `transform()` 적용"
        )

    st.markdown("---")
    st.subheader("4단계. 8:2 층화 분할 (Stratified Split) 및 전처리 전·후 분포 검증")
    st.markdown(
        "가격(`price`)의 **10분위수(`pd.qcut(..., q=10)`)**를 기준으로 층화 분할을 적용해 "
        "**`train.csv`(43,017행, 80%)**와 **`test.csv`(10,755행, 20%)**에 저가~초고가 다이아몬드가 동일한 비율로 배분되도록 했습니다."
    )

    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(
            pd.concat([
                train_df[["price"]].assign(Dataset="Train (80%, 43,017행)"),
                test_df[["price"]].assign(Dataset="Test (20%, 10,755행)"),
            ]),
            x="price",
            color="Dataset",
            barmode="overlay",
            nbins=50,
            histnorm="probability density",
            title="Train vs Test 가격 밀도 분포 일치 확인 (평균: Train $3,930 vs Test $3,937)",
            color_discrete_sequence=["#2b5c8f", "#d95f02"],
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        sample_viz = train_df.sample(2500, random_state=42)
        fig_scatter = px.scatter(
            sample_viz,
            x="log_carat",
            y="log_price",
            opacity=0.4,
            title="로그 변환 후 선형성 확보: log(Carat) vs log(Price) (r = 0.97)",
            labels={"log_carat": "log(Carat)", "log_price": "log(Price)"},
            color_discrete_sequence=["#1f8a70"],
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with st.expander("📄 최종 전처리 완료 데이터셋 미리보기 (train.csv 상위 10행, 총 21개 컬럼)"):
        st.dataframe(train_df.head(10), use_container_width=True)

# --- TAB 3: 모델 평가 & 회귀 계수 분석 ---
with tab3:
    st.subheader("1. 테스트 세트(10,755행) 실제 가격 vs 예측 가격")
    col_m1, col_m2 = st.columns([1.3, 1])

    with col_m1:
        eval_sample = pd.DataFrame({
            "Actual ($)": test_df["price"].values,
            "Predicted ($)": pred_test_dollar,
            "Carat": test_df["carat"].values,
        }).sample(3000, random_state=42)

        fig_pred = px.scatter(
            eval_sample,
            x="Actual ($)",
            y="Predicted ($)",
            opacity=0.35,
            title=f"실제 가격 vs 예측 가격 (Test R² = {metrics['test_r2']:.4f})",
            color_discrete_sequence=["#2b5c8f"],
        )
        fig_pred.add_trace(
            go.Scatter(x=[0, 20000], y=[0, 20000], mode="lines", name="완벽 예측선 (y=x)", line=dict(color="red", dash="dash"))
        )
        st.plotly_chart(fig_pred, use_container_width=True)

    with col_m2:
        eval_df = pd.DataFrame({"price": test_df["price"].values, "pred": pred_test_dollar})
        eval_df["abs_err"] = np.abs(eval_df["price"] - eval_df["pred"])
        eval_df["pct_err"] = eval_df["abs_err"] / eval_df["price"] * 100
        eval_df["구간"] = pd.cut(
            eval_df["price"],
            bins=[0, 1000, 3000, 7000, 20000],
            labels=["~$1,000 (저가)", "$1,000~$3,000", "$3,000~$7,000", "$7,000+ (초고가)"],
        )
        bracket_df = (
            eval_df.groupby("구간", observed=False)
            .agg(샘플수=("price", "count"), 평균실제가=("price", "mean"), MAE=("abs_err", "mean"), MAPE=("pct_err", "mean"))
            .reset_index()
        )
        bracket_df["평균실제가"] = bracket_df["평균실제가"].map(lambda x: f"${x:,.0f}")
        bracket_df["MAE"] = bracket_df["MAE"].map(lambda x: f"${x:,.0f}")
        bracket_df["MAPE"] = bracket_df["MAPE"].map(lambda x: f"{x:.2f}%")
        st.markdown("**가격대별 예측 정확도 요약**")
        st.dataframe(bracket_df, use_container_width=True, hide_index=True)

    st.subheader("2. 회귀 계수 분석 (등급별 가격 프리미엄)")
    c_coef1, c_coef2 = st.columns(2)
    with c_coef1:
        coef_sorted = coef_df.sort_values("Standardized_Coef", ascending=True)
        fig_std_coef = px.bar(
            coef_sorted,
            x="Standardized_Coef",
            y="Feature",
            orientation="h",
            title="표준화 회귀 계수 (변수별 상대적 중요도)",
            color_discrete_sequence=["#2b5c8f"],
        )
        st.plotly_chart(fig_std_coef, use_container_width=True)

    with c_coef2:
        dummy_only = coef_df[coef_df["Feature"] != "log_carat"].copy()
        dummy_only["그룹"] = dummy_only["Feature"].apply(lambda x: x.split("_")[0].upper())        fig_prem = px.bar(
            dummy_only,
            x="Premium_Pct",
            y="Feature",
            color="그룹",
            orientation="h",
            title="최하위 등급(Fair / J / I1) 대비 가격 상승률 (%)",
            labels={"Premium_Pct": "가격 상승률 (%)"},
            color_discrete_map={"CLARITY": "#2b5c8f", "COLOR": "#1f8a70", "CUT": "#e6ab02"},
        )
        st.plotly_chart(fig_prem, use_container_width=True)
