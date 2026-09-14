import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------------------------------------------------------
# 1. 데이터 불러오기 및 캐싱 처리
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/keep-growing-park/data-science/refs/heads/main/dataset/kobis_1year_boxoffice.csv"
    df = pd.read_csv(url)

    # -------------------------------------------------------------------------
    # 2. 날짜 및 데이터 전처리
    # -------------------------------------------------------------------------
    # 결측치(빈 값)가 하나라도 포함된 행 삭제
    df = df.dropna()

    # "기준일자" 컬럼을 날짜(datetime) 형식으로 변환
    df["기준일자"] = pd.to_datetime(df["기준일자"])

    # 데이터를 기준일자 순서대로 오름차순 정렬
    df = df.sort_values(by="기준일자")

    return df


# 전처리된 데이터 로드
df = load_data()

# -----------------------------------------------------------------------------
# 기본 웹앱 레이아웃 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="영화 박스오피스 분석 대시보드", layout="wide")
st.title("🎬 영화 박스오피스 관객수 분석")

# -----------------------------------------------------------------------------
# 3. 영화 선택 기능 (개별 영화 분석용)
# -----------------------------------------------------------------------------
# 영화별 최대 누적관객수를 기준으로 내림차순 정렬하여 중복 없는 영화 목록 추출
movie_order = (
    df.groupby("영화명")["누적관객수"]
    .max()
    .sort_values(ascending=False)
    .index.tolist()
)

st.sidebar.header("⚙️ 분석 옵션")
selected_movie = st.sidebar.selectbox("개별 분석할 영화를 선택하세요", movie_order)

filtered_df = df[df["영화명"] == selected_movie]

# -----------------------------------------------------------------------------
# 5. 구역 나누기 (3개의 탭 구조 활용)
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["📈 일별 관객수 추이", "📊 누적 관객수 추이", "🏆 TOP 5 장기흥행 영화 비교"]
)

# -----------------------------------------------------------------------------
# 첫 번째 탭: 선 그래프 (일별 관객수)
# -----------------------------------------------------------------------------
with tab1:
    st.subheader(f"[{selected_movie}] 일별 관객수 변화 그래프")

    fig_line = px.line(
        filtered_df,
        x="기준일자",
        y="해당일관객수",
        title=f"{selected_movie} - 기준일자별 해당일 관객수 추이",
        markers=True,
    )

    fig_line.update_layout(
        xaxis_title="기준일자", yaxis_title="해당일 관객수", hovermode="x unified"
    )

    st.plotly_chart(fig_line, use_container_width=True)

    st.info(
        "💡 **이 그래프로 알 수 있는 것:** 영화 개봉 후 날짜별 관객수의 증감 추세와 최고 관객수를 기록한 시점을 파악할 수 있습니다."
    )

# -----------------------------------------------------------------------------
# 두 번째 탭: 영역 차트 (누적 관객수)
# -----------------------------------------------------------------------------
with tab2:
    st.subheader(f"[{selected_movie}] 누적 관객수 변화 그래프")

    fig_area = px.area(
        filtered_df,
        x="기준일자",
        y="누적관객수",
        title=f"{selected_movie} - 기준일자별 누적 관객수 추이",
    )

    fig_area.update_layout(
        xaxis_title="기준일자", yaxis_title="누적 관객수", hovermode="x unified"
    )

    st.plotly_chart(fig_area, use_container_width=True)

    st.info(
        "💡 **이 그래프로 알 수 있는 것:** 시간이 지남에 따라 관객수가 누적되는 증가 양상을 한눈에 파악할 수 있으며, 관객수 증가 폭이 둔화되는 흥행 완만 시점을 확인할 수 있습니다."
    )

# -----------------------------------------------------------------------------
# 세 번째 탭: 다중 선 그래프 (20일 이상 등장한 영화 중 TOP 5 비교)
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("🏆 20일 이상 등장한 상위 5개 영화 흥행 추이 비교")

    # 1. 영화별 등장 일수(행 개수)를 구해 20일 이상 등장한 영화 조건 필터링
    movie_counts = df.groupby("영화명")["기준일자"].count()
    movies_over_20days = movie_counts[movie_counts >= 20].index

    # 2. 조건에 맞는 영화들 중 최대 누적관객수가 가장 높은 5개 영화 선별
    top5_movies = (
        df[df["영화명"].isin(movies_over_20days)]
        .groupby("영화명")["누적관객수"]
        .max()
        .nlargest(5)
        .index.tolist()
    )

    # 3. 해당 5개 영화 데이터 추출
    top5_df = df[df["영화명"].isin(top5_movies)]

    # 4. 다중 선 그래프 생성
    fig_multi = px.line(
        top5_df,
        x="기준일자",
        y="누적관객수",
        color="영화명",
        title="20일 이상 등장 영화 중 TOP 5 기준일자별 누적 관객수 추이",
        markers=True,
    )

    fig_multi.update_layout(
        xaxis_title="기준일자",
        yaxis_title="누적 관객수",
        legend_title="영화 제목",
        hovermode="x unified",
    )

    st.plotly_chart(fig_multi, use_container_width=True)

    st.info(
        "💡 **이 그래프로 알 수 있는 것:** 단기 반짝 흥행에 그치지 않고 최소 20일 이상 TOP10 순위에 머무른 장기 흥행작 5개의 관객 모객 속도와 누적 관객 규모를 한눈에 비교할 수 있습니다."
    )
