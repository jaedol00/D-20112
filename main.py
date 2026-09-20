import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="영화 데이터 그래프 - 분포와 관계", layout="wide")

st.title("영화 데이터 그래프 - 분포와 관계")


@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"
    df = pd.read_csv(url)

    # 장르 전처리: 첫 번째 장르만 추출 (.str 접근자 사용으로 안전하게 처리)
    df["genre"] = df["genre"].astype(str).str.split("|").str[0].str.strip()
    return df


df = load_data()

# ---------------------------------------------------------
# 그래프 1: 장르별 영화 편수 (도넛 차트)
# ---------------------------------------------------------
st.header("1. 장르별 영화 편수 분포")

genre_counts = df["genre"].value_counts().reset_index()
genre_counts.columns = ["genre", "count"]

fig1 = px.pie(
    genre_counts,
    values="count",
    names="genre",
    hole=0.4,
    title="장르별 영화 비율 및 편수",
)

fig1.update_traces(
    textposition="inside",
    textinfo="percent+label",
    hovertemplate="<b>장르: %{label}</b><br>편수: %{value}편<br>비율: %{percent}",
)

st.plotly_chart(fig1, use_container_width=True)

st.subheader("💡 이 그래프로 알 수 있는 것")
st.write(
    "박스오피스 상위권 영화 중 특정 장르가 차지하는 비중과 주요 개봉 장르의 편수 분포를 한눈에 확인할 수 있습니다."
)

st.divider()

# ---------------------------------------------------------
# 그래프 2: 장르 및 영화별 총 관객수 분포 (트리맵)
# ---------------------------------------------------------
st.header("2. 장르 및 영화별 총 관객수 분포")

# 동일 영화명 중복으로 인한 Plotly 트리맵 에러 방지 (그룹화 처리)
df_treemap = df.groupby(["genre", "movieNm"], as_index=False)[
    "total_audi"
].sum()

fig2 = px.treemap(
    df_treemap,
    path=[px.Constant("전체 영화"), "genre", "movieNm"],
    values="total_audi",
    color="genre",
    title="장르 및 영화별 총 관객수 트리맵",
)

fig2.update_traces(
    hovertemplate="<b>%{label}</b><br>총 관객수: %{value:,}명<extra></extra>"
)

st.plotly_chart(fig2, use_container_width=True)

st.subheader("💡 이 그래프로 알 수 있는 것")
st.write(
    "장르별 전체 흥행 규모와 함께 각 장르 안에서 어떤 영화가 총 관객수를 주요하게 견인했는지 비교할 수 있습니다."
)

st.divider()

# ---------------------------------------------------------
# 그래프 3: 총 관객수 히스토그램
# ---------------------------------------------------------
st.header("3. 총 관객수 분포")

fig3 = px.histogram(
    df,
    x="total_audi",
    nbins=30,
    title="총 관객수 구간별 영화 편수 분포",
    labels={"total_audi": "총 관객수", "count": "영화 편수"},
    color_discrete_sequence=["#636EFA"],
)

fig3.update_layout(yaxis_title="영화 편수")
fig3.update_traces(
    hovertemplate="관객수 구간: %{x}<br>영화 편수: %{y}편<extra></extra>"
)

st.plotly_chart(fig3, use_container_width=True)

# 데이터 계산 (최다 관객 영화 및 구간 밀집도 분석)
top_movie_row = df.loc[df["total_audi"].idxmax()]
top_movie_name = top_movie_row["movieNm"]
top_movie_audi = top_movie_row["total_audi"]

under_2m_count = (df["total_audi"] < 2000000).sum()
under_2m_ratio = (under_2m_count / len(df)) * 100

st.subheader("💡 이 그래프로 알 수 있는 것")
st.write(
    f"- 대부분의 영화(약 {under_2m_ratio:.1f}%, 전체 216편 중 {under_2m_count}편)가 관객수 **200만 명 미만** 구간에 대거 몰려 있으며 극소수의 영화만 대형 흥행을 기록한 모습을 보여줍니다.\n"
    f"- 이 기간 중 가장 관객이 많은 영화는 **'{top_movie_name}'**(총 {top_movie_audi:,}명)입니다."
)

st.divider()

# ---------------------------------------------------------
# 그래프 4: 개봉일 스크린 수와 총 관객수의 관계 (산점도)
# ---------------------------------------------------------
st.header("4. 개봉일 스크린 수와 총 관객수 관계")

fig4 = px.scatter(
    df,
    x="first_scrn",
    y="total_audi",
    color="genre",
    hover_name="movieNm",
    title="개봉일 스크린 수 대비 총 관객수 산점도",
    labels={
        "first_scrn": "개봉일 스크린 수",
        "total_audi": "총 관객수",
        "genre": "장르",
    },
)

fig4.update_traces(
    hovertemplate="<b>%{hovertext}</b><br>개봉일 스크린 수: %{x:,}개<br>총 관객수: %{y:,}명<extra></extra>"
)

st.plotly_chart(fig4, use_container_width=True)

st.subheader("💡 이 그래프로 알 수 있는 것")
st.write(
    "초기 개봉일 스크린 수와 최종 총 관객수 간의 상관관계를 확인할 수 있으며, 장르별 스크린 확보 규모와 흥행 성공 사례를 비교분석할 수 있습니다."
)

st.divider()

# ---------------------------------------------------------
# 그래프 5: 주요 장르별 총 관객수 박스플롯 (10편 이상 장르)
# ---------------------------------------------------------
st.header("5. 주요 장르별 총 관객수 분포 (박스플롯)")

genre_counts_series = df["genre"].value_counts()
top_genres = genre_counts_series[genre_counts_series >= 10].index
df_top_genres = df[df["genre"].isin(top_genres)]

fig5 = px.box(
    df_top_genres,
    x="genre",
    y="total_audi",
    color="genre",
    hover_name="movieNm",
    points="outliers",
    title="영화 10편 이상 주요 장르별 총 관객수 박스플롯",
    labels={"genre": "장르", "total_audi": "총 관객수"},
)

fig5.update_traces(
    hovertemplate="<b>영화명: %{hovertext}</b><br>총 관객수: %{y:,}명<extra></extra>"
)

st.plotly_chart(fig5, use_container_width=True)

st.subheader("💡 이 그래프로 알 수 있는 것")
st.write(
    "영화 편수가 10편 이상인 주요 장르별 관객수의 중앙값과 분포 범위를 비교할 수 있으며, 박스 바깥으로 튀어나온 점(이상치)에 마우스를 올려 대형 흥행 성공을 거둔 대작 영화를 식별할 수 있습니다."
)

st.divider()

# ---------------------------------------------------------
# 그래프 6: 개봉일 스크린 수, 총 관객수, 첫 주 관객수의 관계 (버블 차트)
# ---------------------------------------------------------
st.header("6. 개봉일 스크린 수, 총 관객수, 첫 주 관객수의 관계 (버블 차트)")

fig6 = px.scatter(
    df,
    x="first_scrn",
    y="total_audi",
    size="first_week_audi",
    color="genre",
    hover_name="movieNm",
    title="스크린 수, 총 관객수 및 첫 주 관객수(버블 크기) 버블 차트",
    labels={
        "first_scrn": "개봉일 스크린 수",
        "total_audi": "총 관객수",
        "first_week_audi": "개봉 첫 주 관객수",
        "genre": "장르",
    },
    size_max=50,
)

fig6.update_traces(
    hovertemplate="<b>%{hovertext}</b><br>개봉일 스크린 수: %{x:,}개<br>총 관객수: %{y:,}명<extra></extra>"
)

st.plotly_chart(fig6, use_container_width=True)

st.subheader("💡 이 그래프로 알 수 있는 것")
st.write(
    "개봉일 스크린 수와 최종 관객수의 관계뿐만 아니라, 버블의 크기(개봉 첫 주 관객수)를 통해 초반 흥행 동력이 최종 흥행에 미친 영향을 종합적으로 관찰할 수 있습니다."
)

st.divider()

# ---------------------------------------------------------
# 그래프 7: 제작 국가 및 장르별 영화 편수 (선버스트 차트)
# ---------------------------------------------------------
st.header("7. 제작 국가 및 장르별 영화 편수")

# 제작 국가와 장르별 영화 편수 집계
df_nation_genre = (
    df.groupby(["nation", "genre"]).size().reset_index(name="count")
)

fig7 = px.sunburst(
    df_nation_genre,
    path=["nation", "genre"],
    values="count",
    color="nation",
    title="제작 국가 및 장르별 영화 편수 선버스트 차트",
)

fig7.update_traces(
    hovertemplate="<b>%{label}</b><br>영화 편수: %{value}편<extra></extra>"
)

st.plotly_chart(fig7, use_container_width=True)

st.subheader("💡 이 그래프로 알 수 있는 것")
st.write(
    "제작 국가별 전체 영화 수와 각 국가 내에서 어떤 장르의 영화가 주로 제작/개봉되었는지 계층 구조로 한눈에 비교할 수 있습니다."
)

st.divider()
