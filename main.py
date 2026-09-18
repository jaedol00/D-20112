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

fig2 = px.treemap(
    df,
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
