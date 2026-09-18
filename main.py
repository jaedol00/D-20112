import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="영화 데이터 그래프 - 분포와 관계", layout="wide")

st.title("영화 데이터 그래프 - 분포와 관계")


@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"
    df = pd.read_csv(url)

    # 장르 전처리: 첫 번째 장르만 추출
    df["genre"] = (
        df["genre"].astype(str).apply(lambda x: x.split("|")[0].strip())
    )
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
# 그래프 2: 개봉일 스크린 수와 총 관객수의 관계 (산점도)
# ---------------------------------------------------------
st.header("2. 개봉일 스크린 수와 총 관객수 관계")

fig2 = px.scatter(
    df,
    x="first_scrn",
    y="total_audi",
    color="genre",
    hover_data=["movieNm", "days_in_top10"],
    labels={
        "first_scrn": "개봉일 스크린 수",
        "total_audi": "총 관객 수",
        "genre": "장르",
    },
    title="스크린 수 대비 총 관객수 분포",
)

st.plotly_chart(fig2, use_container_width=True)

st.subheader("💡 이 그래프로 알 수 있는 것")
st.write(
    "초기 스크린 확보가 최종 총 관객수에 미치는 양의 상관관계와 장르별 스크린 확보 규모의 차이를 분석할 수 있습니다."
)

st.divider()
