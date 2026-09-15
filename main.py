import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
# 5. 구역 나누기 (6개의 탭 구조 활용)
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📈 일별 관객수 추이",
        "📊 누적 관객수 추이",
        "🏆 TOP 5 장기흥행 영화 비교",
        "📉 전체 관객수 이동평균",
        "📊 월별 총 관객수",
        "🗓️ 캘린더 히트맵",
    ]
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

    movie_counts = df.groupby("영화명")["기준일자"].count()
    movies_over_20days = movie_counts[movie_counts >= 20].index

    top5_movies = (
        df[df["영화명"].isin(movies_over_20days)]
        .groupby("영화명")["누적관객수"]
        .max()
        .nlargest(5)
        .index.tolist()
    )

    top5_df = df[df["영화명"].isin(top5_movies)]

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

# -----------------------------------------------------------------------------
# 네 번째 탭: 전체 박스오피스 일별 합계 및 7일 이동평균선
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("📉 전체 박스오피스 관객수 및 7일 이동평균 추이")

    daily_total = df.groupby("기준일자")["해당일관객수"].sum().reset_index()
    daily_total["7일이동평균"] = daily_total["해당일관객수"].rolling(window=7).mean()

    fig_ma = go.Figure()

    fig_ma.add_trace(
        go.Scatter(
            x=daily_total["기준일자"],
            y=daily_total["해당일관객수"],
            mode="lines",
            name="일별 관객수 합계 (원본)",
            line=dict(color="rgba(180, 180, 180, 0.5)", width=1.5),
        )
    )

    fig_ma.add_trace(
        go.Scatter(
            x=daily_total["기준일자"],
            y=daily_total["7일이동평균"],
            mode="lines",
            name="7일 이동평균",
            line=dict(color="#1f77b4", width=3),
        )
    )

    fig_ma.update_layout(
        title="전체 박스오피스 일별 총 관객수 및 7일 이동평균 추이",
        xaxis_title="기준일자",
        yaxis_title="총 관객수",
        hovermode="x unified",
    )

    st.plotly_chart(fig_ma, use_container_width=True)

    st.info(
        "💡 **이 그래프로 알 수 있는 것:** 요일별 변동성(주말 급증, 평일 감소)으로 인한 일별 관객수의 급격한 튀임을 완화하여, 극장가 전체의 성수기/비수기 등 장기적인 시장 관객 흐름을 명확하게 파악할 수 있습니다."
    )

# -----------------------------------------------------------------------------
# 다섯 번째 탭: 월별 전체 박스오피스 총 관객수 막대그래프
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("📊 월별 전체 박스오피스 총 관객수")

    daily_total = df.groupby("기준일자")["해당일관객수"].sum().reset_index()
    daily_total["연월"] = daily_total["기준일자"].dt.strftime("%Y-%m")
    monthly_total = daily_total.groupby("연월")["해당일관객수"].sum().reset_index()

    fig_bar = px.bar(
        monthly_total,
        x="연월",
        y="해당일관객수",
        title="월별 박스오피스 총 관객수 현황",
        text_auto=".2s",
    )

    fig_bar.update_layout(
        xaxis_title="월 (연-월)",
        yaxis_title="월별 총 관객수",
        hovermode="x unified",
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    st.info(
        "💡 **이 그래프로 알 수 있는 것:** 월별 총 관객수 규모 비교를 통해 연중 극장가의 극비수기와 극성수기 달을 한눈에 구분하고, 계절별 관객 동향 패턴을 파악할 수 있습니다."
    )

# -----------------------------------------------------------------------------
# 여섯 번째 탭: 캘린더 히트맵 (월/주차별 x 요일별 관객수)
# -----------------------------------------------------------------------------
with tab6:
    st.subheader("🗓️ 주차별 x 요일별 관객수 캘린더 히트맵")

    # 1. 일별 전체 관객수 합계 계산
    daily_total = df.groupby("기준일자")["해당일관객수"].sum().reset_index()

    # 2. 날짜 정보에서 요일, 주차, 날짜 문자열 추출
    # dt.dayofweek: 월요일=0, 화요일=1, ..., 일요일=6
    day_names = ["월", "화", "수", "목", "금", "토", "일"]
    daily_total["요일_num"] = daily_total["기준일자"].dt.dayofweek
    daily_total["요일"] = daily_total["요일_num"].map(lambda x: day_names[x])

    # YYYY-MM 형태 및 주차 구분문자 생성
    daily_total["연도"] = daily_total["기준일자"].dt.isocalendar().year
    daily_total["주차"] = daily_total["기준일자"].dt.isocalendar().week
    daily_total["연주차"] = (
        daily_total["연도"].astype(str)
        + "년 "
        + daily_total["주차"].astype(str).str.zfill(2)
        + "주차"
    )
    daily_total["날짜str"] = daily_total["기준일자"].dt.strftime("%Y-%m-%d")

    # 3. 히트맵 표현을 위한 피벗 테이블 생성 (행: 연주차, 열: 요일)
    pivot_val = daily_total.pivot(
        index="연주차", columns="요일_num", values="해당일관객수"
    )
    pivot_date = daily_total.pivot(
        index="연주차", columns="요일_num", values="날짜str"
    )

    # 모든 요일(0~6) 컬럼 보장
    for d in range(7):
        if d not in pivot_val.columns:
            pivot_val[d] = None
            pivot_date[d] = None

    pivot_val = pivot_val[[0, 1, 2, 3, 4, 5, 6]]
    pivot_date = pivot_date[[0, 1, 2, 3, 4, 5, 6]]

    # 4. 마우스 오버(Hover) 시 나타날 2차원 텍스트 마스크 생성 (yyyy-mm-dd 표시)
    hover_text = []
    for r in range(len(pivot_val)):
        row_hover = []
        for c in range(7):
            d_str = pivot_date.iloc[r, c]
            v_val = pivot_val.iloc[r, c]
            if pd.notna(d_str) and pd.notna(v_val):
                row_hover.append(
                    f"날짜: {d_str}<br>요일: {day_names[c]}요일<br>관객수: {int(v_val):,}명"
                )
            else:
                row_hover.append("데이터 없음")
        hover_text.append(row_hover)

    # 5. Plotly Heatmap 생성 (관객수가 많을수록 붉은색/진하게 표기)
    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=pivot_val.values,
            x=day_names,  # 월요일부터 일요일 순서
            y=pivot_val.index,
            text=hover_text,
            hoverinfo="text",
            colorscale="Reds",  # 관객수가 많을수록 색이 진해짐
        )
    )

    # 위에서 아래로 시간 순서대로 흐르도록 Y축 반전 설정
    fig_heatmap.update_layout(
        title="주차별 요일 관객수 분포 (마우스를 올리면 YYYY-MM-DD 확인 가능)",
        xaxis_title="요일",
        yaxis_title="주차",
        yaxis=dict(autorange="reversed"),
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.info(
        "💡 **이 그래프로 알 수 있는 것:** 요일별(평일 vs 주말) 관객 차이뿐만 아니라 특정 연휴나 공휴일이 속한 주차의 일별 관객 집중도를 한눈에 시각적으로 파악할 수 있습니다."
    )
