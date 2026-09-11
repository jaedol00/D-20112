import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

# 1. 페이지 기본 설정 (제목, 레이아웃)
st.set_page_config(page_title="일별 박스오피스 조회", layout="wide")

st.title("🎬 일별 박스오피스")

# 2. 한국 시간(KST) 기준 날짜 계산 및 달력 날짜 선택 기능
# 배포 서버의 시계와 무관하게 한국 시간(UTC+9) 기준으로 계산합니다.
kst = timezone(timedelta(hours=9))
today_kst = datetime.now(kst).date()
yesterday_kst = today_kst - timedelta(days=1)

# 오늘 데이터는 아직 집계 전이므로, 달력에서 선택 가능한 최댓값(max_value)을 '어제'로 제한합니다.
selected_date = st.date_input(
    "조회할 날짜를 선택하세요 (최대 어제까지 선택 가능)",
    value=yesterday_kst,
    max_value=yesterday_kst
)

# 선택한 날짜를 API 요구 형식(YYYYMMDD)과 화면 표시용 형식으로 변환
target_dt = selected_date.strftime("%Y%m%d")
display_date = selected_date.strftime("%Y년 %m월 %d일")

# 3. KOBIS API 데이터 불러오기 함수 (1시간 캐싱)
@st.cache_data(ttl=3600)
def fetch_box_office(date_str):
    # Secrets 비밀 금고에서 인증키 가져오기
    if "KOBIS_KEY" not in st.secrets:
        return None, "NO_KEY"
    
    api_key = st.secrets["KOBIS_KEY"]
    url = "https://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json"
    params = {"key": api_key, "targetDt": date_str}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # 1) 인증키 오류 등 faultInfo 응답 처리
        if "faultInfo" in data:
            error_msg = data["faultInfo"].get("message", "인증키를 확인해주세요.")
            return None, f"KOBIS API 오류: {error_msg}"
            
        box_office_list = data.get("boxOfficeResult", {}).get("dailyBoxOfficeList", [])
        
        # 2) 영화 목록이 비어 있는 경우
        if not box_office_list:
            return None, "EMPTY_LIST"
            
        return box_office_list, None

    except requests.exceptions.RequestException as e:
        return None, f"인터넷 연결 또는 KOBIS 서버 요청 실패: {e}"
    except Exception as e:
        return None, f"데이터 처리 중 오류 발생: {e}"

# API 데이터 불러오기 실행
box_office_data, error_message = fetch_box_office(target_dt)

# 4. 에러 및 예외 상황에 따른 사용자 안내 문구 출력
if error_message == "EMPTY_LIST":
    # 날짜를 선택했으나 아직 데이터가 집계되지 않았거나 없는 경우
    st.info("그날은 아직 집계 전입니다.")

elif error_message == "NO_KEY":
    st.error("⚠️ Streamlit Secrets 설정에서 'KOBIS_KEY'를 찾을 수 없습니다.")
    st.markdown("Streamlit Cloud 설정(Secrets)에 `KOBIS_KEY = '발급받은키'` 형태로 추가해 주세요.")

elif error_message:
    st.error("⚠️ 데이터를 불러오지 못했습니다.")
    st.warning(error_message)
    st.markdown("""
    **💡 다음 항목들을 확인해 주세요:**
    1. KOBIS API 키가 유효한지 또는 일일 사용량을 초과하지 않았는지 확인해 주세요.
    2. 네트워크 상태나 KOBIS 서버에 장애가 없는지 확인해 주세요.
    """)

else:
    # 5. 데이터 프레임 변환 및 연산용 숫자 형변환
    df = pd.DataFrame(box_office_data)
    
    # 문자로 들어오는 숫자 데이터를 정수형(int)으로 전환
    numeric_columns = ['rank', 'rankInten', 'audiCnt', 'audiAcc', 'scrnCnt']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # 순위 기준 정렬
    df = df.sort_values(by='rank').reset_index(drop=True)

    # 6. 요구사항 적용: 100만 관객 이상 트로피 🏆 붙이기
    def add_trophy_if_million(row):
        movie_name = row['movieNm']
        # 누적관객수(audiAcc)가 1,000,000명 이상이면 트로피 이모지 추가
        if row['audiAcc'] >= 1_000_000:
            return f"🏆 {movie_name}"
        return movie_name

    df['display_movieNm'] = df.apply(add_trophy_if_million, axis=1)

    # 7. 요구사항 적용: 전날 대비 순위 증감(rankInten) 화살표 표시
    # 양수 -> 빨간 위 화살표(🔺), 음수 -> 파란 아래 화살표(🔻)
    def format_rank_change(inten):
        if inten > 0:
            return f"🔺 {inten}"
        elif inten < 0:
            return f"🔻 {abs(inten)}"
        else:
            return "-"

    df['rank_change'] = df['rankInten'].apply(format_rank_change)

    # 8. 1위 영화 지표 카드 출력
    top1_movie = df.iloc[0]
    st.subheader(f"🥇 {display_date} 1위 영화")
    col1, col2, col3 = st.columns(3)
    col1.metric(label="영화명", value=top1_movie['display_movieNm'])
    col2.metric(label="당일 관객수", value=f"{top1_movie['audiCnt']:,} 명")
    col3.metric(label="누적 관객수", value=f"{top1_movie['audiAcc']:,} 명")

    st.markdown("---")

    # 9. 관객수 상위 5편 막대그래프
    st.subheader("📊 관객수 상위 5개 영화")
    top5_df = df.head(5)
    # 그래프 축에는 이모지 없는 순수 영화명을 사용해 가독성을 높입니다.
    st.bar_chart(data=top5_df, x='movieNm', y='audiCnt', use_container_width=True)

    st.markdown("---")

    # 10. 전체 박스오피스 순위 표 출력
    st.subheader(f"📋 {display_date} 박스오피스 순위")
    
    # 필요한 열 선택 및 사용자용 컬럼 이름 변경
    table_df = df[['rank', 'rank_change', 'display_movieNm', 'openDt', 'audiCnt', 'audiAcc', 'scrnCnt']].copy()
    table_df.columns = ['순위', '순위 변동', '영화명', '개봉일', '관객수', '누적관객', '스크린수']

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "순위": st.column_config.NumberColumn(format="%d위"),
            "관객수": st.column_config.NumberColumn(format="%,d명"),
            "누적관객": st.column_config.NumberColumn(format="%,d명"),
            "스크린수": st.column_config.NumberColumn(format="%,d개"),
        }
    )
