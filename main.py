import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

# 1. 페이지 기본 설정 (제목, 레이아웃)
st.set_page_config(page_title="어제 박스오피스", layout="wide")

# 2. 한국 시간(KST) 기준 어제 날짜 구하기
# 배포 서버 시계가 해외 기준이더라도 한국 시간(UTC+9)을 정확히 계산합니다.
kst = timezone(timedelta(hours=9))
yesterday = datetime.now(kst) - timedelta(days=1)
target_dt = yesterday.strftime("%Y%m%d")       # API 요청용 날짜 형식 (YYYYMMDD)
display_date = yesterday.strftime("%Y년 %m월 %d일") # 화면 표시용 날짜 형식

st.title(f"🎬 {display_date} 박스오피스")

# 3. KOBIS API 데이터 불러오기 함수 (1시간 동안 결과 기억/캐싱)
@st.cache_data(ttl=3600)
def fetch_box_office(date_str):
    # 비밀 금고(st.secrets)에 인증키가 설정되어 있는지 확인
    if "KOBIS_KEY" not in st.secrets:
        return None, "Streamlit Secrets 설정에서 'KOBIS_KEY'를 찾을 수 없습니다."
    
    api_key = st.secrets["KOBIS_KEY"]
    url = "https://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json"
    params = {"key": api_key, "targetDt": date_str}
    
    try:
        # API에 데이터 요청
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # 1) 인증키 오류 등 API 자체에서 에러 상자(faultInfo)를 보낸 경우
        if "faultInfo" in data:
            error_msg = data["faultInfo"].get("message", "인증키를 확인해주세요.")
            return None, f"KOBIS API 오류: {error_msg}"
            
        # 2) 정상 응답 내 박스오피스 목록 확인
        box_office_list = data.get("boxOfficeResult", {}).get("dailyBoxOfficeList", [])
        
        # 3) 영화 목록이 비어 있는 경우
        if not box_office_list:
            return None, "영화 목록 데이터가 비어 있습니다. 해당 날짜의 집계가 완료되었는지 확인하세요."
            
        return box_office_list, None

    except requests.exceptions.RequestException as e:
        # 네트워크 오류 발생 처리
        return None, f"인터넷 연결 또는 KOBIS 서버 요청 실패: {e}"
    except Exception as e:
        # 기타 예상치 못한 오류 처리
        return None, f"데이터 처리 중 오류 발생: {e}"

# API 데이터 불러오기 실행
box_office_data, error_message = fetch_box_office(target_dt)

# 4. 에러 발생 시 사용자 안내 메시지 출력
if error_message:
    st.error("⚠️ 데이터를 불러오지 못했습니다.")
    st.warning(error_message)
    st.markdown("""
    **💡 다음 항목들을 확인해 주세요:**
    1. Streamlit Cloud의 앱 설정(**Secrets**)에 `KOBIS_KEY = "발급받은 키"`가 바르게 등록되어 있는지 확인하세요.
    2. KOBIS API 키가 유효한지 또는 일일 사용량을 초과하지 않았는지 확인해 주세요.
    3. 네트워크 상태나 KOBIS 서버에 장애가 없는지 확인해 주세요.
    """)
else:
    # 5. 데이터프레임 변환 및 숫자 형변환
    df = pd.DataFrame(box_office_data)
    
    # 문자로 전달되는 숫자를 계산/정렬에 사용하기 위해 숫자형(int)으로 변환
    numeric_columns = ['rank', 'audiCnt', 'audiAcc', 'scrnCnt']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 순위 기준 정렬
    df = df.sort_values(by='rank').reset_index(drop=True)

    # 6. 1위 영화 지표 카드 3개 출력
    top1_movie = df.iloc[0]
    st.subheader("🥇 어제의 1위 영화")
    col1, col2, col3 = st.columns(3)
    col1.metric(label="영화명", value=top1_movie['movieNm'])
    col2.metric(label="어제 관객수", value=f"{top1_movie['audiCnt']:,} 명")
    col3.metric(label="누적 관객수", value=f"{top1_movie['audiAcc']:,} 명")

    st.markdown("---")

    # 7. 관객수 상위 5편 막대그래프
    st.subheader("📊 관객수 상위 5개 영화")
    top5_df = df.head(5)
    st.bar_chart(data=top5_df, x='movieNm', y='audiCnt', use_container_width=True)

    st.markdown("---")

    # 8. 전체 박스오피스 순위 표
    st.subheader("📋 전체 박스오피스 순위")
    
    # 필요한 열만 추출 및 이름 변경
    table_df = df[['rank', 'movieNm', 'openDt', 'audiCnt', 'audiAcc', 'scrnCnt']].copy()
    table_df.columns = ['순위', '영화명', '개봉일', '관객수', '누적관객', '스크린수']

    # 천 단위 쉼표 포맷을 적용하여 표 출력
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
