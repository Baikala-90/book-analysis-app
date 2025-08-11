import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import plotly.express as px
import io
from pytrends.request import TrendReq

# ----------------------------------------------------------------------
# 신규 기능: 시장 트렌드 분석 함수
# ----------------------------------------------------------------------


@st.cache_data(ttl=3600)  # 1시간 동안 캐시 유지
def get_daily_trends():
    """Google Trends에서 대한민국의 일별 인기 검색어를 가져옵니다."""
    try:
        pytrends = TrendReq(hl='ko-KR', tz=540)
        trending_searches_df = pytrends.trending_searches(pn='south_korea')
        return trending_searches_df[0].tolist()
    except Exception as e:
        # st.error는 메인 스레드에서만 호출 가능하므로, 여기서는 None을 반환하고 호출부에서 처리
        return None

# ----------------------------------------------------------------------
# 기존 데이터 처리 및 분석 함수들
# ----------------------------------------------------------------------


@st.cache_data
def load_and_process_data(uploaded_file_contents, k, lambda_param, w_amount, w_freq, w_recency):
    """파일 내용을 입력받아 데이터 처리 및 분석을 수행합니다."""
    try:
        file_name = st.session_state.get('file_name', '')
        file_extension = file_name.split('.')[-1]

        if file_extension == 'csv':
            df_raw = pd.read_csv(io.BytesIO(
                uploaded_file_contents), low_memory=False)
        else:
            df_raw = pd.read_excel(io.BytesIO(uploaded_file_contents))

        if df_raw.empty:
            return None, "오류: 업로드된 파일에 분석할 데이터가 없습니다."
    except Exception as e:
        return None, f"파일을 읽는 중 오류가 발생했습니다: {e}"

    date_col = next((col for col in df_raw.columns if '날짜' in col), '날짜')
    book_col = next((col for col in df_raw.columns if '도서명' in col), '도서명')
    amount_col = next((col for col in df_raw.columns if '발주량' in col), '발주량')

    if not all(c in df_raw.columns for c in [date_col, book_col, amount_col]):
        return None, f"분석에 필요한 컬럼('{date_col}', '{book_col}', '{amount_col}') 중 일부를 찾을 수 없습니다."

    df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors='coerce')
    df_raw.dropna(subset=[date_col, book_col, amount_col], inplace=True)

    weekday_map = {"Monday": "월", "Tuesday": "화", "Wednesday": "수",
                   "Thursday": "목", "Friday": "금", "Saturday": "토", "Sunday": "일"}
    df_raw['날짜_라벨'] = df_raw[date_col].dt.strftime(
        '%m월 %d일') + " (" + df_raw[date_col].dt.day_name().map(weekday_map).fillna('') + ")"

    agg_df = df_raw.groupby(book_col).agg(
        총발주량=(amount_col, 'sum'),
        발주횟수=(book_col, 'count'),
        최근발주일=(date_col, 'max'),
        최초발주일=(date_col, 'min')
    ).reset_index()

    duration = (agg_df['최근발주일'] - agg_df['최초발주일']).dt.days
    agg_df['평균 발주 간격'] = np.where(
        agg_df['발주횟수'] > 1, duration / (agg_df['발주횟수'] - 1), np.nan)

    def calculate_recency_score(days_diff, max_days, k_val, lambda_val):
        if max_days == 0:
            return 1.0
        scaled_diff = days_diff / max_days
        return 1 / (1 + scaled_diff * k_val) if scaled_diff <= 1 else np.exp(-scaled_diff * lambda_val)

    기준일 = agg_df['최근발주일'].max()
    agg_df['최초발주후경과일'] = (기준일 - agg_df['최초발주일']).dt.days
    agg_df['경과일'] = (기준일 - agg_df['최근발주일']).dt.days
    max_days = agg_df['경과일'].max()
    agg_df['시간가중치'] = agg_df['경과일'].apply(
        lambda x: calculate_recency_score(x, max_days, k, lambda_param))

    features = ['총발주량', '발주횟수', '시간가중치']
    scaler = MinMaxScaler()

    norm_features = [f + '_정규화' for f in features]
    agg_df[norm_features] = scaler.fit_transform(agg_df[features])

    kmeans = KMeans(n_clusters=5, init='k-means++',
                    n_init=10, max_iter=300, random_state=42)
    agg_df['Cluster'] = kmeans.fit_predict(agg_df[norm_features])

    centroids_df_normalized = pd.DataFrame(
        kmeans.cluster_centers_, columns=norm_features)
    rank_score_series = (w_amount * centroids_df_normalized['총발주량_정규화'] +
                         w_freq * centroids_df_normalized['발주횟수_정규화'] +
                         w_recency * centroids_df_normalized['시간가중치_정규화'])
    centroids_df_normalized['rank_score'] = rank_score_series

    sorted_clusters = centroids_df_normalized['rank_score'].sort_values(
        ascending=False).index

    grade_map = {cluster_id: f"{i+1}등급" for i,
                 cluster_id in enumerate(sorted_clusters)}
    agg_df['등급'] = agg_df['Cluster'].map(grade_map)

    total_weight = w_amount + w_freq + w_recency
    agg_df['종합 점수'] = (
        (w_amount * agg_df['총발주량_정규화'] +
         w_freq * agg_df['발주횟수_정규화'] +
         w_recency * agg_df['시간가중치_정규화']) / total_weight
    ) * 100

    agg_df.drop(columns=norm_features, inplace=True)

    return agg_df, df_raw


def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='analysis_result')
    return output.getvalue()


# ----------------------------------------------------------------------
# Streamlit 웹 애플리케이션 UI
# ----------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="도서 발주 데이터 분석 대시보드")
st.title("📚 도서 발주 데이터 분석 대시보드")

if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

with st.sidebar:
    st.header("⚙️ 1. 분석 파일 업로드")
    uploaded_file = st.file_uploader(
        "CSV 또는 XLSX 발주서 파일을 업로드하세요.", type=["csv", "xlsx"])

    if uploaded_file:
        if 'file_name' not in st.session_state or st.session_state.file_name != uploaded_file.name:
            st.session_state.analysis_done = False
            st.session_state.file_name = uploaded_file.name
            st.session_state.file_contents = uploaded_file.getvalue()

        try:
            date_col_name = '날짜'
            file_contents_for_preview = io.BytesIO(
                st.session_state.file_contents)
            if st.session_state.file_name.endswith('.csv'):
                df_preview = pd.read_csv(
                    file_contents_for_preview, low_memory=False, usecols=[date_col_name])
            else:
                df_preview = pd.read_excel(
                    file_contents_for_preview, usecols=[date_col_name])
            df_preview[date_col_name] = pd.to_datetime(
                df_preview[date_col_name], errors='coerce').dropna()
            time_span_days = (df_preview[date_col_name].max(
            ) - df_preview[date_col_name].min()).days if not df_preview.empty else 0
        except Exception:
            time_span_days = 365

        if time_span_days <= 7:
            k_default, lambda_default, period_text = 7.0, 0.5, "1주일 이내"
        elif time_span_days <= 31:
            k_default, lambda_default, period_text = 5.0, 0.7, "1개월 이내"
        elif time_span_days <= 182:
            k_default, lambda_default, period_text = 2.0, 1.0, "6개월 이내"
        elif time_span_days <= 365:
            k_default, lambda_default, period_text = 1.0, 1.5, "1년 이내"
        else:
            k_default, lambda_default, period_text = 0.5, 2.0, "1년 이상"

        st.header("⚙️ 2. 분석 민감도 설정")
        st.info(f"데이터 기간: **{period_text}**")
        k_param = st.slider("최신성 민감도 (k)", 0.1, 10.0, k_default, 0.1,
                            help="값이 클수록 '최근'에 발주된 도서에 더 높은 가중치를 부여합니다. 단기 데이터를 분석할 때 높게 설정하는 것이 좋습니다.")
        lambda_param = st.slider("장기 비활성 패널티 (λ)", 0.1, 10.0, lambda_default, 0.1,
                                 help="값이 클수록 발주된 지 '아주 오래된' 도서에 대한 패널티를 강하게 부여합니다. 장기 데이터를 분석할 때 유용합니다.")

        st.header("⚙️ 3. 등급/점수 중요도 설정")
        w_amount = st.slider("총발주량 중요도", 1, 5, 4,
                             help="이 값이 높을수록 '많이' 팔린 책이 높은 등급/점수를 받습니다.")
        w_freq = st.slider("발주횟수 중요도", 1, 5, 4,
                           help="이 값이 높을수록 '자주' 팔린 책이 높은 등급/점수를 받습니다.")
        w_recency = st.slider("시간가중치 중요도", 1, 5, 2,
                              help="이 값이 높을수록 '최근에' 팔린 책이 높은 등급/점수를 받습니다.")

        if st.button("🚀 분석 실행", type="primary", use_container_width=True):
            with st.spinner('데이터를 분석 중입니다...'):
                agg_df, df_raw_or_error = load_and_process_data(
                    st.session_state.file_contents, k_param, lambda_param, w_amount, w_freq, w_recency)
                if agg_df is not None:
                    st.session_state.analysis_done = True
                    st.session_state.agg_df = agg_df
                    st.session_state.df_raw = df_raw_or_error
                else:
                    st.session_state.analysis_done = False
                    st.error(df_raw_or_error)
            st.rerun()

if not uploaded_file:
    st.info("👈 사이드바에서 분석할 파일을 업로드해주세요.")
elif st.session_state.analysis_done:
    agg_df = st.session_state.agg_df
    df_raw = st.session_state.df_raw

    st.success(f"✅ **{st.session_state.file_name}** 파일 분석이 완료되었습니다.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["[ 📈 등급 요약 ]", "[ 🌟 유망 도서 발굴 ]", "[ 📊 데이터 인사이트 ]", "[ 🔍 시장 트렌드 분석 ]", "[ 📋 전체 데이터 ]"])

    with tab1:
        st.header("등급별 요약")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("① 등급별 도서 분포")
            grade_order = [f"{i}등급" for i in range(1, 6)]
            grade_counts = agg_df['등급'].value_counts().reindex(
                grade_order).reset_index()
            fig_bar = px.bar(grade_counts, x='등급', y='count', color='등급', text_auto=True, category_orders={"등급": grade_order},
                             color_discrete_map={'1등급': '#0081CF', '2등급': '#00A1E0', '3등급': '#7ECEF4', '4등급': '#B1DFF7', '5등급': '#CCCCCC'}, title="포트폴리오 등급별 도서 수")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.subheader("② 등급별 의미와 전략")
            st.markdown("""
            | 등급 | 의미 및 상태 | 추천 전략 |
            | :---: | :--- | :--- |
            | **1등급** | **최상위 핵심 그룹** | **재고 최우선 확보**, 적극적인 마케팅 |
            | **2등급** | 꾸준한 **우수 그룹** | **안정적인 재고 수준 유지**, 크로스셀링 |
            | **3등급** | **성장/하락 가능성** 보유 그룹 | 수요 예측, 판매 촉진 전략 고민 |
            | **4등급** | 발주가 뜸한 **주의 그룹** | **재고 최소화**, 원인 분석 |
            | **5등급** | **비활성/관리 그룹** | **재고 처분 및 단종** 검토 |
            """)

    with tab2:
        st.header("🌟 신규 유망 도서 발굴")
        book_col_name = next(
            (col for col in df_raw.columns if '도서명' in col), '도서명')

        col1, col2 = st.columns(2)
        with col1:
            days_since_first_limit = st.slider("출시 기간 필터 (일)", 0, int(
                agg_df['최초발주후경과일'].max()), min(180, int(agg_df['최초발주후경과일'].max())))
        with col2:
            min_freq_limit = st.slider(
                "최소 발주 횟수 필터", 1, int(agg_df['발주횟수'].max()), 3)

        col3, col4 = st.columns(2)
        with col3:
            max_interval = int(agg_df['평균 발주 간격'].dropna().max(
            )) if not agg_df['평균 발주 간격'].isna().all() else 90
            interval_limit = st.slider(
                "최대 평균 발주 간격 필터", 1, max_interval, min(30, max_interval))
        with col4:
            min_amount_limit = st.slider(
                "최소 총 발주량 필터", 0, int(agg_df['총발주량'].max()), 10)

        promising_books_df = agg_df[
            (agg_df['최초발주후경과일'] <= days_since_first_limit) &
            (agg_df['발주횟수'] >= min_freq_limit) &
            (agg_df['평균 발주 간격'].fillna(interval_limit + 1) <= interval_limit) &
            (agg_df['총발주량'] >= min_amount_limit)
        ]

        st.subheader(f"필터링 결과: 총 {len(promising_books_df)}권의 유망 도서를 찾았습니다.")

        col_sort1, col_sort2 = st.columns(2)
        with col_sort1:
            sort_by_options = {"종합 점수": "종합 점수", "평균 발주 간격": "평균 발주 간격",
                               "총발주량": "총발주량", "발주 횟수": "발주횟수", "출시일": "최초발주후경과일"}
            sort_by = st.selectbox(
                "정렬 기준 선택", options=list(sort_by_options.keys()))
        with col_sort2:
            sort_order = st.selectbox("정렬 순서 선택", options=["내림차순", "오름차순"])

        promising_books_df = promising_books_df.sort_values(
            by=sort_by_options[sort_by], ascending=(sort_order == "오름차순"))

        for _, row in promising_books_df.iterrows():
            book_title = row[book_col_name]
            with st.expander(f"'{book_title}' (종합점수: {row['종합 점수']:.2f}점 / 평균 {row['평균 발주 간격']:.1f}일 간격)"):
                history_df = df_raw[df_raw[book_col_name] == book_title].copy()
                daily_history = history_df.groupby(next(col for col in history_df.columns if '날짜' in col)).agg(
                    일일_발주량=(
                        next(col for col in history_df.columns if '발주량' in col), 'sum'),
                    날짜_라벨=('날짜_라벨', 'first')
                ).reset_index()

                fig = px.line(daily_history, x='날짜_라벨', y='일일_발주량', title=f"'{book_title}' 발주량 추이", markers=True, labels={
                              '날짜_라벨': '날짜', '일일_발주량': '발주량'})
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("데이터 인사이트 시각화")
        date_col = next((col for col in df_raw.columns if '날짜' in col), '날짜')
        amount_col = next(
            (col for col in df_raw.columns if '발주량' in col), '발주량')

        viz_tab1, viz_tab2, viz_tab3 = st.tabs(
            ["월별 발주 현황", "주별 발주 현황", "일별 발주 현황"])

        with viz_tab1:
            st.subheader("월별 총 발주량")
            monthly_orders = df_raw.groupby(pd.Grouper(key=date_col, freq='ME')).agg(
                합계=(amount_col, 'sum')).reset_index()
            monthly_orders['날짜_라벨'] = monthly_orders[date_col].dt.strftime(
                '%Y년 %m월')
            fig_month = px.line(monthly_orders, x='날짜_라벨', y='합계', title="월별 총 발주량",
                                markers=True, labels={'날짜_라벨': '날짜', '합계': '발주량'})
            st.plotly_chart(fig_month, use_container_width=True)

        with viz_tab2:
            st.subheader("주별 총 발주량")
            weekly_orders = df_raw.groupby(pd.Grouper(
                key=date_col, freq='W-MON')).agg(합계=(amount_col, 'sum')).reset_index()
            weekly_orders['날짜_라벨'] = weekly_orders[date_col].dt.strftime(
                '%m월 %d일')
            fig_week = px.line(weekly_orders, x='날짜_라벨', y='합계', title="주별 총 발주량",
                               markers=True, labels={'날짜_라벨': '날짜', '합계': '발주량'})
            st.plotly_chart(fig_week, use_container_width=True)

        with viz_tab3:
            st.subheader("일별 총 발주량")
            daily_orders = df_raw.groupby(date_col).agg(
                합계=(amount_col, 'sum')).reset_index()
            daily_orders['날짜_라벨'] = daily_orders[date_col].dt.strftime(
                '%m월 %d일 (%a)')
            fig_day = px.line(daily_orders, x='날짜_라벨', y='합계', title="일별 총 발주량", labels={
                              '날짜_라벨': '날짜', '합계': '발주량'})
            st.plotly_chart(fig_day, use_container_width=True)

    with tab4:
        st.header("🔍 시장 트렌드 분석 (Google Trends)")
        st.info("현재 대한민국에서 인기 있는 검색어와 관련된 도서를 찾아 수요를 예측해 보세요.")

        trending_keywords = get_daily_trends()

        if trending_keywords is None:
            st.error("Google 트렌드 데이터를 가져오는 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
        elif not trending_keywords:
            st.warning("현재 인기 검색어 정보를 가져올 수 없습니다.")
        else:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("오늘의 인기 검색어")
                st.dataframe(trending_keywords, hide_index=True,
                             use_container_width=True, column_config={"0": "키워드"})

            with col2:
                st.subheader("관련 도서 목록")
                selected_keyword = st.selectbox(
                    "분석할 인기 검색어를 선택하세요:", trending_keywords)

                if selected_keyword:
                    book_col_name = next(
                        (col for col in agg_df.columns if '도서명' in col), '도서명')
                    matched_books = agg_df[agg_df[book_col_name].str.contains(
                        selected_keyword, case=False, na=False)]

                    if matched_books.empty:
                        st.write(
                            f"**'{selected_keyword}'** 키워드가 포함된 도서를 찾을 수 없습니다.")
                    else:
                        display_cols = ['도서명', '등급', '종합 점수', '총발주량', '최근발주일']
                        st.dataframe(
                            matched_books[display_cols].sort_values(
                                by="종합 점수", ascending=False).style.format({'종합 점수': "{:.2f}"}),
                            use_container_width=True
                        )

    with tab5:
        st.header("전체 분석 데이터")
        display_columns = ['도서명', '등급', '종합 점수', '총발주량',
                           '발주횟수', '시간가중치', '평균 발주 간격', '최초발주일', '최근발주일', '경과일']
        final_df = agg_df[display_columns].sort_values(
            by='종합 점수', ascending=False)
        st.dataframe(final_df.style.format(
            {'종합 점수': "{:.2f}", '시간가중치': "{:.3f}", '평균 발주 간격': "{:.1f}"}), use_container_width=True)

        st.subheader("결과 다운로드")
        col1, col2 = st.columns(2)
        with col1:
            csv_data = final_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("💾 CSV 파일로 다운로드", data=csv_data,
                               file_name='book_cluster_analysis.csv', mime='text/csv', use_container_width=True)
        with col2:
            excel_data = to_excel(final_df)
            st.download_button("💾 XLSX 파일로 다운로드", data=excel_data, file_name='book_cluster_analysis.xlsx',
                               mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', use_container_width=True)
else:
    st.info("사이드바에서 파일을 업로드하고 설정을 마친 후 '분석 실행' 버튼을 눌러주세요.")
