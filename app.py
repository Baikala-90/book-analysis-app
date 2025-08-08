import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import plotly.express as px
import io
import locale

# 한국어 요일 설정을 위해 로케일 설정
try:
    locale.setlocale(locale.LC_TIME, 'ko_KR.UTF-8')
except locale.Error:
    if 'locale_warning_shown' not in st.session_state:
        st.warning("한국어 로케일(ko_KR.UTF-8)을 설정할 수 없습니다. 요일이 영어로 표시될 수 있습니다.")
        st.session_state.locale_warning_shown = True

# ----------------------------------------------------------------------
# 데이터 처리 및 분석 함수
# ----------------------------------------------------------------------


@st.cache_data
def load_and_process_data(uploaded_file, k, lambda_param, w_amount, w_freq, w_recency):
    """
    파일 로드부터 클러스터링, 가중치 기반 등급 부여까지 모든 데이터 처리를 수행합니다.
    """
    try:
        uploaded_file.seek(0)
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file, low_memory=False)
        else:
            df_raw = pd.read_excel(uploaded_file)

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

    df_raw['날짜_포맷'] = df_raw[date_col].dt.strftime('%m월 %d일 (%a)')

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
    agg_df_scaled = agg_df.copy()
    agg_df_scaled[features] = scaler.fit_transform(agg_df_scaled[features])

    kmeans = KMeans(n_clusters=5, init='k-means++',
                    n_init=10, max_iter=300, random_state=42)
    agg_df['Cluster'] = kmeans.fit_predict(agg_df_scaled[features])

    centroids_df_normalized = pd.DataFrame(
        kmeans.cluster_centers_, columns=features)
    rank_score_series = (w_amount * centroids_df_normalized['총발주량'] +
                         w_freq * centroids_df_normalized['발주횟수'] +
                         w_recency * centroids_df_normalized['시간가중치'])
    centroids_df_normalized['rank_score'] = rank_score_series

    sorted_clusters = centroids_df_normalized['rank_score'].sort_values(
        ascending=False).index

    grade_map = {cluster_id: f"{i+1}등급" for i,
                 cluster_id in enumerate(sorted_clusters)}
    score_map = {cluster_id: score for cluster_id,
                 score in zip(sorted_clusters, [5, 4, 3, 2, 1])}
    agg_df['등급'] = agg_df['Cluster'].map(grade_map)
    agg_df['점수'] = agg_df['Cluster'].map(score_map)

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

with st.sidebar:
    st.header("⚙️ 1. 분석 파일 업로드")
    uploaded_file = st.file_uploader(
        "CSV 또는 XLSX 발주서 파일을 업로드하세요.", type=["csv", "xlsx"])

    if uploaded_file:
        time_span_days = get_date_range(uploaded_file)
        if time_span_days <= 7:
            k_default, lambda_default, period_text = 7.0, 0.5, "1주일 이내 (매우 민감)"
        elif time_span_days <= 31:
            k_default, lambda_default, period_text = 5.0, 0.7, "1개월 이내 (민감)"
        elif time_span_days <= 182:
            k_default, lambda_default, period_text = 2.0, 1.0, "6개월 이내 (보통)"
        elif time_span_days <= 365:
            k_default, lambda_default, period_text = 1.0, 1.5, "1년 이내 (표준)"
        else:
            k_default, lambda_default, period_text = 0.5, 2.0, "1년 이상 (둔감)"

        st.header("⚙️ 2. 분석 민감도 설정")
        st.info(f"데이터 기간: **{period_text}**")
        k_param = st.slider("최신성 민감도 (k)", 0.1, 10.0, k_default, 0.1)
        lambda_param = st.slider(
            "장기 비활성 패널티 (λ)", 0.1, 10.0, lambda_default, 0.1)

        st.header("⚙️ 3. 등급 결정 중요도 설정")
        w_amount = st.slider("총발주량 중요도", 1, 5, 4)
        w_freq = st.slider("발주횟수 중요도", 1, 5, 4)
        w_recency = st.slider("시간가중치 중요도", 1, 5, 2)

        run_button = st.button("🚀 분석 실행", type="primary",
                               use_container_width=True)

if 'run_button' not in st.session_state:
    st.session_state.run_button = False

if not uploaded_file:
    st.info("👈 사이드바에서 분석할 파일을 업로드해주세요.")
elif run_button:
    with st.spinner('데이터를 분석 중입니다. 잠시만 기다려주세요...'):
        agg_df, df_raw_or_error = load_and_process_data(
            uploaded_file, k_param, lambda_param, w_amount, w_freq, w_recency)
        if agg_df is not None:
            st.session_state.run_button = True
            st.session_state.agg_df = agg_df
            st.session_state.df_raw = df_raw_or_error
            st.rerun()
        else:
            st.error(df_raw_or_error)
            st.session_state.run_button = False
elif st.session_state.run_button:
    agg_df = st.session_state.agg_df
    df_raw = st.session_state.df_raw

    st.success(f"✅ **{uploaded_file.name}** 파일 분석이 완료되었습니다.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["[ 📈 등급 요약 ]", "[ 🌟 유망 도서 발굴 ]", "[ 📊 데이터 인사이트 ]", "[ 📋 전체 데이터 ]"])

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
            | **1등급** | 발주량, 횟수, 최신성 등 핵심 지표들이 가장 우수한 **최상위 핵심 그룹** | **재고 최우선 확보**, 프로모션/광고 등 적극적인 마케팅 |
            | **2등급** | 매출의 핵심을 담당하고 있는 꾸준한 **우수 그룹** | **안정적인 재고 수준 유지**, 크로스셀링 연계 |
            | **3등급** | 성과가 중간 정도인 그룹, **성장 또는 하락 가능성** 보유 | 판매 데이터 기반 수요 예측, 판매 촉진 전략 고민 |
            | **4등급** | 발주가 뜸하거나 감소 추세인 **주의 그룹** | **재고 최소화**, 발주 감소 원인 분석 (계절성, 경쟁 등) |
            | **5등급** | 사실상 발주가 없는 **비활성/관리 그룹** | **재고 처분 고려 (이벤트, 할인)**, 사실상 단종 검토 |
            """)
    with tab2:
        st.header("🌟 신규 유망 도서 발굴")
        st.info("아래 조건을 조절하여 '새롭고, 꾸준한' 유망 도서를 직접 찾아보세요.")

        col1, col2 = st.columns(2)
        with col1:
            max_days_since_first = int(agg_df['최초발주후경과일'].max())
            days_since_first_limit = st.slider(
                "출시 기간 필터 (일)", 0, max_days_since_first, min(180, max_days_since_first))
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
        ].sort_values(by='평균 발주 간격')

        st.subheader(f"필터링 결과: 총 {len(promising_books_df)}권의 유망 도서를 찾았습니다.")

        book_col_name = next(
            (col for col in df_raw.columns if '도서명' in col), '도서명')
        book_list = ["- 도서 선택 -"] + promising_books_df[book_col_name].tolist()
        selected_book = st.selectbox("추이 그래프를 볼 도서를 선택하세요:", book_list)

        if selected_book != "- 도서 선택 -":
            history_df = df_raw[df_raw[book_col_name] == selected_book].copy()
            daily_history = history_df.groupby('날짜_포맷').agg(
                발주량=(next(col for col in df_raw.columns if '발주량' in col), 'sum'),
                날짜=(next(col for col in df_raw.columns if '날짜' in col), 'min')
            ).reset_index().sort_values(by='날짜')

            fig = px.line(daily_history, x='날짜_포맷', y='발주량',
                          title=f"'{selected_book}' 일별 발주량 추이", markers=True)
            fig.update_layout(yaxis_title="발주량", xaxis_title="날짜")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("데이터 인사이트 시각화")
        date_col = next((col for col in df_raw.columns if '날짜' in col), '날짜')
        amount_col = next(
            (col for col in df_raw.columns if '발주량' in col), '발주량')

        # --- 시각화 탭 구조 수정 ---
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(
            ["전체 발주 현황", "월별 발주 현황", "주별 발주 현황", "일별 발주 현황"])

        with viz_tab1:
            st.subheader("주요 트렌드 요약")
            col1, col2 = st.columns(2)
            with col1:
                monthly_orders = df_raw.groupby(pd.Grouper(key=date_col, freq='ME')).agg(
                    합계=(amount_col, 'sum')).reset_index()
                monthly_orders['날짜_포맷'] = monthly_orders[date_col].dt.strftime(
                    '%Y년 %m월')
                fig_month = px.line(monthly_orders, x='날짜_포맷',
                                    y='합계', title="월별 총 발주량", markers=True)
                st.plotly_chart(fig_month, use_container_width=True)
            with col2:
                weekly_orders = df_raw.groupby(pd.Grouper(
                    key=date_col, freq='W-MON')).agg(합계=(amount_col, 'sum')).reset_index()
                weekly_orders['날짜_포맷'] = weekly_orders[date_col].dt.strftime(
                    '%m월 %d일')
                fig_week = px.line(weekly_orders, x='날짜_포맷',
                                   y='합계', title="주별 총 발주량", markers=True)
                st.plotly_chart(fig_week, use_container_width=True)

        with viz_tab2:
            st.subheader("월별 총 발주량 상세")
            monthly_orders = df_raw.groupby(pd.Grouper(key=date_col, freq='ME')).agg(
                합계=(amount_col, 'sum')).reset_index()
            monthly_orders['날짜_포맷'] = monthly_orders[date_col].dt.strftime(
                '%Y년 %m월')
            fig_month_detail = px.bar(
                monthly_orders, x='날짜_포맷', y='합계', title="월별 총 발주량 (상세)", text_auto=True)
            st.plotly_chart(fig_month_detail, use_container_width=True)

        with viz_tab3:
            st.subheader("주별 총 발주량 상세")
            weekly_orders = df_raw.groupby(pd.Grouper(
                key=date_col, freq='W-MON')).agg(합계=(amount_col, 'sum')).reset_index()
            weekly_orders['날짜_포맷'] = weekly_orders[date_col].dt.strftime(
                '%Y-%m-%d')
            fig_week_detail = px.bar(
                weekly_orders, x='날짜_포맷', y='합계', title="주별 총 발주량 (상세)", text_auto='.2s')
            st.plotly_chart(fig_week_detail, use_container_width=True)

        with viz_tab4:
            st.subheader("일별 총 발주량 상세")
            daily_orders = df_raw.groupby('날짜_포맷').agg(합계=(amount_col, 'sum'), 날짜=(
                date_col, 'min')).reset_index().sort_values(by='날짜')
            fig_day_detail = px.line(
                daily_orders, x='날짜_포맷', y='합계', title="일별 총 발주량")
            fig_day_detail.update_traces(mode="lines+markers")
            st.plotly_chart(fig_day_detail, use_container_width=True)

    with tab4:
        st.header("전체 분석 데이터")
        display_columns = ['도서명', '등급', '점수', '총발주량', '발주횟수',
                           '시간가중치', '평균 발주 간격', '최초발주일', '최근발주일', '경과일']
        final_df = agg_df[display_columns].sort_values(
            by='점수', ascending=False)
        st.dataframe(final_df, use_container_width=True)

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
