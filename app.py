import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import plotly.express as px
import io

# ----------------------------------------------------------------------
# 데이터 처리 및 분석 함수
# ----------------------------------------------------------------------


@st.cache_data
def get_date_range(uploaded_file):
    """파일을 가볍게 읽어 날짜 범위(전체 기간)만 계산합니다."""
    try:
        uploaded_file.seek(0)
        if uploaded_file.name.endswith('.csv'):
            df_preview = pd.read_csv(
                uploaded_file, low_memory=False, usecols=lambda c: '날짜' in c)
        else:
            df_preview = pd.read_excel(
                uploaded_file, usecols=lambda c: '날짜' in c)

        date_col = next(
            (col for col in df_preview.columns if '날짜' in col), None)
        if not date_col:
            return 0

        df_preview[date_col] = pd.to_datetime(
            df_preview[date_col], errors='coerce').dropna()
        if len(df_preview[date_col]) < 2:
            return 0

        time_span = (df_preview[date_col].max() -
                     df_preview[date_col].min()).days
        uploaded_file.seek(0)
        return time_span
    except Exception:
        return 0


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
            st.error("오류: 업로드된 파일에 분석할 데이터가 없습니다. 파일 내용을 확인해주세요.")
            return None, None
    except pd.errors.EmptyDataError:
        st.error("오류: 업로드된 파일이 비어 있거나 데이터가 없습니다. 다른 파일을 업로드해주세요.")
        return None, None
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return None, None

    date_col = next((col for col in df_raw.columns if '날짜' in col), '날짜')
    book_col = next((col for col in df_raw.columns if '도서명' in col), '도서명')
    amount_col = next((col for col in df_raw.columns if '발주량' in col), '발주량')

    if not all(c in df_raw.columns for c in [date_col, book_col, amount_col]):
        st.error(
            f"분석에 필요한 컬럼('{date_col}', '{book_col}', '{amount_col}') 중 일부를 찾을 수 없습니다.")
        return None, None

    df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors='coerce')
    df_raw.dropna(subset=[date_col, book_col, amount_col], inplace=True)

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

if not uploaded_file:
    st.info("👈 사이드바에서 분석할 파일을 업로드해주세요.")
    st.stop()

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

with st.sidebar:
    st.header("⚙️ 2. 분석 민감도 설정")
    st.info(f"데이터 기간: **{period_text}**")
    k_param = st.slider("최신성 민감도 (k)", 0.1, 10.0, k_default,
                        0.1, help="값이 클수록 최근 발주에 더 높은 가중치를 부여합니다.")
    lambda_param = st.slider("장기 비활성 패널티 (λ)", 0.1, 10.0,
                             lambda_default, 0.1, help="이 값은 매우 오래된 데이터에 대한 패널티로 작용합니다.")

    st.header("⚙️ 3. 등급 결정 중요도 설정")
    st.markdown("각 지표가 등급 결정에 얼마나 중요하게 작용할지 가중치를 조절합니다.")
    w_amount = st.slider("총발주량 중요도", 1, 5, 4)
    w_freq = st.slider("발주횟수 중요도", 1, 5, 4)
    w_recency = st.slider("시간가중치 중요도", 1, 5, 2)


agg_df, df_raw = load_and_process_data(
    uploaded_file, k_param, lambda_param, w_amount, w_freq, w_recency)

if agg_df is not None:
    st.success(f"✅ **{uploaded_file.name}** 파일 분석이 완료되었습니다.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["[ 📈 클러스터링 분석 ]", "[ 🌟 신규 유망 도서 발굴 ]", "[ 📊 추가 시각화 ]", "[ 📋 전체 데이터 ]"])

    with tab1:
        # ... (이전과 동일)
        st.header("K-Means 클러스터링 분석 결과")
        grade_order = [f"{i}등급" for i in range(1, 6)]
        fig = px.scatter_3d(
            agg_df, x='총발주량', y='발주횟수', z='시간가중치', color='등급',
            color_discrete_map={'1등급': '#0081CF', '2등급': '#00A1E0',
                                '3등급': '#7ECEF4', '4등급': '#B1DFF7', '5등급': '#CCCCCC'},
            category_orders={"등급": grade_order}, hover_name='도서명',
            hover_data={'등급': True, '총발주량': ':.0f', '발주횟수': ':.0f',
                        '최초발주일': '%Y-%m-%d', '평균 발주 간격': ':.1f', '도서명': False},
            title='총발주량-발주횟수-시간가중치 기반 클러스터'
        )
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=40), height=600)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("⭐ 등급별 의미와 전략", expanded=True):
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
        st.header("🌟 신규 유망 도서 발굴 필터")
        st.info("아래 조건을 조절하여 '새롭고, 꾸준한' 유망 도서를 직접 찾아보세요.")

        col1, col2, col3 = st.columns(3)
        with col1:
            max_days_since_first = int(agg_df['최초발주후경과일'].max())
            days_since_first_limit = st.slider("출시 기간 필터 (최초 발주 후 경과일)", 0, max_days_since_first, min(
                180, max_days_since_first), help="이 슬라이더로 설정한 일수 이내에 최초로 발주된 '신상' 도서들만 필터링합니다.")
        with col2:
            min_freq_limit = st.slider("최소 발주 횟수 필터", 1, int(
                agg_df['발주횟수'].max()), 3, help="적어도 여기서 설정한 횟수 이상 발주된 도서만 필터링하여, 일회성 발주를 거릅니다.")
        with col3:
            max_interval = int(agg_df['평균 발주 간격'].dropna().max(
            )) if not agg_df['평균 발주 간격'].isna().all() else 90
            interval_limit = st.slider("최대 평균 발주 간격 필터", 1, max_interval, min(
                30, max_interval), help="발주 사이의 평균 기간이 여기서 설정한 일수보다 짧은, '꾸준한' 도서들만 필터링합니다.")

        promising_books_df = agg_df[
            (agg_df['최초발주후경과일'] <= days_since_first_limit) &
            (agg_df['발주횟수'] >= min_freq_limit) &
            (agg_df['평균 발주 간격'].fillna(interval_limit + 1) <= interval_limit)
        ].sort_values(by='평균 발주 간격')

        st.subheader(f"필터링 결과: 총 {len(promising_books_df)}권의 유망 도서를 찾았습니다.")

        # --- 개별 도서 판매 추이 그래프 ---
        book_col_name = next(
            (col for col in df_raw.columns if '도서명' in col), '도서명')
        date_col_name = next(
            (col for col in df_raw.columns if '날짜' in col), '날짜')
        amount_col_name = next(
            (col for col in df_raw.columns if '발주량' in col), '발주량')

        for _, row in promising_books_df.iterrows():
            book_title = row[book_col_name]
            with st.expander(f"'{book_title}' (평균 {row['평균 발주 간격']:.1f}일 간격 / 총 {row['발주횟수']}회 발주)"):
                history_df = df_raw[df_raw[book_col_name] == book_title]
                fig = px.bar(history_df, x=date_col_name,
                             y=amount_col_name, title=f"'{book_title}' 일별 발주량 추이")
                fig.update_layout(yaxis_title="발주량")
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("데이터 인사이트 시각화")
        date_col = next((col for col in df_raw.columns if '날짜' in col), '날짜')
        amount_col = next(
            (col for col in df_raw.columns if '발주량' in col), '발주량')
        grade_order = [f"{i}등급" for i in range(1, 6)]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("① 월별 발주 트렌드")
            monthly_orders = df_raw.set_index(date_col).resample('ME')[
                amount_col].sum().reset_index()
            fig_line_month = px.line(monthly_orders, x=date_col, y=amount_col, markers=True,
                                     title="월별 총 발주량 변화", labels={date_col: '월', amount_col: '총 발주량'})
            st.plotly_chart(fig_line_month, use_container_width=True)
        with col2:
            st.subheader("② 주별 발주 트렌드")
            weekly_orders = df_raw.set_index(date_col).resample(
                'W-Mon')[amount_col].sum().reset_index()
            fig_line_week = px.line(weekly_orders, x=date_col, y=amount_col, markers=True,
                                    title="주별 총 발주량 변화", labels={date_col: '주', amount_col: '총 발주량'})
            st.plotly_chart(fig_line_week, use_container_width=True)

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
