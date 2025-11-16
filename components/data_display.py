import streamlit as st
import plotly.express as px
from utils import create_candlestick_chart

def render_data_overview(df):
    """عرض نظرة عامة على البيانات"""
    st.markdown('<h2 class="section-header">📊 نظرة عامة على البيانات</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("إجمالي السجلات", len(df))
    with col2:
        st.metric("الفترة الزمنية", f"{df['Date'].min().date()} إلى {df['Date'].max().date()}")
    with col3:
        st.metric("أقل سعر", f"${df['Price'].min():.4f}")
    with col4:
        st.metric("أعلى سعر", f"${df['Price'].max():.4f}")

def render_data_tabs(df):
    """عرض علامات تبويب البيانات"""
    tab1, tab2, tab3 = st.tabs(["📋 عرض البيانات", "📈 الرسوم البيانية", "🔍 التحليل الإحصائي"])
    
    with tab1:
        st.dataframe(df, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            # رسم بياني للأسعار
            fig1 = px.line(df, x='Date', y='Price', title='تطور سعر السهم مع الوقت')
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # رسم الشموع اليابانية
            candlestick_fig = create_candlestick_chart(df)
            st.plotly_chart(candlestick_fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            # توزيع الأسعار
            fig_hist = px.histogram(df, x='Price', title='توزيع الأسعار')
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # إحصائيات描述ية
            st.subheader("الإحصائيات الوصفية")
            st.dataframe(df.describe(), use_container_width=True)