import streamlit as st
from predictor import StockPredictor
from components.sidebar import render_sidebar
from components.data_display import render_data_overview, render_data_tabs
from components.training import render_training_section, render_welcome_page
import warnings
warnings.filterwarnings('ignore')

# تنسيق الصفحة باللغة العربية
st.set_page_config(
    page_title="نظام التنبؤ بأسعار الأسهم - تداول",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تحميل التنسيقات
def load_css():
    with open('assets/styles.css', 'r', encoding='utf-8') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    # تحميل التنسيقات
    load_css()
    
    # رأس الصفحة الرئيسي
    st.markdown('<h1 class="main-header">📈 نظام التنبؤ بأسعار الأسهم - تداول السعودية</h1>', unsafe_allow_html=True)
    
    # عرض الشريط الجانبي والحصول على الإعدادات
    settings = render_sidebar()
    
    if settings['uploaded_file'] is not None:
        # تهيئة المتنبئ
        predictor = StockPredictor()
        
        # تحميل البيانات
        df = predictor.load_data(settings['uploaded_file'])
        
        # عرض نظرة عامة على البيانات
        render_data_overview(df)
        
        # علامات تبويب للعروض المختلفة
        render_data_tabs(df)
        
        # التدريب والتنبؤ
        if st.session_state.get('run_training', False):
            render_training_section(predictor, df, settings)
    
    else:
        # صفحة الترحيب
        render_welcome_page()

if __name__ == "__main__":
    main()