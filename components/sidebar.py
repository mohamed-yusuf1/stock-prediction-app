import streamlit as st

def render_sidebar():
    """عرض الشريط الجانبي وإرجاع الإعدادات"""
    with st.sidebar:
        st.markdown("### ⚙️ إعدادات النموذج")
        
        # تحميل الملف
        uploaded_file = st.file_uploader("📤 حمّل ملف بيانات الأسهم (CSV)", type=['csv'])
        
        st.markdown("---")
        st.markdown("### 🎛️ معاملات التدريب")
        
        col1, col2 = st.columns(2)
        with col1:
            time_window = st.slider("نافذة الزمن (أيام)", 30, 120, 60)
            test_ratio = st.slider("نسبة الاختبار", 0.1, 0.4, 0.2, 0.05)
        with col2:
            epochs = st.slider("عدد الدورات", 10, 100, 20)
            batch_size = st.slider("حجم الدفعة", 16, 64, 32)
        
        model_type = st.selectbox("اختر النموذج", ["LSTM", "MLP"])
        
        st.markdown("---")
        
        if st.button("🚀 ابدأ التدريب", type="primary", use_container_width=True):
            st.session_state.run_training = True
        else:
            if 'run_training' not in st.session_state:
                st.session_state.run_training = False
    
    return {
        'uploaded_file': uploaded_file,
        'time_window': time_window,
        'test_ratio': test_ratio,
        'epochs': epochs,
        'batch_size': batch_size,
        'model_type': model_type
    }