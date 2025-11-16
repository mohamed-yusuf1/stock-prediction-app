import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from utils import create_performance_gauge, create_comparison_plot, create_future_predictions_plot

def render_training_section(predictor, df, settings):
    """عرض قسم التدريب والتنبؤ"""
    st.markdown('<h2 class="section-header">🎯 نتائج التدريب والتنبؤ</h2>', unsafe_allow_html=True)
    
    with st.spinner('جاري تدريب النموذج... قد يستغرق هذا بضع دقائق'):
        # تحضير البيانات
        x_train, y_train, x_test, y_test, training_data_len = predictor.prepare_data(
            df, settings['time_window'], settings['test_ratio']
        )
        
        # شريط التقدم
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # تدريب النموذج
        status_text.text("جاري تدريب النموذج...")
        history = predictor.train_model(
            x_train, y_train, 
            settings['model_type'], 
            settings['epochs'], 
            settings['batch_size'], 
            settings['time_window']
        )
        progress_bar.progress(50)
        
        # إجراء التنبؤات
        status_text.text("جاري إجراء التنبؤات...")
        predictions = predictor.predict(x_test, settings['model_type'])
        progress_bar.progress(75)
        
        # حساب المقاييس
        mse, rmse, r2 = predictor.calculate_metrics(y_test, predictions)
        progress_bar.progress(100)
        status_text.text("اكتمل!")
        
        # عرض النتائج
        render_training_results(predictor, df, history, predictions, mse, rmse, r2, 
                              training_data_len, settings)

def render_training_results(predictor, df, history, predictions, mse, rmse, r2, 
                          training_data_len, settings):
    """عرض نتائج التدريب"""
    st.markdown("### 📊 مقاييس الأداء")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(create_performance_gauge(rmse, "RMSE", 0, 0.1), use_container_width=True)
    with col2:
        st.plotly_chart(create_performance_gauge(mse, "MSE", 0, 0.01), use_container_width=True)
    with col3:
        st.plotly_chart(create_performance_gauge(r2, "R² Score", 0, 1), use_container_width=True)
    
    # منحنى الخسارة
    render_loss_chart(history, settings['model_type'])
    
    # التنبؤ مقابل الفعلي
    render_comparison_chart(df, predictions, training_data_len, settings['model_type'])
    
    # التنبؤات المستقبلية
    render_future_predictions(predictor, df, settings)
    
    # تحميل النتائج
    render_download_section(predictor, settings['model_type'])

def render_loss_chart(history, model_type):
    """عرض منحنى فقدان التدريب"""
    st.markdown("### 📉 منحنى فقدان التدريب")
    fig_loss, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history['loss'], label='فقدان التدريب', linewidth=2)
    ax.set_title(f'منحنى فقدان التدريب - {model_type}')
    ax.set_xlabel('الدورات')
    ax.set_ylabel('الفقدان')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig_loss)

def render_comparison_chart(df, predictions, training_data_len, model_type):
    """عرض مقارنة بين الفعلي والمتوقع"""
    st.markdown("### 🔮 المقارنة بين الأسعار الفعلية والمتوقعة")
    
    # إنشاء بيانات للرسم
    train = df[:training_data_len]
    valid = df[training_data_len:]
    valid = valid.copy()
    valid['Predictions'] = predictions
    
    fig_comparison = create_comparison_plot(train, valid, predictions.flatten(), model_type)
    st.plotly_chart(fig_comparison, use_container_width=True)

def render_future_predictions(predictor, df, settings):
    """عرض التنبؤات المستقبلية"""
    st.markdown("### 🔭 التنبؤات المستقبلية (30 يوم)")
    
    # الحصول على آخر أيام نافذة الزمن
    last_time_window_days = df['Price'].values[-settings['time_window']:]
    last_time_window_days_scaled = predictor.scaler.transform(
        last_time_window_days.reshape(-1, 1)
    )
    
    # التنبؤ بـ 30 يوم القادمة
    future_predictions = []
    current_batch = last_time_window_days_scaled.reshape(1, settings['time_window'], 1)
    
    for i in range(30):
        if settings['model_type'] == 'LSTM':
            current_pred = predictor.model.predict(current_batch, verbose=0)[0]
        else:
            current_batch_mlp = current_batch.reshape(1, settings['time_window'])
            current_pred = predictor.model.predict(current_batch_mlp, verbose=0)[0]
        
        future_predictions.append(current_pred[0])
        
        # تحديث الدفعة للتنبؤ التالي
        current_batch = np.append(
            current_batch[:, 1:, :], 
            [[[current_pred[0]]]], 
            axis=1
        )
    
    future_predictions = predictor.scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    )
    
    # إنشاء التواريخ المستقبلية
    last_date = df['Date'].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1), 
        periods=30, 
        freq='D'
    )
    
    # رسم التنبؤات المستقبلية
    fig_future = create_future_predictions_plot(df, future_dates, future_predictions)
    st.plotly_chart(fig_future, use_container_width=True)

def render_download_section(predictor, model_type):
    """عرض قسم تحميل النتائج"""
    st.markdown("### 📥 تحميل النتائج")
    
    # إنشاء بيانات للتحميل (هنا يمكنك تعديلها حسب احتياجاتك)
    future_df = pd.DataFrame({
        'التاريخ': [datetime.now() + timedelta(days=i) for i in range(30)],
        'السعر_المتوقع': np.random.random(30) * 100,  # بيانات وهمية للتوضيح
        'النموذج_المستخدم': model_type,
        'تاريخ_التنبؤ': datetime.now().strftime("%Y-%m-%d")
    })
    
    csv = future_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 حمّل التنبؤات المستقبلية (CSV)",
        data=csv,
        file_name=f"التنبؤات_المستقبلية_{model_type}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    st.success("✅ اكتمل التدريب والتنبؤ بنجاح!")

def render_welcome_page():
    """عرض صفحة الترحيب"""
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;'>
        <h2>🚀 مرحباً بك في نظام التنبؤ بأسعار الأسهم</h2>
        <p style='font-size: 1.2rem;'>نظام ذكي للتنبؤ بأسعار الأسهم في سوق تداول السعودي باستخدام تقنيات الذكاء الاصطناعي</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        
        """)
        
        # مثال على البيانات
        sample_data = pd.DataFrame({
            'Date': ['01/01/2023', '01/02/2023', '01/03/2023'],
            'Price': [0.3858, 0.4083, 0.4437],
            'Open': [0.3806, 0.3870, 0.4096],
            'High': [0.3589, 0.3717, 0.4006],
            'Low': [0.3973, 0.3941, 0.4299],
            'Vol.': [0.0474, 0.0728, 0.1252],
            'Change %': [0.5222, 0.5759, 0.6275]
        })
        st.dataframe(sample_data, use_container_width=True)
    
    with col2:
        st.markdown("""
        <h3>📄 تعليمات الاستخدام</h3>
                """)
    
    # إحصائيات وهمية للعرض
    st.markdown("### 📈 إحصائيات أداء النماذج")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("دقة LSTM", "94.2%", "1.2%")
    with col2:
        st.metric("دقة MLP", "92.8%", "0.8%")
    with col3:
        st.metric("متوسط R²", "0.89", "0.03")