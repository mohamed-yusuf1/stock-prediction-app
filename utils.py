import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

def create_candlestick_chart(df):
    """إنشاء رسم الشموع اليابانية"""
    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Price'],
        name='أسعار الأسهم'
    )])
    
    fig.update_layout(
        title='الرسم البياني للشموع اليابانية',
        xaxis_title='التاريخ',
        yaxis_title='السعر',
        template='plotly_white',
        height=500
    )
    
    return fig

def create_performance_gauge(value, title, min_val, max_val):
    """إنشاء مقياس أداء تفاعلي"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_val, min_val + (max_val-min_val)*0.6], 'color': "lightgray"},
                {'range': [min_val + (max_val-min_val)*0.6, min_val + (max_val-min_val)*0.8], 'color': "gray"},
                {'range': [min_val + (max_val-min_val)*0.8, max_val], 'color': "darkgray"}
            ],
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_future_predictions_plot(df, future_dates, future_predictions):
    """إنشاء رسم للتنبؤات المستقبلية"""
    fig_future = go.Figure()
    fig_future.add_trace(go.Scatter(
        x=df['Date'][-100:], y=df['Price'][-100:],
        name='السعر التاريخي',
        line=dict(color='blue', width=2)
    ))
    fig_future.add_trace(go.Scatter(
        x=future_dates, y=future_predictions.flatten(),
        name='التنبؤات المستقبلية',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig_future.update_layout(
        title='التنبؤ بأسعار الأسهم للـ 30 يوم القادمة',
        xaxis_title='التاريخ',
        yaxis_title='السعر',
        height=500,
        template='plotly_white'
    )
    
    return fig_future

def create_comparison_plot(train, valid, predictions, model_type):
    """إنشاء رسم للمقارنة بين الفعلي والمتوقع"""
    fig_comparison = go.Figure()
    fig_comparison.add_trace(go.Scatter(
        x=train['Date'], y=train['Price'],
        name='بيانات التدريب',
        line=dict(color='blue', width=2)
    ))
    fig_comparison.add_trace(go.Scatter(
        x=valid['Date'], y=valid['Price'],
        name='السعر الفعلي',
        line=dict(color='green', width=2)
    ))
    fig_comparison.add_trace(go.Scatter(
        x=valid['Date'], y=predictions,
        name='السعر المتوقع',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig_comparison.update_layout(
        title=f'مقارنة الأسعار الفعلية والمتوقعة - {model_type}',
        xaxis_title='التاريخ',
        yaxis_title='السعر',
        height=500,
        template='plotly_white'
    )
    
    return fig_comparison