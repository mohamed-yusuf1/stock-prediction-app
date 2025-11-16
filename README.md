# 📈 Stock Price Prediction System - تداول السعودية

A sophisticated stock price prediction web application built with Streamlit and TensorFlow, specifically designed for Saudi stock market analysis.

## 🌟 Features

- **Deep Learning Models**: LSTM and MLP neural networks
- **Real-time Visualization**: Interactive charts and candlestick patterns
- **Performance Metrics**: RMSE, MSE, and R² scoring
- **Future Predictions**: 30-day price forecasting
- **Arabic Interface**: Fully localized for Arabic users
- **Data Export**: Download predictions for further analysis

## 🛠️ Installation

1. Clone the repository:

git clone https://github.com/your-username/stock-prediction-app.git
cd stock-prediction-app

2. Install dependencies:
pip install -r requirements.txt

3. Run the application:
streamlit run app.py  
or 
python -m streamlit run app.py  #Run as a Python module
          


** 📁 Project Structure **
stock_prediction_app/
├── app.py                 # الملف الرئيسي للتطبيق
├── predictor.py           # فئة StockPredictor
├── utils.py              # الدوال المساعدة
├── components/           # مجلد للمكونات
│   ├── __init__.py
│   ├── sidebar.py        # شريط جانبي
│   ├── data_display.py   # عرض البيانات
│   └── training.py       # مكونات التدريب
└── assets/              # ملفات التنسيق
    └── styles.css


📊 Data Format
Your CSV file should contain these columns:

Date: Trading date

Price: Closing price

Open: Opening price

High: Highest price

Low: Lowest price

Vol.: Trading volume

Change %: Percentage change



🎯 Usage
Upload your stock data CSV file

Configure model parameters in the sidebar

Choose between LSTM or MLP models

Train the model and view performance metrics

Analyze future price predictions

Download results for further analysis

🤝 Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for improvements.