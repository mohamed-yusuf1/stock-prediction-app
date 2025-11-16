import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.model_type = None
        
    def load_data(self, file_path):
        """تحميل وتجهيز بيانات الأسهم"""
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        return df
    
    def prepare_data(self, df, time_window=60, test_ratio=0.2):
        """تحضير البيانات للتدريب"""
        # استخدام عمود 'Price' كهدف
        prices = df['Price'].values.reshape(-1, 1)
        
        # تطبيع البيانات
        scaled_data = self.scaler.fit_transform(prices)
        
        # إنشاء بيانات التدريب والاختبار
        training_data_len = int(len(scaled_data) * (1 - test_ratio))
        
        # إنشاء مجموعة بيانات التدريب
        train_data = scaled_data[0:training_data_len, :]
        
        # تقسيم إلى x_train و y_train
        x_train = []
        y_train = []
        
        for i in range(time_window, len(train_data)):
            x_train.append(train_data[i-time_window:i, 0])
            y_train.append(train_data[i, 0])
        
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        # إنشاء مجموعة بيانات الاختبار
        test_data = scaled_data[training_data_len - time_window:, :]
        x_test = []
        y_test = prices[training_data_len:, :]
        
        for i in range(time_window, len(test_data)):
            x_test.append(test_data[i-time_window:i, 0])
        
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        return x_train, y_train, x_test, y_test, training_data_len
    
    def build_lstm_model(self, time_window, lstm_units=50, dropout_rate=0.2):
        """بناء نموذج LSTM"""
        model = Sequential()
        model.add(LSTM(units=lstm_units, return_sequences=True, 
            input_shape=(time_window, 1)))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units=lstm_units, return_sequences=False))
        model.add(Dropout(dropout_rate))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                loss='mean_squared_error')
        return model
    
    def build_mlp_model(self, time_window, layers=[64, 32, 16]):
        """بناء نموذج MLP"""
        model = Sequential()
        model.add(Dense(layers[0], activation='relu', input_shape=(time_window,)))
        
        for units in layers[1:]:
            model.add(Dense(units, activation='relu'))
        
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.001), 
        loss='mean_squared_error')
        return model
    
    def train_model(self, x_train, y_train, model_type='LSTM', 
                    epochs=20, batch_size=32, time_window=60):
        """تدريب النموذج المختار"""
        self.model_type = model_type
        
        if model_type == 'LSTM':
            self.model = self.build_lstm_model(time_window)
            history = self.model.fit(x_train, y_train, 
                                batch_size=batch_size, 
                                epochs=epochs,
                                verbose=0)
        else:  # MLP
            # إعادة تشكيل البيانات لـ MLP
            x_train_mlp = x_train.reshape(x_train.shape[0], x_train.shape[1])
            self.model = self.build_mlp_model(time_window)
            history = self.model.fit(x_train_mlp, y_train, 
                                   batch_size=batch_size, 
                                   epochs=epochs,
                                   verbose=0)
        
        return history
    
    def predict(self, x_test, model_type='LSTM'):
        """إجراء التنبؤات"""
        if model_type == 'LSTM':
            predictions = self.model.predict(x_test, verbose=0)
        else:  # MLP
            x_test_mlp = x_test.reshape(x_test.shape[0], x_test.shape[1])
            predictions = self.model.predict(x_test_mlp, verbose=0)
        
        predictions = self.scaler.inverse_transform(predictions)
        return predictions
    
    def calculate_metrics(self, y_true, y_pred):
        """حساب مقاييس الأداء"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return mse, rmse, r2