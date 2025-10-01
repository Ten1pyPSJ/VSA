import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import datetime
from matplotlib.lines import Line2D
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout, Lambda
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

# Настройки
symbol = 'BTC/USDT'
exchange_id = 'binance'
days = 120
window_size = 24
prediction_horizon = 4

def fetch_ohlcv_data(timeframe, days):
    exchange = getattr(ccxt, exchange_id)()
    since = exchange.parse8601((datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S'))
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def prepare_data(df_1h, df_4h):
    # Создаем отдельные скейлеры для признаков и целей
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    # Нормализация признаков
    df_1h_features = df_1h[['open', 'high', 'low', 'close', 'volume']]
    df_1h[['open', 'high', 'low', 'close', 'volume']] = feature_scaler.fit_transform(df_1h_features)
    
    df_4h_features = df_4h[['open', 'high', 'low', 'close', 'volume']]
    df_4h[['open', 'high', 'low', 'close', 'volume']] = feature_scaler.transform(df_4h_features)
    
    # Проверка и корректировка данных
    def validate_ohlc(data):
        data['high'] = np.maximum(data['high'], data['open'])
        data['low'] = np.minimum(data['low'], data['close'])
        data['high'] = np.maximum(data['high'], data['close'])
        data['low'] = np.minimum(data['low'], data['open'])
        return data
    
    df_1h = validate_ohlc(df_1h)
    df_4h = validate_ohlc(df_4h)
    
    # Нормализация целей (только OHLC)
    targets = df_1h[['open', 'high', 'low', 'close']].copy()
    targets = validate_ohlc(targets)
    targets_scaled = target_scaler.fit_transform(targets)

    # Создание временных окон
    def create_dataset(data, window_size):
        X = []
        for i in range(len(data) - window_size - prediction_horizon + 1):
            X.append(data[i:i+window_size])
        return np.array(X)

    X_1h = create_dataset(df_1h[['open', 'high', 'low', 'close', 'volume']].values, window_size)
    X_4h = create_dataset(df_4h[['open', 'high', 'low', 'close', 'volume']].values, window_size // 4)

    # Целевые переменные
    y = []
    for i in range(window_size, len(targets_scaled) - prediction_horizon + 1):
        y.append(targets_scaled[i:i+prediction_horizon])
    y = np.array(y)

    # Выравнивание размерностей
    min_len = min(len(X_1h), len(X_4h), len(y))
    X_1h = X_1h[:min_len]
    X_4h = X_4h[:min_len]
    y = y[:min_len]

    return X_1h, X_4h, y, feature_scaler, target_scaler

def build_model():
    input_1h = Input(shape=(window_size, 5))
    lstm_1h = LSTM(64, return_sequences=False)(input_1h)
    lstm_1h = Dropout(0.2)(lstm_1h)

    input_4h = Input(shape=(window_size // 4, 5))
    lstm_4h = LSTM(64, return_sequences=False)(input_4h)
    lstm_4h = Dropout(0.2)(lstm_4h)

    merged = Concatenate()([lstm_1h, lstm_4h])
    dense = Dense(64, activation='relu')(merged)
    dense = Dropout(0.2)(dense)
    
    # Модифицированный выходной слой с ограничениями
    def ohlc_activation(x):
        x = K.reshape(x, (-1, 4))  # Разделяем на отдельные свечи
        open_p = x[:, 0:1]
        high_p = K.maximum(x[:, 1:2], open_p)
        low_p = K.minimum(x[:, 2:3], open_p)
        close_p = x[:, 3:4]
        
        # Дополнительные ограничения
        high_p = K.maximum(high_p, close_p)
        low_p = K.minimum(low_p, close_p)
        
        return K.reshape(K.concatenate([open_p, high_p, low_p, close_p], axis=-1), (-1, prediction_horizon * 4))
    
    output = Dense(prediction_horizon * 4)(dense)
    output = Lambda(ohlc_activation)(output)

    model = Model(inputs=[input_1h, input_4h], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

def plot_predictions(df_1h, predictions, target_scaler):
    predictions = predictions.reshape(-1, 4)
    
    # Корректировка предсказаний перед денормализацией
    for i in range(len(predictions)):
        open_p = predictions[i, 0]
        close_p = predictions[i, 3]
        predictions[i, 1] = max(predictions[i, 1], open_p, close_p)  # high >= max(open, close)
        predictions[i, 2] = min(predictions[i, 2], open_p, close_p)  # low <= min(open, close)
    
    predictions = target_scaler.inverse_transform(predictions)
    
    # Создание DataFrame для предсказаний
    last_date = df_1h.index[-1]
    pred_dates = [last_date + pd.Timedelta(hours=i+1) for i in range(prediction_horizon)]
    pred_df = pd.DataFrame(predictions, columns=['open', 'high', 'low', 'close'], index=pred_dates)

    # Визуализация
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(16, 7))

    # Исторические свечи (последние 24 часа)
    history = df_1h[-60:]
    for idx, row in history.iterrows():
        body_color = 'green' if row['close'] >= row['open'] else 'red'
        ax.plot([mdates.date2num(idx), mdates.date2num(idx)],
                [row['low'], row['high']],
                color=body_color,
                linewidth=1.5,
                alpha=0.7)
        ax.bar(mdates.date2num(idx),
               abs(row['close'] - row['open']),
               bottom=min(row['open'], row['close']),
               width=0.02,
               color=body_color,
               edgecolor='black',
               linewidth=1)

    # Предсказанные свечи
    for idx, row in pred_df.iterrows():
        body_color = 'purple' if row['close'] >= row['open'] else 'black'
        ax.plot([mdates.date2num(idx), mdates.date2num(idx)],
                [row['low'], row['high']],
                color=body_color,
                linewidth=1.5,
                alpha=0.7)
        ax.bar(mdates.date2num(idx),
               abs(row['close'] - row['open']),
               bottom=min(row['open'], row['close']),
               width=0.02,
               color=body_color,
               edgecolor='black',
               linewidth=1)

    # Настройки графика
    ax.set_title(f'Предсказание {symbol} на 4 часа вперед', fontsize=16)
    ax.set_xlabel('Дата', fontsize=12)
    ax.set_ylabel('Цена (USDT)', fontsize=12)
    
    date_format = DateFormatter("%Y-%m-%d %H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    
    legend_elements = [
        Line2D([0], [0], color='green', lw=4, label='История: рост'),
        Line2D([0], [0], color='red', lw=4, label='История: падение'),
        Line2D([0], [0], color='purple', lw=4, label='Прогноз: рост'),
        Line2D([0], [0], color='black', lw=4, label='Прогноз: падение')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    try:
        # Загрузка данных
        df_1h = fetch_ohlcv_data('1h', days)
        df_4h = fetch_ohlcv_data('4h', days)
        
        # Подготовка данных
        X_1h, X_4h, y, feature_scaler, target_scaler = prepare_data(df_1h, df_4h)
        
        # Построение модели
        model = build_model()
        
        # Обучение модели
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        history = model.fit(
            [X_1h, X_4h], y.reshape(-1, prediction_horizon * 4),
            epochs=1,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping]
        )
        
        # Предсказание
        last_1h = X_1h[-1].reshape(1, window_size, 5)
        last_4h = X_4h[-1].reshape(1, window_size // 4, 5)
        predictions = model.predict([last_1h, last_4h])
        
        # Визуализация
        plot_predictions(df_1h, predictions, target_scaler)
        
    except Exception as e:
        print(f"Произошла ошибка: {e}")