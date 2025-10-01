import ccxt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import datetime
from matplotlib.lines import Line2D
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Multiply, Activation, Reshape, Lambda, concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

# Настройки
symbol = 'BTC/USDT'
timeframe = '1h'
limit_days = 30
exchange_id = 'binance'

def fetch_ohlcv_data():
    exchange = getattr(ccxt, exchange_id)()
    now = exchange.milliseconds()
    since = exchange.parse8601((datetime.datetime.now() - datetime.timedelta(days=limit_days)).strftime('%Y-%m-%d %H:%M:%S'))
    
    print(f"Загрузка актуальных данных {symbol} с {exchange_id}...")
    all_ohlcv = []
    while since < now:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
        if len(ohlcv):
            since = ohlcv[-1][0] + 1
            all_ohlcv += ohlcv
        else:
            break
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Удаляем возможные дубликаты
    df = df[~df.index.duplicated(keep='last')]
    
    return df

# Слой внимания
def attention_block(inputs):
    attention = Dense(1, activation='tanh')(inputs)
    attention = Activation('softmax')(attention)
    attention = Multiply()([inputs, attention])
    attention = Lambda(lambda x: K.sum(x, axis=1))(attention)
    return attention

def create_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # LSTM слои
    lstm1 = LSTM(128, return_sequences=True)(inputs)
    lstm1 = Dropout(0.2)(lstm1)
    
    lstm2 = LSTM(64, return_sequences=True)(lstm1)
    lstm2 = Dropout(0.2)(lstm2)
    
    # Механизм внимания
    attention = attention_block(lstm2)
    
    # Дополнительные слои
    dense1 = Dense(64, activation='relu')(attention)
    outputs = Dense(4)(dense1)  # 4 выхода: open, high, low, close
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def prepare_data(df, n_steps=24):
    # Нормализуем только нужные колонки
    scaler = MinMaxScaler()
    cols_to_scale = ['open', 'high', 'low', 'close', 'volume']
    scaled_data = scaler.fit_transform(df[cols_to_scale])
    
    X, y = [], []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i-n_steps:i])
        y.append(scaled_data[i, :4])  # open, high, low, close
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, scaler

def plot_candlestick_with_predictions(df, train_predictions=None, next_candle=None):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(16, 9))
    
    candle_width = 0.03
    
    # Рисуем фактические свечи
    for idx, row in df.iterrows():
        if row['close'] >= row['open']:
            body_color = 'green'
            shadow_color = 'green'
            body_bottom = row['open']
            body_top = row['close']
        else:
            body_color = 'red'
            shadow_color = 'red'
            body_bottom = row['close']
            body_top = row['open']
        
        ax.plot([mdates.date2num(idx), mdates.date2num(idx)], 
                [row['low'], row['high']], 
                color=shadow_color, 
                linewidth=1.5,
                alpha=0.7)
        
        ax.bar(mdates.date2num(idx), 
               body_top - body_bottom, 
               bottom=body_bottom, 
               width=candle_width, 
               color=body_color, 
               edgecolor='black',
               linewidth=1)
    
    # Рисуем прогнозы обучения (пунктирная линия)
    if train_predictions is not None:
        ax.plot(train_predictions.index, 
                train_predictions['close_pred'], 
                'b--', 
                linewidth=1.5, 
                alpha=0.8,
                label='Прогноз модели (обучение)')
    
    # Рисуем прогнозируемую свечу
    if next_candle is not None:
        timestamp = next_candle['timestamp']
        open_val = next_candle['open']
        high_val = next_candle['high']
        low_val = next_candle['low']
        close_val = next_candle['close']
        
        # Определяем цвет свечи
        candle_color = '#800080' if close_val >= open_val else 'black'
        
        # Рисуем тень
        ax.plot([mdates.date2num(timestamp), mdates.date2num(timestamp)],
                [low_val, high_val],
                color=candle_color,
                linewidth=1.5,
                alpha=0.7)
        
        # Рисуем тело
        body_bottom = min(open_val, close_val)
        body_top = max(open_val, close_val)
        ax.bar(mdates.date2num(timestamp),
               body_top - body_bottom,
               bottom=body_bottom,
               width=candle_width,
               color=candle_color,
               edgecolor='black',
               linewidth=1,
               label='Прогноз на след. период')
    
    ax.set_title(f'Свечной график {symbol} с прогнозами ({timeframe}, {exchange_id})', fontsize=16)
    ax.set_xlabel('Дата', fontsize=12)
    ax.set_ylabel('Цена (USDT)', fontsize=12)
    
    date_format = DateFormatter("%Y-%m-%d %H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    
    # Настраиваем легенду
    legend_elements = [
        Line2D([0], [0], color='green', lw=4, label='Рост (бычья)'),
        Line2D([0], [0], color='red', lw=4, label='Падение (медвежья)'),
        Line2D([0], [0], color='blue', linestyle='--', lw=2, label='Прогноз модели (обучение)'),
        Line2D([0], [0], color='#800080', lw=4, label='Прогноз: рост'),
        Line2D([0], [0], color='black', lw=4, label='Прогноз: падение')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    try:
        # Загрузка данных
        data = fetch_ohlcv_data()
        print("Последние 5 актуальных свечей:")
        print(data.tail())
        
        # Подготовка данных для модели
        n_steps = 24
        X, y, scaler = prepare_data(data, n_steps)
        
        # Разделение данных
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Создание и обучение модели
        model = create_model((X_train.shape[1], X_train.shape[2]))
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, 
                 validation_data=(X_val, y_val),
                 epochs=100, 
                 batch_size=32, 
                 callbacks=[early_stop],
                 verbose=1)
        
        # Прогнозирование на обучающих данных
        train_predict = model.predict(X)
        
        # Денормализация прогнозов
        dummy_volume = np.zeros((train_predict.shape[0], 1))
        train_predict_full = np.hstack((train_predict, dummy_volume))
        train_predict_inv = scaler.inverse_transform(train_predict_full)[:, :4]
        
        # Создаем DataFrame для прогнозов
        train_predict_df = pd.DataFrame(train_predict_inv, 
                                      columns=['open_pred', 'high_pred', 'low_pred', 'close_pred'],
                                      index=data.index[n_steps:])
        
        # Прогнозирование следующей свечи
        last_sequence = X[-1].reshape(1, n_steps, 5)
        next_pred = model.predict(last_sequence)
        
        # Денормализация прогноза
        dummy_volume_next = np.zeros((1, 1))
        next_pred_full = np.hstack((next_pred, dummy_volume_next))
        next_candle_values = scaler.inverse_transform(next_pred_full)[0, :4]
        
        next_candle = {
            'timestamp': data.index[-1] + pd.Timedelta(hours=1),
            'open': next_candle_values[0],
            'high': next_candle_values[1],
            'low': next_candle_values[2],
            'close': next_candle_values[3]
        }
        
        # Остальной код остается без изменений
        print("\nПрогноз на следующий период:")
        print(f"Открытие: {next_candle['open']:.2f}")
        print(f"Максимум: {next_candle['high']:.2f}")
        print(f"Минимум: {next_candle['low']:.2f}")
        print(f"Закрытие: {next_candle['close']:.2f}")
        
        plot_candlestick_with_predictions(data, train_predict_df[['close_pred']], next_candle)
        
    except Exception as e:
        print(f"Произошла ошибка: {e}")