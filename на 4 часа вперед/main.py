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
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Multiply, Activation, Reshape, Lambda, concatenate, BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from tensorflow.keras.losses import Huber

import warnings

warnings.filterwarnings('ignore')

# Настройки
symbol = 'BTC/USDT'
timeframe = '1h'
limit_days = 365  # Увеличено количество дней для обучения
exchange_id = 'binance'

def fetch_ohlcv_data():
    """Загрузка исторических данных с биржи"""
    exchange = getattr(ccxt, exchange_id)()
    now = exchange.milliseconds()
    since = exchange.parse8601((datetime.datetime.now() - datetime.timedelta(days=limit_days)).strftime('%Y-%m-%d %H:%M:%S'))
    
    print(f"Загрузка данных {symbol} с {exchange_id}...")
    all_ohlcv = []
    while since < now:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if len(ohlcv):
                since = ohlcv[-1][0] + 1
                all_ohlcv += ohlcv
            else:
                break
        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            break
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Удаление дубликатов и заполнение пропусков
    df = df[~df.index.duplicated(keep='last')]
    df = df.asfreq('1H').ffill()
    
    # Логарифмирование объема для нормализации распределения
    df['volume'] = np.log1p(df['volume'])
    
    return df

def create_sequences(data, n_steps):
    """Создание последовательностей для обучения"""
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i])
        y.append(data[i, :4])  # open, high, low, close
    return np.array(X), np.array(y)

def prepare_data(df, n_steps=48, test_size=0.15):
    """Подготовка и нормализация данных"""
    scaler = MinMaxScaler()
    cols_to_scale = ['open', 'high', 'low', 'close', 'volume']
    scaled_data = scaler.fit_transform(df[cols_to_scale])
    
    # Создание последовательностей
    X, y = create_sequences(scaled_data, n_steps)
    
    # Разделение на train/validation
    train_size = int(len(X) * (1 - test_size))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_val, y_val, scaler

def attention_block(inputs):
    """Улучшенный механизм внимания"""
    # Внимание с двумя слоями и регуляризацией
    attention = Dense(64, activation='tanh', kernel_regularizer=l2(0.01))(inputs)
    attention = Dense(1, activation='relu')(attention)
    attention = Activation('softmax', name='attention_weights')(attention)
    attention = Multiply()([inputs, attention])
    attention = Lambda(lambda x: K.sum(x, axis=1))(attention)
    return attention

def create_model(input_shape):
    """Создание модели с улучшенной архитектурой"""
    inputs = Input(shape=input_shape)
    
    # Нормализация входных данных
    normalized = BatchNormalization()(inputs)
    
    # Первый LSTM слой с возвратом последовательности
    lstm1 = LSTM(256, return_sequences=True, 
                kernel_regularizer=l2(0.01),
                recurrent_regularizer=l2(0.01))(normalized)
    lstm1 = Dropout(0.2)(lstm1)
    lstm1 = BatchNormalization()(lstm1)
    
    # Второй LSTM слой
    lstm2 = LSTM(128, return_sequences=True,
                kernel_regularizer=l2(0.01),
                recurrent_regularizer=l2(0.01))(lstm1)
    lstm2 = Dropout(0.2)(lstm2)
    lstm2 = BatchNormalization()(lstm2)
    
    # Третий LSTM слой
    lstm3 = LSTM(64, return_sequences=True,
                kernel_regularizer=l2(0.01))(lstm2)
    lstm3 = Dropout(0.2)(lstm3)
    
    # Механизм внимания
    attention = attention_block(lstm3)
    
    # Полносвязные слои
    dense1 = Dense(128, activation='relu', 
                  kernel_regularizer=l2(0.01))(attention)
    dense1 = Dropout(0.1)(dense1)
    dense1 = BatchNormalization()(dense1)
    
    dense2 = Dense(64, activation='relu',
                  kernel_regularizer=l2(0.01))(dense1)
    dense2 = Dropout(0.1)(dense2)
    dense2 = BatchNormalization()(dense2)
    
    dense3 = Dense(32, activation='relu')(dense2)
    outputs = Dense(4)(dense3)  # 4 выхода: open, high, low, close
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Компиляция с оптимизированными параметрами
    model.compile(
        optimizer='adam',
        loss=Huber(),  # Явное использование Huber loss
        metrics=[
            MeanAbsoluteError(name='mae'),
            MeanSquaredError(name='mse'),
        ]
    )
    
    return model

def train_model(X_train, y_train, X_val, y_val):
    """Обучение модели с улучшенными callback'ами"""
    model = create_model((X_train.shape[1], X_train.shape[2]))
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    # Загрузка лучшей версии модели
    model.load_weights('best_model.h5')
    
    return model, history

def evaluate_model(model, X_val, y_val, scaler):
    """Оценка модели на валидационных данных"""
    val_predict = model.predict(X_val)
    
    # Денормализация прогнозов
    dummy_volume = np.zeros((val_predict.shape[0], 1))
    val_predict_full = np.hstack((val_predict, dummy_volume))
    val_predict_inv = scaler.inverse_transform(val_predict_full)[:, :4]
    
    # Денормализация истинных значений
    y_val_full = np.hstack((y_val, np.zeros((y_val.shape[0], 1))))
    y_val_inv = scaler.inverse_transform(y_val_full)[:, :4]
    
    # Расчет метрик
    metrics = {
        'MAE': np.mean(np.abs(y_val_inv - val_predict_inv)),
        'MAPE': mean_absolute_percentage_error(y_val_inv, val_predict_inv) * 100,
        'MSE': np.mean((y_val_inv - val_predict_inv)**2),
        'R2': r2_score(y_val_inv, val_predict_inv)
    }
    
    return metrics, val_predict_inv, y_val_inv

def predict_next_candle(model, last_sequence, scaler):
    """Прогнозирование следующей свечи"""
    next_pred = model.predict(last_sequence)
    next_pred_full = np.hstack((next_pred, np.zeros((1, 1))))
    next_candle_values = scaler.inverse_transform(next_pred_full)[0, :4]
    
    return {
        'open': next_candle_values[0],
        'high': next_candle_values[1],
        'low': next_candle_values[2],
        'close': next_candle_values[3]
    }

def plot_results(df, train_predictions, val_predictions, y_val, next_candle, metrics):
    """Визуализация результатов"""
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Конвертация дат для matplotlib
    dates = mdates.date2num(df.index)
    
    # Построение свечей
    for i in range(len(df)):
        row = df.iloc[i]
        if row['close'] >= row['open']:
            color = 'green'
            bottom = row['open']
            top = row['close']
        else:
            color = 'red'
            bottom = row['close']
            top = row['open']
        
        # Тени свечей
        ax.plot([dates[i], dates[i]], 
                [row['low'], row['high']], 
                color=color, 
                linewidth=1,
                alpha=0.7)
        
        # Тела свечей
        ax.bar(dates[i], 
               top - bottom, 
               bottom=bottom, 
               width=0.02, 
               color=color, 
               edgecolor='black',
               linewidth=0.5)
    
    # Прогнозы на тренировочных данных
    train_dates = mdates.date2num(train_predictions.index)
    ax.plot(train_dates, 
            train_predictions['close_pred'], 
            'b-', 
            linewidth=1.5, 
            alpha=0.7,
            label='Прогноз (обучение)')
    
    # Прогнозы на валидационных данных
    val_dates = mdates.date2num(df.index[-len(val_predictions):])
    ax.plot(val_dates, 
            val_predictions[:, 3],  # close price
            'm-', 
            linewidth=1.5, 
            alpha=0.7,
            label='Прогноз (валидация)')
    
    # Фактические значения на валидации
    ax.plot(val_dates, 
            y_val[:, 3], 
            'g-', 
            linewidth=1, 
            alpha=0.5,
            label='Факт (валидация)')
    
    # Прогноз следующей свечи
    if next_candle:
        next_time = df.index[-1] + pd.Timedelta(hours=1)
        next_date = mdates.date2num(next_time)
        
        # Определение цвета свечи
        candle_color = '#800080' if next_candle['close'] >= next_candle['open'] else 'black'
        
        # Тени
        ax.plot([next_date, next_date],
                [next_candle['low'], next_candle['high']],
                color=candle_color,
                linewidth=1.5,
                alpha=0.7)
        
        # Тело
        body_bottom = min(next_candle['open'], next_candle['close'])
        body_top = max(next_candle['open'], next_candle['close'])
        ax.bar(next_date,
               body_top - body_bottom,
               bottom=body_bottom,
               width=0.02,
               color=candle_color,
               edgecolor='black',
               linewidth=0.5,
               label='Прогноз след. свечи')
    
    # Настройки графика
    ax.set_title(f'Прогнозирование {symbol} ({timeframe})', fontsize=16)
    ax.set_xlabel('Дата', fontsize=12)
    ax.set_ylabel('Цена (USDT)', fontsize=12)
    
    # Форматирование дат
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M"))
    fig.autofmt_xdate()
    
    # Добавление метрик
    if metrics:
        metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        ax.text(0.02, 0.98, metrics_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    # Легенда
    ax.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    try:
        # 1. Загрузка данных
        print("Загрузка данных...")
        data = fetch_ohlcv_data()
        print("Данные успешно загружены:")
        print(data.tail())
        
        # 2. Подготовка данных
        print("\nПодготовка данных...")
        X_train, y_train, X_val, y_val, scaler = prepare_data(data)
        print(f"Размеры данных: Train {X_train.shape}, Validation {X_val.shape}")
        
        # 3. Обучение модели
        print("\nОбучение модели...")
        model, history = train_model(X_train, y_train, X_val, y_val)
        
        # 4. Оценка модели
        print("\nОценка модели...")
        metrics, val_predictions, y_val_inv = evaluate_model(model, X_val, y_val, scaler)
        print("\nМетрики на валидационных данных:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        
        # 5. Прогнозирование на всех данных для визуализации
        train_predict = model.predict(X_train)
        train_predict_full = np.hstack((train_predict, np.zeros((train_predict.shape[0], 1))))
        train_predict_inv = scaler.inverse_transform(train_predict_full)[:, :4]
        
        train_predict_df = pd.DataFrame(train_predict_inv, 
                                      columns=['open_pred', 'high_pred', 'low_pred', 'close_pred'],
                                      index=data.index[48:48+len(train_predict)])  # 48 - n_steps
        
        # 6. Прогнозирование следующей свечи
        last_sequence = np.expand_dims(X_val[-1], axis=0)
        next_candle = predict_next_candle(model, last_sequence, scaler)
        next_candle_time = data.index[-1] + pd.Timedelta(hours=1)
        
        print("\nПрогноз на следующий период:")
        print(f"Время: {next_candle_time}")
        print(f"Открытие: {next_candle['open']:.2f}")
        print(f"Максимум: {next_candle['high']:.2f}")
        print(f"Минимум: {next_candle['low']:.2f}")
        print(f"Закрытие: {next_candle['close']:.2f}")
        
        # 7. Визуализация результатов
        print("\nПостроение графиков...")
        plot_results(
            data,
            train_predict_df[['close_pred']],
            val_predictions,
            y_val_inv,
            {'timestamp': next_candle_time, **next_candle},
            metrics
        )
        
    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")

if __name__ == '__main__':
    main()