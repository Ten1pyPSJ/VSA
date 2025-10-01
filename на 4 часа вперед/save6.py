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
from tensorflow.keras.layers import (Input, Bidirectional, LSTM, Dense, Dropout, Multiply, 
                                    Activation, Reshape, Lambda, concatenate,
                                    BatchNormalization, Masking)
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import mean_absolute_percentage_error, r2_score, accuracy_score

# Настройки
symbol = 'BTC/USDT'
timeframe = '1h'
limit_days = 1095
exchange_id = 'binance'
EPOCH = 1
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
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)  # Добавлен параметр utc=True
    df.set_index('timestamp', inplace=True)
    
    # Удаляем возможные дубликаты
    df = df[~df.index.duplicated(keep='last')]
    
    # Определяем незакрытую свечу (последняя свеча)
    current_time = pd.Timestamp.now(tz='UTC')
    last_candle_time = df.index[-1]
    df['is_open'] = 0
    
    # Убедимся, что оба времени в UTC
    if isinstance(last_candle_time, pd.Timestamp) and last_candle_time.tz is None:
        last_candle_time = last_candle_time.tz_localize('UTC')
    
    if (current_time - last_candle_time) < pd.Timedelta(timeframe):
        df.loc[df.index[-1], 'is_open'] = 1
    
    return df

def attention_block(inputs):
    # Механизм внимания
    attention = Dense(1, activation='sigmoid')(inputs)
    attention = Multiply()([inputs, attention])
    attention = Lambda(lambda x: K.sum(x, axis=1))(attention)
    return attention

def create_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Маскирование незакрытых свечей
    masked = Masking(mask_value=-1)(inputs)
    
    # Bidirectional LSTM слои с L1 регуляризацией
    lstm1 = Bidirectional(LSTM(128, return_sequences=True, 
                             kernel_regularizer=l1_l2(l1=0.001, l2=0.001)))(masked)
    lstm1 = BatchNormalization()(lstm1)
    lstm1 = Dropout(0.5)(lstm1)
    
    lstm2 = Bidirectional(LSTM(64, return_sequences=True, 
                             kernel_regularizer=l1_l2(l1=0.001, l2=0.001)))(lstm1)
    lstm2 = BatchNormalization()(lstm2)
    lstm2 = Dropout(0.5)(lstm2)
    
    lstm3 = Bidirectional(LSTM(32, return_sequences=True, 
                             kernel_regularizer=l1_l2(l1=0.001, l2=0.001)))(lstm2)
    lstm3 = BatchNormalization()(lstm3)
    lstm3 = Dropout(0.5)(lstm3)
    
    # Механизм внимания
    attention = attention_block(lstm3)
    
    # Дополнительные слои
    dense1 = Dense(64, activation='relu', 
                  kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(attention)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.4)(dense1)
    
    dense2 = Dense(32, activation='relu', 
                  kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.4)(dense2)
    
    # Два выхода: регрессия и классификация
    regression_output = Dense(4, name='regression')(dense2)  # open, high, low, close
    classification_output = Dense(1, activation='sigmoid', name='classification')(dense2)  # направление
    
    model = Model(inputs=inputs, outputs=[regression_output, classification_output])
    model.compile(
        optimizer='adam',
        loss={
            'regression': 'mse',
            'classification': 'binary_crossentropy'
        },
        metrics={
            'regression': [MeanAbsoluteError(name='mae')],
            'classification': ['accuracy']
        },
        loss_weights=[0.7, 0.3]  # Веса для loss
    )
    return model

def calculate_vsa_features(df):
    """Расширенные признаки VSA с новыми фичами"""
    # Базовые параметры
    df['spread'] = df['high'] - df['low']
    df['avg_spread_5'] = df['spread'].rolling(5).mean()
    df['relative_spread'] = df['spread'] / df['avg_spread_5']
    
    # Базовые признаки объема
    df['volume_ma_10'] = df['volume'].rolling(10).mean()
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    
    # --- Бычий/медвежий объем ---
    df['bull_volume'] = np.where(
        ((df['close'] > df['open']) & (df['volume'] > df['volume'].shift(1))) |
        ((df['close'] < df['open']) & (df['volume'] < df['volume'].shift(1))),
        df['volume'], 
        0
    )
    
    df['bear_volume'] = np.where(
        ((df['close'] < df['open']) & (df['volume'] > df['volume'].shift(1))) |
        ((df['close'] > df['open']) & (df['volume'] < df['volume'].shift(1))),
        df['volume'],
        0
    )
    
    # Отношение к среднему объему
    df['bull_volume_ratio'] = df['bull_volume'] / df['volume_ma_10']
    df['bear_volume_ratio'] = df['bear_volume'] / df['volume_ma_10']
    
    # --- Улучшенные сигналы VSA ---
    df['is_weak_bar'] = ((df['close'] < df['open']) & 
                        (df['volume'] > df['volume_ma_10'] * 1.2)).astype(int)
    
    df['is_strong_bar'] = ((df['close'] > df['open']) & 
                          (df['volume'] > df['volume_ma_10'] * 1.5)).astype(int)
    
    # Импульс тренда
    df['volume_impulse'] = df['volume'] * (df['close'] - df['open']) / df['spread']
    
    # Кластеры спроса/предложения
    df['demand_zone'] = ((df['close'] > df['open']) & 
                        (df['bull_volume_ratio'] > 1.5) &
                        (df['volume'] > df['volume_ma_10'])).astype(int)
    
    df['supply_zone'] = ((df['close'] < df['open']) & 
                        (df['bear_volume_ratio'] > 1.5) &
                        (df['volume'] > df['volume_ma_10'])).astype(int)
    
    # Дефицит спроса/предложения
    df['demand_deficit'] = ((df['close'] < df['open']) & 
                          (df['volume'] < df['volume_ma_10'] * 0.6)).astype(int)
    
    df['supply_deficit'] = ((df['close'] > df['open']) & 
                          (df['volume'] < df['volume_ma_10'] * 0.6)).astype(int)
    
    # Сигналы продолжения тренда
    df['uptrend_continuation'] = ((df['close'] > df['open']) & 
                                (df['close'] > df['close'].shift(1)) & 
                                (df['bull_volume_ratio'] > 1.2)).astype(int)
    
    df['downtrend_continuation'] = ((df['close'] < df['open']) & 
                                  (df['close'] < df['close'].shift(1)) & 
                                  (df['bear_volume_ratio'] > 1.2)).astype(int)
    
    # ===== ДОБАВЛЕННЫЕ СИГНАЛЫ VSA =====
    df['spring'] = ((df['low'] < df['low'].shift(1)) & 
                   (df['close'] > (df['open'].shift(1) + df['close'].shift(1)) / 2) & 
                   (df['volume'] > df['volume_ma_10'] * 1.3)).astype(int)
    
    df['upthrust'] = ((df['high'] > df['high'].shift(1)) & 
                     (df['close'] < (df['open'].shift(1) + df['close'].shift(1)) / 2) & 
                     (df['volume'] > df['volume_ma_10'] * 1.3)).astype(int)
    
    df['bullish_engulfing'] = ((df['close'] > df['open']) & 
                              (df['open'] < df['close'].shift(1)) & 
                              (df['close'] > df['open'].shift(1)) & 
                              (df['volume'] > df['volume_ma_10'] * 1.4)).astype(int)
    
    df['bearish_engulfing'] = ((df['close'] < df['open']) & 
                              (df['open'] > df['close'].shift(1)) & 
                              (df['close'] < df['open'].shift(1)) & 
                              (df['volume'] > df['volume_ma_10'] * 1.4)).astype(int)
    
    df['buying_climax'] = ((df['high'] > df['high'].shift(1)) & 
                          (df['low'] < df['low'].shift(1)) & 
                          (df['close'] < df['close'].shift(1)) & 
                          (df['volume'] > df['volume_ma_10'] * 2.0)).astype(int)
    
    df['selling_climax'] = ((df['high'] > df['high'].shift(1)) & 
                           (df['low'] < df['low'].shift(1)) & 
                           (df['close'] > df['close'].shift(1)) & 
                           (df['volume'] > df['volume_ma_10'] * 2.0)).astype(int)
    
    df['market_balance'] = np.where(
        df['bull_volume_ratio'] > df['bear_volume_ratio'],
        df['bull_volume_ratio'] - df['bear_volume_ratio'],
        df['bear_volume_ratio'] - df['bull_volume_ratio']
    )
    
    df['accumulation'] = ((df['close'] > df['open']) & 
                         (df['close'] > df['close'].shift(1)) & 
                         (df['volume'] > df['volume_ma_10'] * 1.2) & 
                         (df['bull_volume_ratio'] > 1.5)).astype(int)
    
    df['distribution'] = ((df['close'] < df['open']) & 
                         (df['close'] < df['close'].shift(1)) & 
                         (df['volume'] > df['volume_ma_10'] * 1.2) & 
                         (df['bear_volume_ratio'] > 1.5)).astype(int)
    
    # ===== НОВЫЕ ПРИЗНАКИ =====
    # Осциллятор объема
    df['volume_oscillator'] = (df['volume'] - df['volume_ma_10']) / df['volume_ma_10']
    
    # Корреляция цена/объем
    df['price_volume_corr'] = df['close'].rolling(20).corr(df['volume'])
    
    # VWAP и отклонение
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['cum_vol_price'] = (df['typical_price'] * df['volume']).cumsum()
    df['cum_vol'] = df['volume'].cumsum()
    df['vwap'] = df['cum_vol_price'] / df['cum_vol']
    df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']
    
    # Момент и волатильность
    df['momentum_5'] = df['close'].pct_change(5)
    df['volatility_10'] = df['close'].pct_change().rolling(10).std()
    
    # Разворотные паттерны
    df['hammer'] = (((df['close'] - df['low']) > 1.5 * (df['high'] - df['low']).abs()) & 
                  (df['close'] > df['open']) & 
                  (df['volume'] > df['volume_ma_10'])).astype(int)
    
    df['shooting_star'] = (((df['high'] - df['close']) > 1.5 * (df['high'] - df['low']).abs()) & 
                         (df['close'] < df['open']) & 
                         (df['volume'] > df['volume_ma_10'])).astype(int)
    
    # ===== ОШИБКА ПРОГНОЗА =====
    # Прогноз с помощью скользящего среднего
    df['forecast_close'] = df['close'].rolling(5).mean().shift(1)
    df['forecast_error'] = abs(df['forecast_close'] - df['close'])
    df['forecast_direction_error'] = np.where(
        df['forecast_close'] > df['close'], 1,  # Прогноз был выше
        np.where(df['forecast_close'] < df['close'], -1, 0)  # Прогноз был ниже
    )
    
    # ===== ВРЕМЕННЫЕ ПРИЗНАКИ =====
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek  # Понедельник=0, Воскресенье=6
    
    # Заполнение пропусков
    df.fillna(0, inplace=True)
    return df

def prepare_data(df, n_steps=72):
    # Добавляем VSA признаки
    df = calculate_vsa_features(df)
    
    # Определяем признаки для масштабирования
    cols_to_scale = [
        'open', 'high', 'low', 'close', 'volume',
        'bull_volume', 'bear_volume', 'bull_volume_ratio', 'bear_volume_ratio',
        'relative_spread', 'is_weak_bar', 'is_strong_bar',
        'volume_impulse', 'demand_zone', 'supply_zone',
        'demand_deficit', 'uptrend_continuation', 'downtrend_continuation',
        'spring', 'upthrust', 'bullish_engulfing', 'bearish_engulfing',
        'buying_climax', 'selling_climax', 'market_balance',
        'accumulation', 'distribution',
        'volume_oscillator', 'price_volume_corr', 
        'vwap', 'vwap_deviation', 
        'momentum_5', 'volatility_10',
        'hammer', 'shooting_star',
        'forecast_error', 'forecast_direction_error',
        'hour', 'day_of_week'
    ]
    
    # Нормализуем данные
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[cols_to_scale])
    
    # Для незакрытых свечей заменяем значения на -1
    for i in range(len(df)):
        if df.iloc[i]['is_open'] == 1:
            scaled_data[i, :] = -1
    
    X, y_reg, y_class = [], [], []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i-n_steps:i])
        # Регрессия: OHLC следующей свечи
        y_reg.append(scaled_data[i, :4])
        # Классификация: направление (1 - рост, 0 - падение)
        direction = 1 if df['close'].iloc[i] > df['close'].iloc[i-1] else 0
        y_class.append(direction)
    
    X = np.array(X)
    y_reg = np.array(y_reg)
    y_class = np.array(y_class)
    
    return X, y_reg, y_class, scaler, len(cols_to_scale)

def get_entry_signal(data, next_candle, direction_prob):
    last_candle = data.iloc[-1]
    current_close = last_candle['close']
    predicted_change = next_candle['close'] - current_close
    
    # Сигнал от классификации
    class_signal = 'BUY' if direction_prob > 0.5 else 'SELL'
    
    # Анализ VSA сигналов
    vsa_signals = []
    
    # Бычьи сигналы
    bull_signals = 0
    if last_candle['spring']: 
        bull_signals += 1.5
        vsa_signals.append("SPRING (тест спроса)")
    if last_candle['bullish_engulfing']: 
        bull_signals += 1.2
        vsa_signals.append("BULLISH ENGULFING")
    if last_candle['accumulation']: 
        bull_signals += 1.0
        vsa_signals.append("НАКОПЛЕНИЕ")
    if last_candle['is_strong_bar']: 
        bull_signals += 0.8
        vsa_signals.append("СИЛЬНАЯ СВЕЧА")
    if last_candle['demand_zone']: 
        bull_signals += 0.7
        vsa_signals.append("ЗОНА СПРОСА")
    if last_candle['hammer']: 
        bull_signals += 1.2
        vsa_signals.append("HAMMER (разворотный паттерн)")
    if last_candle['vwap_deviation'] > 0.01:
        bull_signals += 0.5
        vsa_signals.append("Цена выше VWAP")
    if last_candle['price_volume_corr'] > 0.7:
        bull_signals += 0.8
        vsa_signals.append("Сильная корреляция цена/объем")
    
    # Медвежьи сигналы
    bear_signals = 0
    if last_candle['upthrust']: 
        bear_signals += 1.5
        vsa_signals.append("UPTHRUST (тест предложения)")
    if last_candle['bearish_engulfing']: 
        bear_signals += 1.2
        vsa_signals.append("BEARISH ENGULFING")
    if last_candle['distribution']: 
        bear_signals += 1.0
        vsa_signals.append("РАСПРЕДЕЛЕНИЕ")
    if last_candle['is_weak_bar']: 
        bear_signals += 0.8
        vsa_signals.append("СЛАБАЯ СВЕЧА")
    if last_candle['supply_zone']: 
        bear_signals += 0.7
        vsa_signals.append("ЗОНА ПРЕДЛОЖЕНИЯ")
    if last_candle['shooting_star']: 
        bear_signals += 1.2
        vsa_signals.append("SHOOTING STAR (разворотный паттерн)")
    if last_candle['vwap_deviation'] < -0.01:
        bear_signals += 0.5
        vsa_signals.append("Цена ниже VWAP")
    if last_candle['price_volume_corr'] < -0.7:
        bear_signals += 0.8
        vsa_signals.append("Обратная корреляция цена/объем")
    
    # Корректировка сигнала
    signal_strength = bull_signals - bear_signals
    confidence = abs(signal_strength)
    
    # Комбинируем сигналы классификатора и VSA
    if class_signal == 'BUY' and signal_strength > 0:
        final_signal = 'BUY'
        confidence += direction_prob
    elif class_signal == 'SELL' and signal_strength < 0:
        final_signal = 'SELL'
        confidence += (1 - direction_prob)
    else:
        final_signal = class_signal
        confidence = max(direction_prob, 1 - direction_prob)
    
    # Форматирование вывода
    vsa_info = " | ".join(vsa_signals) if vsa_signals else "Нет значимых сигналов"
    print(f"\nVSA АНАЛИЗ: {vsa_info}")
    print(f"Сила бычьих сигналов: {bull_signals:.2f}")
    print(f"Сила медвежьих сигналов: {bear_signals:.2f}")
    print(f"Сигнал классификатора: {class_signal} ({direction_prob:.2%})")
    print(f"Итоговый сигнал: {final_signal} (уверенность: {confidence:.2f})")
    
    return final_signal, confidence

def calculate_metrics(y_true, y_pred):
    metrics = {
        'MAE': np.mean(np.abs(y_true - y_pred)),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'MSE': np.mean((y_true - y_pred)**2),
        'R2': r2_score(y_true, y_pred)
    }
    return metrics

def calculate_direction_accuracy(data, predictions, n_steps):
    """Рассчитывает точность предсказания направления движения цены"""
    actual_close = data['close'].iloc[n_steps:].values
    pred_close = predictions['close_pred'].values
    
    actual_direction = np.sign(data['close'].iloc[n_steps:].values - data['close'].iloc[n_steps-1:-1].values)
    pred_direction = np.sign(pred_close - data['close'].iloc[n_steps-1:-1].values)
    
    correct = np.sum(actual_direction == pred_direction)
    total = len(actual_direction)
    accuracy = correct / total * 100
    
    return accuracy, correct, total

def plot_candlestick_with_predictions(df, train_predictions=None, next_candle=None, metrics=None, 
                                    entry_signal=None, direction_accuracy=None):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(18, 10))
    
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
    
    # Рисуем прогнозы обучения
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
        
        candle_color = '#800080' if close_val >= open_val else 'black'
        
        ax.plot([mdates.date2num(timestamp), mdates.date2num(timestamp)],
                [low_val, high_val],
                color=candle_color,
                linewidth=1.5,
                alpha=0.7)
        
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
    
    # Отображаем точку входа
    if entry_signal:
        last_timestamp = df.index[-1]
        last_price = df['close'].iloc[-1]
        
        if entry_signal == 'BUY':
            ax.plot(mdates.date2num(last_timestamp), last_price, 'g^', markersize=15, 
                   label='Точка входа: ПОКУПКА')
        elif entry_signal == 'SELL':
            ax.plot(mdates.date2num(last_timestamp), last_price, 'rv', markersize=15, 
                   label='Точка входа: ПРОДАЖА')
    
    # Добавляем метрики на график
    metrics_text = ""
    if metrics:
        metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    
    if direction_accuracy:
        metrics_text += f"\n\nТочность направления: {direction_accuracy['accuracy']:.2f}%"
        metrics_text += f"\n({direction_accuracy['correct']} из {direction_accuracy['total']})"
    
    if metrics_text:
        ax.text(0.02, 0.98, metrics_text,
               transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    ax.set_title(f'Свечной график {symbol} с прогнозами VSA ({timeframe}, {exchange_id})', fontsize=16)
    ax.set_xlabel('Дата', fontsize=12)
    ax.set_ylabel('Цена (USDT)', fontsize=12)
    
    date_format = DateFormatter("%Y-%m-%d %H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    
    legend_elements = [
        Line2D([0], [0], color='green', lw=4, label='Рост (бычья)'),
        Line2D([0], [0], color='red', lw=4, label='Падение (медвежья)'),
        Line2D([0], [0], color='blue', linestyle='--', lw=2, label='Прогноз модели (обучение)'),
        Line2D([0], [0], color='#800080', lw=4, label='Прогноз: рост'),
        Line2D([0], [0], color='black', lw=4, label='Прогноз: падение'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='g', markersize=10, label='Точка входа: ПОКУПКА'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='r', markersize=10, label='Точка входа: ПРОДАЖА')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """Визуализация истории обучения"""
    plt.figure(figsize=(15, 10))
    
    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Общая ошибка')
    plt.plot(history.history['regression_loss'], label='Ошибка регрессии')
    plt.plot(history.history['classification_loss'], label='Ошибка классификации')
    plt.title('Эволюция ошибок')
    plt.ylabel('Ошибка')
    plt.xlabel('Эпоха')
    plt.legend()
    
    # Regression MAE
    plt.subplot(2, 2, 2)
    plt.plot(history.history['regression_mae'], label='MAE обучения')
    plt.plot(history.history['val_regression_mae'], label='MAE валидации')
    plt.title('Точность регрессии (MAE)')
    plt.ylabel('MAE')
    plt.xlabel('Эпоха')
    plt.legend()
    
    # Classification Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(history.history['classification_accuracy'], label='Точность обучения')
    plt.plot(history.history['val_classification_accuracy'], label='Точность валидации')
    plt.title('Точность классификации')
    plt.ylabel('Точность')
    plt.xlabel('Эпоха')
    plt.legend()
    
    # Learning Rate
    plt.subplot(2, 2, 4)
    if 'lr' in history.history:
        plt.plot(history.history['lr'], label='Learning Rate')
        plt.title('Learning Rate')
        plt.ylabel('LR')
        plt.xlabel('Эпоха')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    try:
        # Загрузка данных
        data = fetch_ohlcv_data()
        print("Последние 5 актуальных свечей:")
        print(data.tail())
        
        # Подготовка данных для модели
        n_steps = 50
        X, y_reg, y_class, scaler, num_features = prepare_data(data, n_steps)
        num_extra_features = num_features - 4  # 4 - это OHLC
        
        # Разделение данных
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_reg_train, y_reg_val = y_reg[:train_size], y_reg[train_size:]
        y_class_train, y_class_val = y_class[:train_size], y_class[train_size:]
        
        # Создание модели
        model = create_model((X_train.shape[1], X_train.shape[2]))
        model.summary()
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.000001,
            verbose=1
        )
        
        # Обучение модели
        history = model.fit(
            X_train, 
            {'regression': y_reg_train, 'classification': y_class_train},
            validation_data=(X_val, {'regression': y_reg_val, 'classification': y_class_val}),
            epochs=EPOCH,  # Исправлено: было epochs=1
            batch_size=64,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        # Визуализация истории обучения
        plot_training_history(history)
        
        # Прогнозирование на валидационных данных
        val_reg_pred, val_class_pred = model.predict(X_val)
        
        # Денормализация прогнозов регрессии
        dummy_features = np.zeros((val_reg_pred.shape[0], num_extra_features))
        val_pred_full = np.hstack((val_reg_pred, dummy_features))
        val_pred_inv = scaler.inverse_transform(val_pred_full)[:, :4]
        
        # Денормализация истинных значений регрессии
        dummy_features_y = np.zeros((y_reg_val.shape[0], num_extra_features))
        y_val_full = np.hstack((y_reg_val, dummy_features_y))
        y_val_inv = scaler.inverse_transform(y_val_full)[:, :4]
        
        # Расчет метрик регрессии
        metrics = calculate_metrics(y_val_inv, val_pred_inv)
        print("\nМетрики качества регрессии на валидационных данных:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        
        # Расчет точности классификации
        class_accuracy = accuracy_score(y_class_val, val_class_pred > 0.5)
        print(f"\nТочность классификации: {class_accuracy:.2%}")
        
        # Прогнозирование на всех данных для визуализации
        train_reg_pred, _ = model.predict(X)
        dummy_features_train = np.zeros((train_reg_pred.shape[0], num_extra_features))
        train_pred_full = np.hstack((train_reg_pred, dummy_features_train))
        train_pred_inv = scaler.inverse_transform(train_pred_full)[:, :4]
        
        train_predict_df = pd.DataFrame(train_pred_inv, 
                                      columns=['open_pred', 'high_pred', 'low_pred', 'close_pred'],
                                      index=data.index[n_steps:])
        
        # Сглаживание прогнозов
        train_predict_df['close_pred'] = train_predict_df['close_pred'].ewm(span=5).mean()
        
        # Расчет точности направления движения
        direction_accuracy, correct, total = calculate_direction_accuracy(data, train_predict_df, n_steps)
        print(f"\nТочность предсказания направления: {direction_accuracy:.2f}%")
        print(f"({correct} из {total} правильных предсказаний)")
        
        # Прогнозирование следующей свечи
        last_sequence = X[-1].reshape(1, n_steps, num_features)
        next_reg_pred, next_class_pred = model.predict(last_sequence)
        direction_prob = next_class_pred[0][0]
        
        # Денормализация прогноза
        dummy_next = np.zeros((1, num_extra_features))
        next_pred_full = np.hstack((next_reg_pred, dummy_next))
        next_candle_values = scaler.inverse_transform(next_pred_full)[0, :4]
        
        next_candle = {
            'timestamp': data.index[-1] + pd.Timedelta(hours=1),
            'open': next_candle_values[0],
            'high': next_candle_values[1],
            'low': next_candle_values[2],
            'close': next_candle_values[3]
        }
        
        print("\nПрогноз на следующий период:")
        print(f"Открытие: {next_candle['open']:.2f}")
        print(f"Максимум: {next_candle['high']:.2f}")
        print(f"Минимум: {next_candle['low']:.2f}")
        print(f"Закрытие: {next_candle['close']:.2f}")
        print(f"Вероятность роста: {direction_prob:.2%}")
        
        # Определение точки входа
        entry_signal, confidence = get_entry_signal(data, next_candle, direction_prob)
        print(f"\nТочка входа: {entry_signal} (уверенность: {confidence:.2f})")
        
        # Формируем данные о точности направления
        direction_info = {
            'accuracy': direction_accuracy,
            'correct': correct,
            'total': total
        }
        
        # Визуализация с прогнозами, метриками и точкой входа
        plot_candlestick_with_predictions(data, train_predict_df[['close_pred']], next_candle, 
                                        metrics, entry_signal, direction_info)
        
        print("\n" + "="*50)
        print("АКТУАЛЬНЫЕ ДАННЫЕ ПОСЛЕДНЕЙ СВЕЧИ:")
        last_candle = data.iloc[-1]
        msk_time = last_candle.name.tz_convert('Europe/Moscow')
        print(f"Время (МСК): {msk_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Open: {last_candle['open']:.2f}")
        print(f"High: {last_candle['high']:.2f}")
        print(f"Low: {last_candle['low']:.2f}")
        print(f"Close: {last_candle['close']:.2f}")
        print(f"Volume: {last_candle['volume']:.2f}")
        
        print("\nVSA-СИГНАЛЫ ПОСЛЕДНЕЙ СВЕЧИ:")
        print(f"Spring: {'Да' if last_candle['spring'] else 'Нет'}")
        print(f"Upthrust: {'Да' if last_candle['upthrust'] else 'Нет'}")
        print(f"Бычье поглощение: {'Да' if last_candle['bullish_engulfing'] else 'Нет'}")
        print(f"Медвежье поглощение: {'Да' if last_candle['bearish_engulfing'] else 'Нет'}")
        print(f"Кульминация покупок: {'Да' if last_candle['buying_climax'] else 'Нет'}")
        print(f"Кульминация продаж: {'Да' if last_candle['selling_climax'] else 'Нет'}")
        print(f"Накопление: {'Да' if last_candle['accumulation'] else 'Нет'}")
        print(f"Распределение: {'Да' if last_candle['distribution'] else 'Нет'}")
        print(f"Hammer: {'Да' if last_candle['hammer'] else 'Нет'}")
        print(f"Shooting Star: {'Да' if last_candle['shooting_star'] else 'Нет'}")
        
        print("\nНОВЫЕ ПРИЗНАКИ ПОСЛЕДНЕЙ СВЕЧИ:")
        print(f"Ошибка прогноза: {last_candle['forecast_error']:.2f}")
        print(f"Направление ошибки: {last_candle['forecast_direction_error']}")
        print(f"Час: {last_candle['hour']}")
        print(f"День недели: {last_candle['day_of_week']}")
        print(f"Незакрытая свеча: {'Да' if last_candle['is_open'] == 1 else 'Нет'}")
        
    except Exception as e:
        print(f"Произошла ошибка: {e}")