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
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, Multiply, 
                                    Activation, Reshape, Lambda, concatenate,
                                    BatchNormalization)
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import talib  # Добавлено для технических индикаторов

# Настройки
symbol = 'BTC/USDT'
timeframe = '1h'
limit_days = 365  
exchange_id = 'binance'
EPOCH = 50


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


def attention_block(inputs):
    # Исправленный механизм внимания
    attention = Dense(1, activation='sigmoid')(inputs)  # Замена softmax на sigmoid
    attention = Multiply()([inputs, attention])
    attention = Lambda(lambda x: K.sum(x, axis=1))(attention)
    return attention

def create_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # LSTM слои с регуляризацией
    lstm1 = LSTM(256, return_sequences=True, kernel_regularizer=l2(0.001))(inputs)
    lstm1 = BatchNormalization()(lstm1)
    lstm1 = Dropout(0.5)(lstm1)
    
    lstm2 = LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001))(lstm1)
    lstm2 = BatchNormalization()(lstm2)
    lstm2 = Dropout(0.5)(lstm2)
    
    lstm3 = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))(lstm2)
    lstm3 = BatchNormalization()(lstm3)
    lstm3 = Dropout(0.5)(lstm3)
    
    # Механизм внимания
    attention = attention_block(lstm3)
    
    # Дополнительные слои
    dense1 = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(attention)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.4)(dense1)
    
    dense2 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.4)(dense2)
    
    outputs = Dense(4)(dense2)  # 4 выхода: open, high, low, close
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=[
            MeanAbsoluteError(name='mae'),
        ]
    )
    return model



def calculate_vsa_features(df):
    """Расширенные признаки VSA с разделением бычьего/медвежьего объема"""
    # Базовые параметры
    df['spread'] = df['high'] - df['low']
    df['avg_spread_5'] = df['spread'].rolling(5).mean()
    df['relative_spread'] = df['spread'] / df['avg_spread_5']
    
    # Базовые признаки объема
    df['volume_ma_10'] = df['volume'].rolling(10).mean()
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    
    # --- Бычий/медвежий объем ---
    # Бычий объем: растет при росте цены ИЛИ снижается при падении
    df['bull_volume'] = np.where(
        ((df['close'] > df['open']) & (df['volume'] > df['volume'].shift(1))) |
        ((df['close'] < df['open']) & (df['volume'] < df['volume'].shift(1))),
        df['volume'], 
        0
    )
    
    # Медвежий объем: растет при падении цены ИЛИ снижается при росте
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
    # Сила/слабость с учетом объема
    df['is_weak_bar'] = ((df['close'] < df['open']) & 
                        (df['volume'] > df['volume_ma_10'] * 1.2)).astype(int)
    
    df['is_strong_bar'] = ((df['close'] > df['open']) & 
                          (df['volume'] > df['volume_ma_10'] * 1.5)).astype(int)
    
    # Импульс тренда (учитывает направление и объем)
    df['volume_impulse'] = df['volume'] * (df['close'] - df['open']) / df['spread']
    
    # Кластеры спроса/предложения с фильтрацией
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
    
    # Сигналы продолжения тренда с подтверждением объемом
    df['uptrend_continuation'] = ((df['close'] > df['open']) & 
                                (df['close'] > df['close'].shift(1)) & 
                                (df['bull_volume_ratio'] > 1.2)).astype(int)
    
    df['downtrend_continuation'] = ((df['close'] < df['open']) & 
                                  (df['close'] < df['close'].shift(1)) & 
                                  (df['bear_volume_ratio'] > 1.2)).astype(int)
    
    # ===== ДОБАВЛЕННЫЕ СИГНАЛЫ VSA =====
    # 1. Тест на спрос (Spring) - бычий сигнал
    df['spring'] = ((df['low'] < df['low'].shift(1)) & 
                   (df['close'] > (df['open'].shift(1) + df['close'].shift(1)) / 2) & 
                   (df['volume'] > df['volume_ma_10'] * 1.3)).astype(int)
    
    # 2. Тест на предложение (Upthrust) - медвежий сигнал
    df['upthrust'] = ((df['high'] > df['high'].shift(1)) & 
                     (df['close'] < (df['open'].shift(1) + df['close'].shift(1)) / 2) & 
                     (df['volume'] > df['volume_ma_10'] * 1.3)).astype(int)
    
    # 3. Поглощение (Engulfing)
    df['bullish_engulfing'] = ((df['close'] > df['open']) & 
                              (df['open'] < df['close'].shift(1)) & 
                              (df['close'] > df['open'].shift(1)) & 
                              (df['volume'] > df['volume_ma_10'] * 1.4)).astype(int)
    
    df['bearish_engulfing'] = ((df['close'] < df['open']) & 
                              (df['open'] > df['close'].shift(1)) & 
                              (df['close'] < df['open'].shift(1)) & 
                              (df['volume'] > df['volume_ma_10'] * 1.4)).astype(int)
    
    # 4. Кульминация покупок/продаж
    df['buying_climax'] = ((df['high'] > df['high'].shift(1)) & 
                          (df['low'] < df['low'].shift(1)) & 
                          (df['close'] < df['close'].shift(1)) & 
                          (df['volume'] > df['volume_ma_10'] * 2.0)).astype(int)
    
    df['selling_climax'] = ((df['high'] > df['high'].shift(1)) & 
                           (df['low'] < df['low'].shift(1)) & 
                           (df['close'] > df['close'].shift(1)) & 
                           (df['volume'] > df['volume_ma_10'] * 2.0)).astype(int)
    
    # 5. Баланс рынка
    df['market_balance'] = np.where(
        df['bull_volume_ratio'] > df['bear_volume_ratio'],
        df['bull_volume_ratio'] - df['bear_volume_ratio'],
        df['bear_volume_ratio'] - df['bull_volume_ratio']
    )
    
    # 6. Сигналы накопления/распределения
    df['accumulation'] = ((df['close'] > df['open']) & 
                         (df['close'] > df['close'].shift(1)) & 
                         (df['volume'] > df['volume_ma_10'] * 1.2) & 
                         (df['bull_volume_ratio'] > 1.5)).astype(int)
    
    df['distribution'] = ((df['close'] < df['open']) & 
                         (df['close'] < df['close'].shift(1)) & 
                         (df['volume'] > df['volume_ma_10'] * 1.2) & 
                         (df['bear_volume_ratio'] > 1.5)).astype(int)
    
    # ===== НОВЫЕ ПРИЗНАКИ ДЛЯ УЛУЧШЕНИЯ ТОЧНОСТИ =====
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
    
    # ===== ДОБАВЛЕННЫЕ ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ (TA-LIB) =====
    # RSI индикаторы
    df['rsi_6'] = talib.RSI(df['close'], timeperiod=6)
    df['rsi_12'] = talib.RSI(df['close'], timeperiod=12)
    df['rsi_24'] = talib.RSI(df['close'], timeperiod=24)
    
    # MACD
    macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist
    
    # KDJ (Stochastic Oscillator)
    slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'], 
                              fastk_period=9, slowk_period=3, slowk_matype=0, 
                              slowd_period=3, slowd_matype=0)
    df['kdj_k'] = slowk
    df['kdj_d'] = slowd
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
    
    # Williams %R
    df['wr_6'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=6)
    
    # Скользящие средние
    df['ma_5'] = talib.SMA(df['close'], timeperiod=5)
    df['ma_10'] = talib.SMA(df['close'], timeperiod=10)
    df['ma_30'] = talib.SMA(df['close'], timeperiod=30)
    df['ma_50'] = talib.SMA(df['close'], timeperiod=50)
    df['ma_100'] = talib.SMA(df['close'], timeperiod=100)
    
    # Экспоненциальные скользящие
    df['ema_7'] = talib.EMA(df['close'], timeperiod=7)
    df['ema_30'] = talib.EMA(df['close'], timeperiod=30)
    
    # Полосы Боллинджера
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower
    df['bb_width'] = (upper - lower) / middle  # Относительная ширина полос
    
    # Дополнительные индикаторы
    df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
    df['obv'] = talib.OBV(df['close'], df['volume'])
    
    # ===== УРОВНИ ПОДДЕРЖКИ И СОПРОТИВЛЕНИЯ =====
    window = 20  # Размер окна для определения уровней
    df['support'] = df['low'].rolling(window=window).min()
    df['resistance'] = df['high'].rolling(window=window).max()
    
    # Расстояние до уровней
    df['dist_to_support'] = df['close'] - df['support']
    df['dist_to_resistance'] = df['resistance'] - df['close']
    
    # Относительное положение между уровнями
    df['support_resistance_ratio'] = (df['close'] - df['support']) / (df['resistance'] - df['support'] + 1e-10)
    
    # ===== ОПРЕДЕЛЕНИЕ ТРЕНДА =====
    # Используем EMA для определения тренда
    df['ema_trend'] = np.where(df['ema_7'] > df['ema_30'], 1, -1)
    
    # ADX для силы тренда
    df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['trend_strength'] = np.where(df['adx'] > 25, 1, 0)  # Сильный тренд при ADX > 25
    
    # Комбинированный тренд: 1 - восходящий, -1 - нисходящий, 0 - боковой
    df['trend_direction'] = 0
    df.loc[(df['ema_7'] > df['ema_30']) & (df['close'] > df['ma_50']), 'trend_direction'] = 1
    df.loc[(df['ema_7'] < df['ema_30']) & (df['close'] < df['ma_50']), 'trend_direction'] = -1
    
    # Фильтр боковика (низкая волатильность)
    df['volatility_ratio'] = df['volatility_10'] / df['volatility_10'].rolling(50).mean()
    df.loc[(df['trend_direction'] != 0) & (df['volatility_ratio'] < 0.5), 'trend_direction'] = 0
    
    # Заполнение пропусков
    df.fillna(0, inplace=True)
    return df


def prepare_data(df, n_steps=50):  # Увеличено окно с 30 до 50
    # Добавляем VSA признаки
    df = calculate_vsa_features(df)
    
    # Определяем признаки для масштабирования (добавлены новые индикаторы)
    cols_to_scale = [
        'open', 'high', 'low', 'close', 'volume',
        'bull_volume', 'bear_volume', 'bull_volume_ratio', 'bear_volume_ratio',
        'relative_spread', 'is_weak_bar', 'is_strong_bar',
        'volume_impulse', 'demand_zone', 'supply_zone',
        'demand_deficit', 'uptrend_continuation', 'downtrend_continuation',
        'spring', 'upthrust', 'bullish_engulfing', 'bearish_engulfing',
        'buying_climax', 'selling_climax', 'market_balance',
        'accumulation', 'distribution',
        # Новые признаки
        'volume_oscillator', 'price_volume_corr', 
        'vwap', 'vwap_deviation', 
        'momentum_5', 'volatility_10',
        'hammer', 'shooting_star',
        # Добавленные индикаторы
        'rsi_6', 'rsi_12', 'rsi_24',
        'macd', 'macd_signal', 'macd_hist',
        'kdj_k', 'kdj_d', 'kdj_j',
        'wr_6', 'ma_5', 'ma_10', 'ma_30', 'ma_50', 'ma_100',
        'ema_7', 'ema_30', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
        'cci', 'obv', 'adx',
        # Уровни и тренд
        'support', 'resistance', 'dist_to_support', 'dist_to_resistance',
        'support_resistance_ratio', 'trend_direction', 'trend_strength'
    ]
    
    # Нормализуем данные
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[cols_to_scale])
    
    X, y = [], []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i-n_steps:i])
        y.append(scaled_data[i, :4])  # open, high, low, close
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, scaler, len(cols_to_scale)


def get_entry_signal(data, next_candle):
    last_candle = data.iloc[-1]
    current_close = last_candle['close']
    predicted_change = next_candle['close'] - current_close
    
    # Базовый сигнал по прогнозу
    base_signal = 'BUY' if predicted_change > 0 else 'SELL'
    
    # Анализ VSA сигналов и индикаторов
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
    
    # Анализ индикаторов
    if last_candle['rsi_6'] < 30:
        bull_signals += 0.7
        vsa_signals.append("RSI(6) в зоне перепроданности")
    elif last_candle['rsi_6'] > 70:
        bear_signals += 0.7
        vsa_signals.append("RSI(6) в зоне перекупленности")
        
    if last_candle['macd'] > last_candle['macd_signal']:
        bull_signals += 0.6
        vsa_signals.append("MACD бычий кроссовер")
    elif last_candle['macd'] < last_candle['macd_signal']:
        bear_signals += 0.6
        vsa_signals.append("MACD медвежий кроссовер")
        
    if last_candle['kdj_j'] < 20:
        bull_signals += 0.5
        vsa_signals.append("KDJ в зоне перепроданности")
    elif last_candle['kdj_j'] > 80:
        bear_signals += 0.5
        vsa_signals.append("KDJ в зоне перекупленности")
    
    # Анализ тренда
    if last_candle['trend_direction'] == 1:
        bull_signals += 1.0
        vsa_signals.append("Восходящий тренд")
    elif last_candle['trend_direction'] == -1:
        bear_signals += 1.0
        vsa_signals.append("Нисходящий тренд")
    
    # Уровни поддержки/сопротивления
    support_dist = last_candle['dist_to_support']
    resistance_dist = last_candle['dist_to_resistance']
    
    if support_dist < 0.005 * current_close:  # Близко к поддержке
        bull_signals += 0.8
        vsa_signals.append("Цена около уровня поддержки")
    if resistance_dist < 0.005 * current_close:  # Близко к сопротивлению
        bear_signals += 0.8
        vsa_signals.append("Цена около уровня сопротивления")
    
    # Корректировка сигнала
    signal_strength = bull_signals - bear_signals
    confidence = abs(signal_strength)
    
    if signal_strength > 1.0:  # Сильные бычьи сигналы
        final_signal = 'BUY'
    elif signal_strength < -1.0:  # Сильные медвежьи сигналы
        final_signal = 'SELL'
    elif signal_strength > 0.5:  # Умеренные бычьи сигналы
        final_signal = 'BUY' if base_signal == 'BUY' else 'HOLD'
    elif signal_strength < -0.5:  # Умеренные медвежьи сигналы
        final_signal = 'SELL' if base_signal == 'SELL' else 'HOLD'
    else:  # Нейтрально или слабые сигналы
        final_signal = base_signal
    
    # Форматирование вывода
    vsa_info = " | ".join(vsa_signals) if vsa_signals else "Нет значимых сигналов"
    print(f"\nVSA АНАЛИЗ: {vsa_info}")
    print(f"Сила бычьих сигналов: {bull_signals:.2f}")
    print(f"Сила медвежьих сигналов: {bear_signals:.2f}")
    print(f"Итоговый сигнал: {final_signal} (базовый: {base_signal})")
    
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
    plt.figure(figsize=(12, 6))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Evolution')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('MAE Evolution')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
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
        n_steps = 50  # Увеличено окно
        X, y, scaler, num_features = prepare_data(data, n_steps)
        num_extra_features = num_features - 4  # 4 - это OHLC
        
        # Разделение данных
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Создание модели
        model = create_model((X_train.shape[1], X_train.shape[2]))
        model.summary()
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=25,  # Увеличенное терпение
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # Менее агрессивное уменьшение
            patience=10,
            min_lr=0.000001,
            verbose=1
        )
        
        # Обучение модели
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCH,  # Увеличенное количество эпох
            batch_size=64,  # Увеличенный размер батча
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        # Визуализация истории обучения
        plot_training_history(history)
        
        # Прогнозирование на валидационных данных
        val_predict = model.predict(X_val)
        
        # Денормализация прогнозов
        dummy_features = np.zeros((val_predict.shape[0], num_extra_features))
        val_predict_full = np.hstack((val_predict, dummy_features))
        val_predict_inv = scaler.inverse_transform(val_predict_full)[:, :4]
        
        # Денормализация истинных значений
        dummy_features_y = np.zeros((y_val.shape[0], num_extra_features))
        y_val_full = np.hstack((y_val, dummy_features_y))
        y_val_inv = scaler.inverse_transform(y_val_full)[:, :4]
        
        # Расчет метрик
        metrics = calculate_metrics(y_val_inv, val_predict_inv)
        print("\nМетрики качества на валидационных данных:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        
        # Прогнозирование на всех данных для визуализации
        train_predict = model.predict(X)
        dummy_features_train = np.zeros((train_predict.shape[0], num_extra_features))
        train_predict_full = np.hstack((train_predict, dummy_features_train))
        train_predict_inv = scaler.inverse_transform(train_predict_full)[:, :4]
        
        train_predict_df = pd.DataFrame(train_predict_inv, 
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
        next_pred = model.predict(last_sequence)
        
        dummy_features_next = np.zeros((1, num_extra_features))
        next_pred_full = np.hstack((next_pred, dummy_features_next))
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
        
        # Определение точки входа
        current_close = data['close'].iloc[-1]
        predicted_change = next_candle['close'] - current_close
        entry_signal, confidence = get_entry_signal(data, next_candle)
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
        # Вывод последних данных свечи
        last_candle = data.iloc[-1]
        msk_time = last_candle.name.tz_localize('UTC').tz_convert('Europe/Moscow')

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
        
        # Вывод индикаторов и уровней
        print("\nТЕХНИЧЕСКИЕ ИНДИКАТОРЫ:")
        print(f"RSI(6): {last_candle['rsi_6']:.2f}")
        print(f"MACD: {last_candle['macd']:.4f}, Signal: {last_candle['macd_signal']:.4f}")
        print(f"KDJ: K={last_candle['kdj_k']:.2f}, D={last_candle['kdj_d']:.2f}, J={last_candle['kdj_j']:.2f}")
        print(f"Williams %R: {last_candle['wr_6']:.2f}")
        print(f"ADX: {last_candle['adx']:.2f}")
        print(f"Поддержка: {last_candle['support']:.2f}, Сопротивление: {last_candle['resistance']:.2f}")
        print(f"Дистанция до поддержки: {last_candle['dist_to_support']:.2f}")
        print(f"Дистанция до сопротивления: {last_candle['dist_to_resistance']:.2f}")
        print(f"Тренд: {'Восходящий' if last_candle['trend_direction'] == 1 else 'Нисходящий' if last_candle['trend_direction'] == -1 else 'Боковой'}")
        
        # Интерпретация с учетом новых данных
        print("\nИНТЕРПРЕТАЦИЯ:")
        if last_candle['trend_direction'] == 1:
            print("- Явный восходящий тренд")
        elif last_candle['trend_direction'] == -1:
            print("- Явный нисходящий тренд")
        else:
            print("- Рынок в боковом движении (флэт)")
            
        if last_candle['dist_to_support'] < 0.01 * current_close:
            print("- Цена вблизи важного уровня поддержки")
        if last_candle['dist_to_resistance'] < 0.01 * current_close:
            print("- Цена вблизи важного уровня сопротивления")
            
        if last_candle['rsi_6'] < 30:
            print("- Сильная перепроданность по RSI(6)")
        elif last_candle['rsi_6'] > 70:
            print("- Сильная перекупленность по RSI(6)")
            
        if last_candle['macd'] > last_candle['macd_signal']:
            print("- Бычий сигнал MACD")
        elif last_candle['macd'] < last_candle['macd_signal']:
            print("- Медвежий сигнал MACD")
            
        print("="*50 + "\n")

    except Exception as e:
        print(f"Произошла ошибка: {e}")