import ccxt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Bidirectional, LSTM, Dense, Dropout, 
                                    Multiply, Activation, Lambda, BatchNormalization, 
                                    Masking)
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.dates import DateFormatter
import datetime
from scipy.stats import linregress

# Настройки
symbol = 'BTC/USDT'
timeframe = '4h'
limit_days = 90
exchange_id = 'binance'
EPOCH = 100
ORDER_BOOK_DEPTH = 5

def initialize_exchange():
    exchange = getattr(ccxt, exchange_id)({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
            'adjustForTimeDifference': True,
        }
    })
    exchange.load_markets()
    return exchange

def fetch_ohlcv_data(exchange):
    since = exchange.parse8601((datetime.datetime.now() - datetime.timedelta(days=limit_days)).strftime('%Y-%m-%d %H:%M:%S'))
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    
    # Удаляем незакрытую свечу
    current_time = pd.Timestamp.now(tz='UTC')
    last_candle_time = df.index[-1]
    if (current_time - last_candle_time) < pd.Timedelta(timeframe):
        df = df.iloc[:-1]
    
    return df

def fetch_order_book(exchange):
    try:
        order_book = exchange.fetch_order_book(symbol, limit=ORDER_BOOK_DEPTH)
        return {
            'bids': order_book['bids'],
            'asks': order_book['asks'],
            'bid_ask_spread': order_book['asks'][0][0] - order_book['bids'][0][0],
            'order_book_imbalance': (sum(bid[1] for bid in order_book['bids']) - 
                                   sum(ask[1] for ask in order_book['asks'])) / 
                                  (sum(bid[1] for bid in order_book['bids']) + 
                                   sum(ask[1] for ask in order_book['asks']))
        }
    except Exception as e:
        print(f"Ошибка получения стакана: {e}")
        return None

def calculate_price_action_features(df):
    """Расширенные признаки Price Action"""
    # 1. Свечные паттерны
    df['is_hammer'] = ((df['close'] > df['open']) & 
                      ((df['close'] - df['low']) > 2 * (df['high'] - df['close'])) & 
                      ((df['close'] - df['low']) > 1.5 * (df['open'] - df['low']))).astype(int)
    
    df['is_shooting_star'] = ((df['close'] < df['open']) & 
                            ((df['high'] - df['open']) > 2 * (df['open'] - df['close'])) & 
                            ((df['high'] - df['open']) > 1.5 * (df['high'] - df['low']))).astype(int)
    
    df['is_engulfing_bull'] = ((df['close'] > df['open']) & 
                              (df['close'].shift(1) < df['open'].shift(1)) & 
                              (df['close'] > df['open'].shift(1)) & 
                              (df['open'] < df['close'].shift(1))).astype(int)
    
    df['is_engulfing_bear'] = ((df['close'] < df['open']) & 
                              (df['close'].shift(1) > df['open'].shift(1)) & 
                              (df['close'] < df['open'].shift(1)) & 
                              (df['open'] > df['close'].shift(1))).astype(int)
    
    df['is_pinbar_bull'] = ((df['close'] > df['open']) & 
                           ((df['low'] - df['open']) > 3 * (df['close'] - df['open']))).astype(int)
    
    df['is_pinbar_bear'] = ((df['close'] < df['open']) & 
                           ((df['high'] - df['open']) > 3 * (df['open'] - df['close']))).astype(int)
    
    # 2. Кластерные анализ
    df['body_size'] = abs(df['close'] - df['open'])
    df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
    df['total_range'] = df['high'] - df['low']
    
    # 3. Соотношение теней
    df['shadow_ratio'] = np.where(df['upper_shadow'] + df['lower_shadow'] > 0,
                                (df['upper_shadow'] - df['lower_shadow']) / 
                                (df['upper_shadow'] + df['lower_shadow']), 0)
    
    # 4. Скользящие уровни
    df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
    df['r1'] = 2 * df['pivot'] - df['low']
    df['s1'] = 2 * df['pivot'] - df['high']
    
    # 5. Трендовые индикаторы
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    df['ema_cross'] = np.where(df['ema_20'] > df['ema_50'], 1, -1)
    df['price_above_ema'] = np.where(df['close'] > df['ema_200'], 1, -1)
    
    # 6. Уровни поддержки/сопротивления
    df['support'] = df['low'].rolling(20).min()
    df['resistance'] = df['high'].rolling(20).max()
    df['at_support'] = (df['low'] <= df['support'] * 1.005).astype(int)
    df['at_resistance'] = (df['high'] >= df['resistance'] * 0.995).astype(int)
    
    # 7. Объемный профиль
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    df['price_vwap_ratio'] = df['close'] / df['vwap']
    
    # 8. Импульс и ускорение
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_10'] = df['close'].pct_change(10)
    df['acceleration'] = df['momentum_5'] - df['momentum_10']
    
    # 9. Регрессионный наклон
    def get_slope(series):
        x = np.arange(len(series))
        slope, _, _, _, _ = linregress(x, series)
        return slope
    
    df['regression_slope_10'] = df['close'].rolling(10).apply(get_slope, raw=True)
    df['regression_slope_20'] = df['close'].rolling(20).apply(get_slope, raw=True)
    
    # 10. Фракталы
    df['fractal_top'] = ((df['high'] > df['high'].shift(1)) & 
                        (df['high'] > df['high'].shift(2)) & 
                        (df['high'] > df['high'].shift(-1)) & 
                        (df['high'] > df['high'].shift(-2))).astype(int)
    
    df['fractal_bottom'] = ((df['low'] < df['low'].shift(1)) & 
                           (df['low'] < df['low'].shift(2)) & 
                           (df['low'] < df['low'].shift(-1)) & 
                           (df['low'] < df['low'].shift(-2))).astype(int)
    
    # 11. Свечные комбинации
    df['three_white_soldiers'] = ((df['close'] > df['open']) & 
                                 (df['close'].shift(1) > df['open'].shift(1)) & 
                                 (df['close'].shift(2) > df['open'].shift(2)) & 
                                 (df['close'] > df['close'].shift(1)) & 
                                 (df['close'].shift(1) > df['close'].shift(2))).astype(int)
    
    df['three_black_crows'] = ((df['close'] < df['open']) & 
                              (df['close'].shift(1) < df['open'].shift(1)) & 
                              (df['close'].shift(2) < df['open'].shift(2)) & 
                              (df['close'] < df['close'].shift(1)) & 
                              (df['close'].shift(1) < df['close'].shift(2))).astype(int)
    
    # Заполняем пропущенные значения
    df.fillna(0, inplace=True)
    
    return df

def calculate_vsa_features(df):
    """Расширенные признаки VSA"""
    # Базовые параметры
    df['spread'] = df['high'] - df['low']
    df['avg_spread_5'] = df['spread'].rolling(5).mean().fillna(0)
    df['relative_spread'] = df['spread'] / df['avg_spread_5'].replace(0, 1)
    
    # Базовые признаки объема
    df['volume_ma_10'] = df['volume'].rolling(10).mean().fillna(0)
    df['volume_ma_20'] = df['volume'].rolling(20).mean().fillna(0)
    
    # Бычий/медвежий объем
    df['bull_volume'] = np.where(
        (df['close'] > df['open']) & (df['volume'] > df['volume_ma_10']),
        df['volume'] * (df['close'] - df['open']) / df['spread'].replace(0, 1),
        0
    )
    
    df['bear_volume'] = np.where(
        (df['close'] < df['open']) & (df['volume'] > df['volume_ma_10']),
        df['volume'] * (df['open'] - df['close']) / df['spread'].replace(0, 1),
        0
    )
    
    # Отношение к среднему объему
    df['bull_volume_ratio'] = df['bull_volume'] / df['volume_ma_10'].replace(0, 1)
    df['bear_volume_ratio'] = df['bear_volume'] / df['volume_ma_10'].replace(0, 1)
    
    # Сигналы VSA
    df['is_weak_bar'] = ((df['close'] < df['open']) & 
                        (df['volume'] > df['volume_ma_10'] * 1.2)).astype(int)
    
    df['is_strong_bar'] = ((df['close'] > df['open']) & 
                          (df['volume'] > df['volume_ma_10'] * 1.5)).astype(int)
    
    # Кластеры спроса/предложения
    df['demand_zone'] = ((df['close'] > df['open']) & 
                        (df['bull_volume_ratio'] > 1.5)).astype(int)
    
    df['supply_zone'] = ((df['close'] < df['open']) & 
                        (df['bear_volume_ratio'] > 1.5)).astype(int)
    
    # Расширенные сигналы VSA
    df['absorption'] = np.where(
        (df['close'] > df['open']) & (df['volume'] > df['volume_ma_10'] * 2) & 
        (df['close'] == df['high']),
        1,  # Бычья абсорбция
        np.where(
            (df['close'] < df['open']) & (df['volume'] > df['volume_ma_10'] * 2) & 
            (df['close'] == df['low']),
            -1,  # Медвежья абсорбция
            0
        )
    )
    
    # Новые VSA сигналы
    df['strength_test'] = np.where(
        (df['close'] > df['open']) & (df['volume'] < df['volume_ma_10'] * 0.7) & 
        (df['spread'] > df['avg_spread_5']),
        1,  # Бычий тест на прочность
        np.where(
            (df['close'] < df['open']) & (df['volume'] < df['volume_ma_10'] * 0.7) & 
            (df['spread'] > df['avg_spread_5']),
            -1,  # Медвежий тест на прочность
            0
        )
    )
    
    df['stopping_volume'] = np.where(
        (df['high'].shift(1) < df['high']) & (df['low'].shift(1) > df['low']) & 
        (df['volume'] > df['volume_ma_10'] * 1.8) & 
        (df['close'] < (df['high'] + df['low']) / 2),
        -1,  # Медвежий stopping volume
        np.where(
            (df['high'].shift(1) > df['high']) & (df['low'].shift(1) < df['low']) & 
            (df['volume'] > df['volume_ma_10'] * 1.8) & 
            (df['close'] > (df['high'] + df['low']) / 2),
            1,  # Бычий stopping volume
            0
        )
    )
    
    df['no_demand'] = ((df['close'] > df['open']) & 
                      (df['volume'] < df['volume_ma_10'] * 0.5) & 
                      (df['spread'] < df['avg_spread_5'] * 0.7)).astype(int)
    
    df['no_supply'] = ((df['close'] < df['open']) & 
                      (df['volume'] < df['volume_ma_10'] * 0.5) & 
                      (df['spread'] < df['avg_spread_5'] * 0.7)).astype(int)
    
    df['wide_spread_up'] = ((df['close'] > df['open']) & 
                           (df['spread'] > df['avg_spread_5'] * 2) & 
                           (df['volume'] > df['volume_ma_10'])).astype(int)
    
    df['wide_spread_down'] = ((df['close'] < df['open']) & 
                             (df['spread'] > df['avg_spread_5'] * 2) & 
                             (df['volume'] > df['volume_ma_10'])).astype(int)
    
    df['hidden_strength'] = np.where(
        (df['close'] < df['open']) & (df['volume'] > df['volume_ma_10'] * 1.5) & 
        (df['close'] > (df['open'] + df['low']) / 2),
        1,  # Скрытая бычья сила
        np.where(
            (df['close'] > df['open']) & (df['volume'] > df['volume_ma_10'] * 1.5) & 
            (df['close'] < (df['open'] + df['high']) / 2),
            -1,  # Скрытая медвежья сила
            0
        )
    )
    
    df['ultra_high_volume'] = np.where(
        df['volume'] > df['volume_ma_10'] * 3,
        np.where(df['close'] > df['open'], 1, -1),
        0
    )
    
    # Баланс рынка
    df['volume_balance'] = df['bull_volume_ratio'] - df['bear_volume_ratio']
    
    # Заполняем данные стакана нулями (будут обновлены)
    for i in range(1, ORDER_BOOK_DEPTH + 1):
        df[f'bid_{i}_price'] = 0.0
        df[f'bid_{i}_amount'] = 0.0
        df[f'ask_{i}_price'] = 0.0
        df[f'ask_{i}_amount'] = 0.0
    
    df['bid_ask_spread'] = 0.0
    df['order_book_imbalance'] = 0.0
    
    # Заполняем пропущенные значения
    df.fillna(0, inplace=True)
    
    return df

def update_with_order_book(df, order_book):
    """Обновляем последнюю свечу данными из стакана"""
    if order_book is None:
        return df
    
    last_idx = df.index[-1]
    
    # Обновляем стакан
    for i in range(min(ORDER_BOOK_DEPTH, len(order_book['bids']))):
        df.at[last_idx, f'bid_{i+1}_price'] = order_book['bids'][i][0]
        df.at[last_idx, f'bid_{i+1}_amount'] = order_book['bids'][i][1]
    
    for i in range(min(ORDER_BOOK_DEPTH, len(order_book['asks']))):
        df.at[last_idx, f'ask_{i+1}_price'] = order_book['asks'][i][0]
        df.at[last_idx, f'ask_{i+1}_amount'] = order_book['asks'][i][1]
    
    # Обновляем метрики стакана
    df.at[last_idx, 'bid_ask_spread'] = order_book['bid_ask_spread']
    df.at[last_idx, 'order_book_imbalance'] = order_book['order_book_imbalance']
    
    return df

def create_radar_model(input_shape):
    inputs = Input(shape=input_shape)
    
    masked = Masking(mask_value=0)(inputs)  # Изменяем mask_value на 0
    
    # Улучшенная архитектура с инициализацией весов
    lstm1 = Bidirectional(LSTM(64, return_sequences=False,
                          kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                          kernel_initializer=GlorotUniform(seed=42)))(masked)
    lstm1 = BatchNormalization()(lstm1)
    lstm1 = Dropout(0.3)(lstm1)
    
    dense1 = Dense(32, activation='relu',
                 kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                 kernel_initializer=GlorotUniform(seed=42))(lstm1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.2)(dense1)
    
    # Выходной слой с сигмоидой
    output = Dense(5, activation='sigmoid', name='radar_output',
                 kernel_initializer=GlorotUniform(seed=42))(dense1)
    
    model = Model(inputs=inputs, outputs=output)
    
    # Настраиваем оптимизатор с learning rate
    optimizer = Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='mse',  # Изменяем на MSE для численной стабильности
        metrics=['mae', 'accuracy']
    )
    return model

def predict_future_movement(model, last_sequence, scaler, feature_cols, target_cols, future_steps=3):
    """Прогнозирование будущих значений показателей"""
    predictions = []
    current_sequence = last_sequence.copy()
    
    # Получаем индексы целевых переменных в feature_cols
    target_indices = [feature_cols.index(col) for col in target_cols]
    
    for _ in range(future_steps):
        # Прогнозируем следующий шаг
        pred = model.predict(current_sequence[np.newaxis, ...])[0]
        predictions.append(pred)
        
        # Создаем новую последовательность
        new_sequence = np.roll(current_sequence, -1, axis=0)
        
        # Обновляем только целевые переменные в последнем элементе последовательности
        new_sequence[-1, target_indices] = pred
        
        current_sequence = new_sequence
    
    return np.array(predictions)

def prepare_radar_data(df, n_steps=24):
    """Подготовка данных для обучения"""
    # Целевые переменные (5 показателей радара)
    target_cols = [
        'bull_volume_ratio',    # Сила покупателей
        'bear_volume_ratio',    # Сила продавцов
        'volume_balance',       # Баланс объема
        'relative_spread',      # Волатильность
        'order_book_imbalance'  # Баланс стакана
    ]
    
    # Все используемые признаки
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'bull_volume', 'bear_volume', 'bull_volume_ratio', 'bear_volume_ratio',
        'relative_spread', 'is_weak_bar', 'is_strong_bar',
        'demand_zone', 'supply_zone', 'absorption',
        'volume_balance', 'bid_ask_spread', 'order_book_imbalance',
        # Добавляем новые Price Action признаки
        'is_hammer', 'is_shooting_star', 'is_engulfing_bull', 'is_engulfing_bear',
        'is_pinbar_bull', 'is_pinbar_bear', 'body_size', 'upper_shadow', 'lower_shadow',
        'total_range', 'shadow_ratio', 'pivot', 'r1', 's1', 'ema_20', 'ema_50', 'ema_200',
        'ema_cross', 'price_above_ema', 'support', 'resistance', 'at_support', 'at_resistance',
        'vwap', 'price_vwap_ratio', 'momentum_5', 'momentum_10', 'acceleration',
        'regression_slope_10', 'regression_slope_20', 'fractal_top', 'fractal_bottom',
        'three_white_soldiers', 'three_black_crows'
    ]
    
    # Добавляем колонки стакана
    for i in range(1, ORDER_BOOK_DEPTH + 1):
        feature_cols.append(f'bid_{i}_price')
        feature_cols.append(f'bid_{i}_amount')
        feature_cols.append(f'ask_{i}_price')
        feature_cols.append(f'ask_{i}_amount')
    
    # Нормализуем данные
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])
    
    # Подготовка данных для обучения
    X, y = [], []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i-n_steps:i])
        y.append(scaled_data[i, [feature_cols.index(col) for col in target_cols]])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, scaler, feature_cols, target_cols

def plot_radar_chart(values, labels):
    """Визуализация радара"""
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.fill(angles, values, color='blue', alpha=0.25)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=10)
    plt.xticks(angles[:-1], labels)
    
    plt.title('VSA Радар рынка', size=16, y=1.1)
    plt.show()

def determine_market_trend(df, period=20):
    """Улучшенное определение текущего рыночного состояния"""
    # Берем последние N свечей для анализа
    recent_data = df.iloc[-period:]
    
    # Рассчитываем скользящие средние
    ma_short = recent_data['close'].rolling(5).mean().iloc[-1]
    ma_long = recent_data['close'].rolling(20).mean().iloc[-1]
    
    # Вычисляем среднее направление движения
    avg_direction = (recent_data['close'] - recent_data['open']).mean()
    price_range = recent_data['high'].max() - recent_data['low'].min()
    avg_price = recent_data['close'].mean()
    
    # Определяем волатильность
    volatility = price_range / avg_price if avg_price > 0 else 0
    
    # Определяем тренд по нескольким критериям
    trend_score = 0
    
    # Критерий 1: Разница между короткой и длинной MA
    if ma_short > ma_long * 1.005:  # Короткая MA выше длинной на 0.5%
        trend_score += 1
    elif ma_short < ma_long * 0.995:  # Короткая MA ниже длинной на 0.5%
        trend_score -= 1
    
    # Критерий 2: Среднее направление свечей
    if avg_direction > 0.002 * avg_price:  # Преобладают бычьи свечи
        trend_score += 1
    elif avg_direction < -0.002 * avg_price:  # Преобладают медвежьи свечи
        trend_score -= 1
    
    # Критерий 3: Максимальный/минимальный уровень
    if recent_data['close'].iloc[-1] > recent_data['high'].rolling(5).mean().iloc[-1]:
        trend_score += 1
    elif recent_data['close'].iloc[-1] < recent_data['low'].rolling(5).mean().iloc[-1]:
        trend_score -= 1
    
    # Определяем тренд на основе совокупности критериев
    if volatility < 0.01:  # Очень низкая волатильность
        return "Флет (боковик)"
    elif trend_score >= 2:  # Явный восходящий тренд
        return "Восходящий тренд"
    elif trend_score <= -2:  # Явный нисходящий тренд
        return "Нисходящий тренд"
    elif -1 <= trend_score <= 1:  # Неопределенное состояние
        return "Неопределенный тренд"
    else:  # Слабый тренд
        return "Слабый тренд (возможен флет)"

def generate_trading_signals(radar_values, last_candle, df):
    """Генерация торговых сигналов с учетом новых VSA паттернов"""
    bull_power = radar_values[0]
    bear_power = radar_values[1]
    volume_balance = radar_values[2]
    volatility = radar_values[3]
    order_imbalance = radar_values[4]
    
    signals = []
    confidence = 0
    recommendation = ""
    
    # Определяем текущее состояние рынка
    market_state = determine_market_trend(df)
    
    # Анализ VSA паттернов
    vsa_patterns = []
    
    # 1. Абсорбция
    if last_candle['absorption'] == 1:
        vsa_patterns.append("Бычья абсорбция: сильные покупатели поглотили продажи")
        confidence += 1.5
    elif last_candle['absorption'] == -1:
        vsa_patterns.append("Медвежья абсорбция: сильные продавцы поглотили покупки")
        confidence += 1.5
    
    # 2. Тест на прочность
    if last_candle['strength_test'] == 1:
        vsa_patterns.append("Бычий тест на прочность: движение вверх на низком объеме")
        confidence += 0.8
    elif last_candle['strength_test'] == -1:
        vsa_patterns.append("Медвежий тест на прочность: движение вниз на низком объеме")
        confidence += 0.8
    
    # 3. Stopping Volume
    if last_candle['stopping_volume'] == 1:
        vsa_patterns.append("Бычий stopping volume: продажи остановлены")
        confidence += 1.2
    elif last_candle['stopping_volume'] == -1:
        vsa_patterns.append("Медвежий stopping volume: покупки остановлены")
        confidence += 1.2
    
    # 4. Нет спроса/предложения
    if last_candle['no_demand']:
        vsa_patterns.append("Нет спроса: покупатели отсутствуют")
        confidence += 0.7
    if last_candle['no_supply']:
        vsa_patterns.append("Нет предложения: продавцы отсутствуют")
        confidence += 0.7
    
    # 5. Wide Spread
    if last_candle['wide_spread_up']:
        vsa_patterns.append("Широкое распространение вверх: сильное бычье давление")
        confidence += 1.0
    if last_candle['wide_spread_down']:
        vsa_patterns.append("Широкое распространение вниз: сильное медвежье давление")
        confidence += 1.0
    
    # 6. Hidden Strength
    if last_candle['hidden_strength'] == 1:
        vsa_patterns.append("Скрытая бычья сила: продажи не смогли опустить цену")
        confidence += 1.0
    elif last_candle['hidden_strength'] == -1:
        vsa_patterns.append("Скрытая медвежья сила: покупки не смогли поднять цену")
        confidence += 1.0
    
    # 7. Ultra High Volume
    if last_candle['ultra_high_volume'] == 1:
        vsa_patterns.append("Ультра-высокий объем на росте: возможен разворот вверх")
        confidence += 1.5
    elif last_candle['ultra_high_volume'] == -1:
        vsa_patterns.append("Ультра-высокий объем на падении: возможен разворот вниз")
        confidence += 1.5
    
    # Анализ Price Action паттернов
    price_action_patterns = []
    
    # 1. Свечные паттерны
    if last_candle['is_hammer']:
        price_action_patterns.append("Молот: потенциальный разворот вверх")
        confidence += 0.8
    if last_candle['is_shooting_star']:
        price_action_patterns.append("Падающая звезда: потенциальный разворот вниз")
        confidence += 0.8
    if last_candle['is_engulfing_bull']:
        price_action_patterns.append("Бычье поглощение: сильный сигнал на покупку")
        confidence += 1.2
    if last_candle['is_engulfing_bear']:
        price_action_patterns.append("Медвежье поглощение: сильный сигнал на продажу")
        confidence += 1.2
    if last_candle['is_pinbar_bull']:
        price_action_patterns.append("Бычий пин-бар: разворот вверх")
        confidence += 1.0
    if last_candle['is_pinbar_bear']:
        price_action_patterns.append("Медвежий пин-бар: разворот вниз")
        confidence += 1.0
    
    # 2. Уровни поддержки/сопротивления
    if last_candle['at_support']:
        price_action_patterns.append("Цена у поддержки: возможен отскок вверх")
        confidence += 0.7
    if last_candle['at_resistance']:
        price_action_patterns.append("Цена у сопротивления: возможен отскок вниз")
        confidence += 0.7
    
    # 3. Трендовые индикаторы
    if last_candle['ema_cross'] == 1:
        price_action_patterns.append("Бычий кросс EMA 20/50")
        confidence += 0.5
    elif last_candle['ema_cross'] == -1:
        price_action_patterns.append("Медвежий кросс EMA 20/50")
        confidence += 0.5
    
    if last_candle['price_above_ema'] == 1:
        price_action_patterns.append("Цена выше EMA 200: долгосрочный бычий тренд")
        confidence += 0.6
    elif last_candle['price_above_ema'] == -1:
        price_action_patterns.append("Цена ниже EMA 200: долгосрочный медвежий тренд")
        confidence += 0.6
    
    # 4. Фракталы
    if last_candle['fractal_top']:
        price_action_patterns.append("Фрактальный верх: возможен разворот вниз")
        confidence += 0.5
    if last_candle['fractal_bottom']:
        price_action_patterns.append("Фрактальный низ: возможен разворот вверх")
        confidence += 0.5
    
    # 5. Свечные комбинации
    if last_candle['three_white_soldiers']:
        price_action_patterns.append("Три белых солдата: сильный бычий сигнал")
        confidence += 1.3
    if last_candle['three_black_crows']:
        price_action_patterns.append("Три черные вороны: сильный медвежий сигнал")
        confidence += 1.3
    
    # Генерация сигнала на основе радара, VSA и Price Action
    buy_conditions = (
        bull_power > 0.7 and 
        volume_balance > 0.3 and 
        order_imbalance > 0.2 and
        volatility > 0.4
    )
    
    sell_conditions = (
        bear_power > 0.7 and 
        volume_balance < -0.3 and 
        order_imbalance < -0.2 and
        volatility > 0.4
    )
    
    # Дополнительные условия из Price Action
    if last_candle['is_hammer'] or last_candle['is_engulfing_bull'] or last_candle['is_pinbar_bull']:
        buy_conditions = buy_conditions or (
            bull_power > 0.6 and 
            volume_balance > 0.2 and 
            order_imbalance > 0.1 and
            volatility > 0.3
        )
    
    if last_candle['is_shooting_star'] or last_candle['is_engulfing_bear'] or last_candle['is_pinbar_bear']:
        sell_conditions = sell_conditions or (
            bear_power > 0.6 and 
            volume_balance < -0.2 and 
            order_imbalance < -0.1 and
            volatility > 0.3
        )
    
    if buy_conditions:
        signal = "BUY"
        confidence += (bull_power - 0.7) * 2
        recommendation = f"Рекомендация: Рассмотреть покупку. Сильные покупатели контролируют рынок. Текущее состояние: {market_state}"
    elif sell_conditions:
        signal = "SELL"
        confidence += (bear_power - 0.7) * 2
        recommendation = f"Рекомендация: Рассмотреть продажу. Сильные продавцы контролируют рынок. Текущее состояние: {market_state}"
    else:
        signal = "HOLD"
        if confidence > 3:
            recommendation = f"Рекомендация: Ожидать подтверждения. Обнаружены противоречивые сигналы. Текущее состояние: {market_state}"
        else:
            recommendation = f"Рекомендация: Оставаться вне рынка. Недостаточно четких сигналов. Текущее состояние: {market_state}"
    
    # Дополнительные рекомендации на основе VSA, Price Action и состояния рынка
    if last_candle['no_demand'] and signal == "BUY":
        recommendation += " Внимание: обнаружен сигнал 'Нет спроса' - возможна слабость покупателей."
    
    if last_candle['no_supply'] and signal == "SELL":
        recommendation += " Внимание: обнаружен сигнал 'Нет предложения' - возможна слабость продавцов."
    
    if last_candle['ultra_high_volume'] != 0:
        recommendation += " Внимание: экстремальный объем - возможен разворот тренда."
    
    if last_candle['at_support'] and signal == "BUY":
        recommendation += " Дополнительное подтверждение: цена находится у уровня поддержки."
    
    if last_candle['at_resistance'] and signal == "SELL":
        recommendation += " Дополнительное подтверждение: цена находится у уровня сопротивления."
    
    # Адаптация рекомендации под текущее состояние рынка
    if "тренд" in market_state and signal in ["BUY", "SELL"]:
        if ("Восходящий" in market_state and signal == "BUY") or ("Нисходящий" in market_state and signal == "SELL"):
            recommendation += " Сигнал соответствует текущему тренду - высокая вероятность успеха."
        else:
            recommendation += " Сигнал против тренда - требуется дополнительная осторожность."
    
    confidence = min(max(confidence, 0), 10)
    
    return signal, confidence, vsa_patterns + price_action_patterns, recommendation, market_state

if __name__ == '__main__':
    try:
        # Инициализация подключения
        exchange = initialize_exchange()
        
        # Загрузка данных
        data = fetch_ohlcv_data(exchange)
        
        # Получение текущего стакана
        order_book = fetch_order_book(exchange)
        
        # Расчет VSA признаков
        data = calculate_vsa_features(data)
        
        # Расчет Price Action признаков
        data = calculate_price_action_features(data)
        
        # Обновление последней свечи данными стакана
        data = update_with_order_book(data, order_book)
        
        # Подготовка данных для обучения
        X, y, scaler, feature_cols, target_cols = prepare_radar_data(data)
        
        # Разделение данных
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Создание и обучение модели
        model = create_radar_model((X_train.shape[1], X_train.shape[2]))
        
        # Добавляем callback для сохранения лучшей модели
        checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        
        print("\nОбучение модели...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCH,
            batch_size=32,
            callbacks=[checkpoint],
            verbose=1
        )
        
        # Загружаем лучшую модель
        model.load_weights('best_model.h5')
        
        # Прогнозирование на последних данных
        last_sequence = X[-1].reshape(1, X.shape[1], X.shape[2])
        radar_values = model.predict(last_sequence)[0]
        
        # Прогнозируем будущие значения (3 шага вперед)
        future_predictions = predict_future_movement(model, X[-1], scaler, feature_cols, target_cols)
        
        # Генерация торгового сигнала
        last_candle = data.iloc[-1]
        signal, confidence, patterns, recommendation, market_state = generate_trading_signals(
            radar_values, last_candle, data
        )
        
        # Анализ будущих прогнозов
        future_trend = ""
        if len(future_predictions) > 0:
            avg_future_bull = future_predictions[:, 0].mean()
            avg_future_bear = future_predictions[:, 1].mean()
            
            if avg_future_bull - avg_future_bear > 0.2:
                future_trend = "Прогноз: вероятен рост в ближайшие периоды"
            elif avg_future_bear - avg_future_bull > 0.2:
                future_trend = "Прогноз: вероятно падение в ближайшие периоды"
            else:
                future_trend = "Прогноз: вероятна консолидация (флет)"
        
        # Визуализация результатов
        radar_labels = [
            'Сила покупателей', 
            'Сила продавцов', 
            'Баланс объема', 
            'Волатильность', 
            'Баланс стакана'
        ]
        
        print("\n" + "="*50)
        print("VSA РАДАР: АНАЛИЗ РЫНКА")
        print(f"Текущее состояние рынка: {market_state}")
        print(f"Текущий сигнал: {signal} (уверенность: {confidence:.1f}/10.0)")
        print(f"\n{recommendation}")
        
        if future_trend:
            print(f"\n{future_trend}")
        
        if patterns:
            print("\nОбнаруженные паттерны:")
            for s in patterns:
                print(f" - {s}")
        
        print("\nПоказатели радара:")
        for label, value in zip(radar_labels, radar_values):
            print(f"{label}: {value:.2f}")
        
        plot_radar_chart(radar_values.tolist(), radar_labels)
        
        # Вывод информации о стакане
        if order_book:
            print("\nТЕКУЩИЙ СТАКАН:")
            print(f"Спред: {order_book['bid_ask_spread']:.2f}")
            print(f"Имбаланс: {order_book['order_book_imbalance']:.2%}")
            print(f"Лучший bid: {order_book['bids'][0][0]:.2f} ({order_book['bids'][0][1]:.4f} BTC)")
            print(f"Лучший ask: {order_book['asks'][0][0]:.2f} ({order_book['asks'][0][1]:.4f} BTC)")
        
    except Exception as e:
        print(f"Произошла ошибка: {e}")