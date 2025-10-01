import ccxt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta
import ta

# 1. Загрузка данных
exchange = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '1h'
limit = 180 * 24

ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# 2. Расширенная подготовка данных с учетом VSA
# Базовые признаки
df['spread'] = df['high'] - df['low']
df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
df['money_flow'] = df['typical_price'] * df['volume']
df['close_diff'] = df['close'].diff()

# Тип свечи
df['up_bar'] = (df['close'] > df['open']).astype(int)
df['down_bar'] = (df['close'] < df['open']).astype(int)

# Объемные характеристики
df['avg_volume'] = df['volume'].rolling(window=24).mean()
df['volume_ratio'] = df['volume'] / df['avg_volume']

# Кульминации
df['buying_climax'] = ((df['up_bar'] == 1) & 
                       (df['volume_ratio'] > 1.8) & 
                       (df['spread'] > df['spread'].rolling(24).mean())).astype(int)

df['selling_climax'] = ((df['down_bar'] == 1) & 
                        (df['volume_ratio'] > 1.8) & 
                        (df['spread'] > df['spread'].rolling(24).mean())).astype(int)

# Накопление/распределение
df['accumulation'] = ((df['close'] > df['open']) & 
                      (df['volume_ratio'] > 1.2) & 
                      (df['spread'] < df['spread'].rolling(24).mean() * 0.7)).astype(int)

df['distribution'] = ((df['close'] < df['open']) & 
                      (df['volume_ratio'] > 1.2) & 
                      (df['spread'] < df['spread'].rolling(24).mean() * 0.7)).astype(int)

# Сила тренда
df['trend_strength'] = df['close'].rolling(24).apply(lambda x: (x[-1] - x[0]) / x.mean())

# Новые VSA признаки (спрос/предложение)
df['demand_supply_ratio'] = (df['high'] - df['close']) / (df['close'] - df['low'] + 1e-10)
df['demand_deficit'] = ((df['high'] - df['close']) / df['spread']).rolling(12).mean()
df['supply_deficit'] = ((df['close'] - df['low']) / df['spread']).rolling(12).mean()

# Уровни сопротивления/поддержки
df['resistance'] = df['high'].rolling(24).max()
df['support'] = df['low'].rolling(24).min()

# Поведение толпы (волатильность + объем)
df['crowd_behavior'] = (df['spread'] * df['volume_ratio']).rolling(12).mean()

# Сила рынка
df['market_strength'] = (df['close'] - df['open']) / df['spread']

# Тренд (скользящие средние)
df['ma_fast'] = df['close'].rolling(12).mean()
df['ma_slow'] = df['close'].rolling(24).mean()
df['trend_direction'] = np.where(df['ma_fast'] > df['ma_slow'], 1, -1)

# Покупательское давление
df['buy_pressure'] = ((df['close'] - df['low']) / df['spread']).rolling(6).mean()

# Продавцовское давление
df['sell_pressure'] = ((df['high'] - df['close']) / df['spread']).rolling(6).mean()

# Заполняем пропуски
df.fillna(0, inplace=True)
df.replace([np.inf, -np.inf], 0, inplace=True)

# 3. Масштабирование данных
features = [
    'open', 'high', 'low', 'close', 'volume', 'spread', 
    'typical_price', 'money_flow', 'close_diff',
    'up_bar', 'down_bar', 'buying_climax', 'selling_climax',
    'accumulation', 'distribution', 'trend_strength',
    'demand_supply_ratio', 'demand_deficit', 'supply_deficit',
    'resistance', 'support', 'crowd_behavior', 'market_strength',
    'trend_direction', 'buy_pressure', 'sell_pressure'
]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[features])

# 4. Создание последовательностей для VSA-модели
def create_vsa_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data)-seq_length-1):
        X.append(data[i:i+seq_length])
        # Цель: OHLC + ключевые VSA-сигналы
        next_data = data[i+seq_length]
        y.append(np.concatenate([
            next_data[:4],  # OHLC
            next_data[9:11],  # buying_climax, selling_climax
            next_data[13:15],  # accumulation, distribution
            next_data[24:26]  # buy_pressure, sell_pressure
        ]))
    return np.array(X), np.array(y)

SEQ_LENGTH = 72
X, y = create_vsa_sequences(scaled_data, SEQ_LENGTH)

# Разделение данных
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. Построение VSA-ориентированной модели LSTM
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(256, return_sequences=True, input_shape=(SEQ_LENGTH, len(features))),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.LSTM(192, return_sequences=True),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(96, activation='relu'),
    tf.keras.layers.Dense(10)  # OHLC + 6 VSA-сигналов
])

# Компиляция
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
              loss=tf.keras.losses.Huber(),
              metrics=['mae'])

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

print("Training VSA model...")
history = model.fit(X_train, y_train, 
                    epochs=1, 
                    batch_size=128,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping, lr_scheduler],
                    verbose=1)

# 6. Прогнозирование
print("Making predictions...")
train_predict = model.predict(X_train, verbose=0)
test_predict = model.predict(X_test, verbose=0)
last_sequence = scaled_data[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, len(features))
next_hour_pred = model.predict(last_sequence, verbose=0)

# 7. Обратное масштабирование (исправленная версия)
def inverse_transform_vsa(data):
    # Создаем полный массив с фиктивными значениями
    full_data = np.zeros((data.shape[0], len(features)))
    
    # Заполняем известные значения
    full_data[:, :4] = data[:, :4]  # OHLC
    
    # VSA-сигналы
    full_data[:, 9:11] = data[:, 4:6]  # buying_climax, selling_climax
    full_data[:, 13:15] = data[:, 6:8]  # accumulation, distribution
    full_data[:, 24:26] = data[:, 8:10]  # buy_pressure, sell_pressure
    
    # Обратное масштабирование
    unscaled = scaler.inverse_transform(full_data)
    
    # Возвращаем только нужные колонки
    return unscaled[:, :10]  # OHLC + 6 VSA-сигналов

train_predict = inverse_transform_vsa(train_predict)
test_predict = inverse_transform_vsa(test_predict)
next_hour_values = inverse_transform_vsa(next_hour_pred)[0]

# 8. Подготовка данных для визуализации
plot_df = df[['open', 'high', 'low', 'close', 'volume']].copy()

# Добавляем прогнозы
for col in ['train_open', 'train_high', 'train_low', 'train_close']:
    plot_df[col] = np.nan

for col in ['test_open', 'test_high', 'test_low', 'test_close']:
    plot_df[col] = np.nan

# Заполняем тренировочные прогнозы
train_start_idx = SEQ_LENGTH
train_end_idx = train_start_idx + len(train_predict)
train_cols = ['train_open', 'train_high', 'train_low', 'train_close']

for i, col in enumerate(train_cols):
    plot_df.loc[plot_df.index[train_start_idx:train_end_idx], col] = train_predict[:, i]

# Заполняем тестовые прогнозы
test_start_idx = train_end_idx + 1
test_end_idx = test_start_idx + len(test_predict)
test_cols = ['test_open', 'test_high', 'test_low', 'test_close']

for i, col in enumerate(test_cols):
    plot_df.loc[plot_df.index[test_start_idx:test_end_idx], col] = test_predict[:, i]

# 9. Визуализация (исправленная)
print("Creating visualization...")
fig, axes = plt.subplots(2, 1, figsize=(20, 12), gridspec_kw={'height_ratios': [3, 1]})
ax1 = axes[0]
ax2 = axes[1]

# Дополнительные графики
apd = [
    mpf.make_addplot(plot_df['train_close'], ax=ax1, color='blue', linestyle='--', label='Train Close'),
    mpf.make_addplot(plot_df['test_close'], ax=ax1, color='green', linestyle='--', label='Test Close'),
    mpf.make_addplot(df['resistance'], ax=ax1, color='red', alpha=0.5, label='Resistance'),
    mpf.make_addplot(df['support'], ax=ax1, color='green', alpha=0.5, label='Support')
]

# Фактические свечи
mpf.plot(plot_df[['open', 'high', 'low', 'close', 'volume']], 
         type='candle', 
         style='charles', 
         addplot=apd,
         ax=ax1,
         volume=ax2,
         warn_too_much_data=10000,
         show_nontrading=False)

# Прогнозируемая свеча
pred_time = plot_df.index[-1] + timedelta(hours=1)
ax1.plot([pred_time], [next_hour_values[3]], 'ro', markersize=8, label='Predicted Close')
ax1.vlines(x=pred_time, ymin=next_hour_values[2], ymax=next_hour_values[1], colors='red', linewidth=2)
ax1.hlines(y=next_hour_values[0], xmin=pred_time - timedelta(minutes=10), 
          xmax=pred_time, colors='red', linewidth=2)
ax1.hlines(y=next_hour_values[3], xmin=pred_time, 
          xmax=pred_time + timedelta(minutes=10), colors='red', linewidth=2)

# Разделение выборки
split_date = plot_df.index[split + SEQ_LENGTH]
ax1.axvline(x=split_date, color='purple', linestyle='--', alpha=0.7, label='Train/Test Split')

# Настройки графика
ax1.set_title(f'{symbol} VSA Price Prediction (1-hour forecast)', fontsize=16)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Price (USDT)', fontsize=12)
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.3)

# Информация о прогнозе
pred_text = (f"Next Hour Prediction:\n"
             f"O: {next_hour_values[0]:.2f}  H: {next_hour_values[1]:.2f}\n"
             f"L: {next_hour_values[2]:.2f}  C: {next_hour_values[3]:.2f}\n\n"
             f"VSA Analysis:\n"
             f"Buy Pressure: {next_hour_values[8]:.2f}\n"
             f"Sell Pressure: {next_hour_values[9]:.2f}\n"
             f"Accumulation: {'YES' if next_hour_values[6] > 0.7 else 'NO'}\n"
             f"Distribution: {'YES' if next_hour_values[7] > 0.7 else 'NO'}")

ax1.text(0.02, 0.95, pred_text, transform=ax1.transAxes, 
         fontsize=12, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
plt.savefig('vsa_price_prediction.png', dpi=300)
plt.show()

# 10. Профессиональный VSA-анализ
current_candle = df.iloc[-1]
print("\nProfessional VSA Analysis:")
print(f"Current Resistance: {current_candle['resistance']:.2f}")
print(f"Current Support: {current_candle['support']:.2f}")
print(f"Market Strength: {current_candle['market_strength']:.2f}")

# Анализ спроса/предложения
demand_level = current_candle['buy_pressure']
supply_level = current_candle['sell_pressure']
print(f"Demand Level: {demand_level:.2f}, Supply Level: {supply_level:.2f}")

if demand_level > supply_level + 0.2:
    print("Strong Demand (Buyers in control)")
elif supply_level > demand_level + 0.2:
    print("Strong Supply (Sellers in control)")
else:
    print("Balanced Market")

# Анализ дефицита
if current_candle['demand_deficit'] > 0.6:
    print("Demand Deficit (Lack of buyers)")
if current_candle['supply_deficit'] > 0.6:
    print("Supply Deficit (Lack of sellers)")

# Анализ давления
buy_pressure = next_hour_values[8]
sell_pressure = next_hour_values[9]
pressure_diff = buy_pressure - sell_pressure

if pressure_diff > 0.3:
    print("Strong BUYING pressure expected")
elif pressure_diff < -0.3:
    print("Strong SELLING pressure expected")
else:
    print("Balanced market pressure")

# Анализ тренда
if next_hour_values[3] > current_candle['close']:
    trend_strength = "Bullish"
    if buy_pressure > 0.6:
        strength_desc = "Strong bullish momentum - BUY opportunity"
    else:
        strength_desc = "Moderate bullish momentum - cautious BUY"
else:
    trend_strength = "Bearish"
    if sell_pressure > 0.6:
        strength_desc = "Strong bearish momentum - SELL opportunity"
    else:
        strength_desc = "Moderate bearish momentum - cautious SELL"

print(f"Predicted Trend: {trend_strength}")
print(strength_desc)

# Анализ уровней
resistance_distance = current_candle['resistance'] - next_hour_values[3]
support_distance = next_hour_values[3] - current_candle['support']

if resistance_distance < 0.01 * current_candle['close']:
    print(f"Approaching RESISTANCE at {current_candle['resistance']:.2f} - caution advised")
    
if support_distance < 0.01 * current_candle['close']:
    print(f"Approaching SUPPORT at {current_candle['support']:.2f} - potential bounce")

# Рекомендация
if next_hour_values[6] > 0.7:  # Accumulation
    rec = "STRONG BUY (smart money accumulation)"
elif next_hour_values[7] > 0.7:  # Distribution
    rec = "STRONG SELL (smart money distribution)"
elif buy_pressure > 0.6 and pressure_diff > 0.3:
    rec = "BUY (strong buying pressure)"
elif sell_pressure > 0.6 and pressure_diff < -0.3:
    rec = "SELL (strong selling pressure)"
else:
    rec = "HOLD (neutral market conditions)"

print(f"\nTrade Recommendation: {rec}")

# Контекстный анализ
if next_hour_values[6] > 0.7 and trend_strength == "Bullish":
    print("Context: Bullish trend with accumulation - high probability setup")
elif next_hour_values[7] > 0.7 and trend_strength == "Bearish":
    print("Context: Bearish trend with distribution - high probability setup")
elif next_hour_values[4] > 0.7:
    print("Warning: Potential Buying Climax - possible reversal")
elif next_hour_values[5] > 0.7:
    print("Warning: Potential Selling Climax - possible reversal")