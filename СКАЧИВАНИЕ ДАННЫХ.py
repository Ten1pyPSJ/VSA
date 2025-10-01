import ccxt
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta

# Настройки
exchange = ccxt.binance()  # Можно заменить на bybit, kucoin и др.
symbol = 'BTC/USDT'        # Торговая пара
timeframe = '1d'           # Таймфрейм (1 день)
days = 365                 # Сколько дней данных скачать
output_file = 'crypto_data_with_indicators.csv'  # Выходной файл

# --- 1. Скачиваем OHLCV-данные ---
since = exchange.parse8601((datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S'))
ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=days)

# Создаем DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['date'] = pd.to_datetime(df['timestamp'], unit='ms')

# --- 2. Рассчитываем индикаторы с помощью TA-Lib ---
close_prices = df['close'].values  # TA-Lib работает с numpy-массивами

# EMA (Exponential Moving Average)
df['ema_20'] = talib.EMA(close_prices, timeperiod=20)
df['ema_50'] = talib.EMA(close_prices, timeperiod=50)
df['ema_100'] = talib.EMA(close_prices, timeperiod=100)

# RSI (Relative Strength Index)
df['rsi'] = talib.RSI(close_prices, timeperiod=14)

# MACD (Moving Average Convergence Divergence)
macd, macd_signal, macd_hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
df['macd'] = macd
df['macd_signal'] = macd_signal
df['macd_hist'] = macd_hist

# --- 3. Сохраняем в CSV ---
df.to_csv(output_file, index=False)
print(df)
print(f"Данные сохранены в {output_file}")
print("Колонки:", df.columns.tolist())
print(f"Период: {df['date'].min()} — {df['date'].max()}")