import ccxt
import datetime

# Создаем объект биржи
exchange = ccxt.binance()

# Настройки
symbol = 'UNI/USDT'         # Можно заменить на любой другой
timeframe = '1h'           # 15 минут
limit = 24                  # Количество свечей

# Загружаем последние 7 свечей
ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

print(f"\nПоследние {limit} свечей по {symbol} (таймфрейм: {timeframe}):\n")

# Перебор и вывод свечей
for i, candle in enumerate(ohlcv):
    timestamp, open_, high, low, close, volume = candle
    time_str = datetime.datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M')

    print(f"Свеча {i+1} ({time_str}):")
    print(f"  Открытие: {open_}")
    print(f"  Закрытие: {close}")
    print(f"  Максимум: {high}")
    print(f"  Минимум: {low}")
    print(f"  Объём: {volume}")
    print(" " * 1)

print("Как думаешь куда пойдет цена?")
