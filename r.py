import ccxt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import datetime
from matplotlib.lines import Line2D

# Настройки
symbol = 'BTC/USDT'
timeframe = '1d'
days = 365
exchange_id = 'binance'  # Можно изменить на другую биржу

def fetch_ohlcv_data():
    exchange = getattr(ccxt, exchange_id)()
    since = exchange.parse8601((datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S'))
    
    print(f"Загрузка данных {symbol} с {exchange_id}...")
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    return df

def plot_candlestick_colored_shadows(df):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # Ширина свечи (в днях)
    candle_width = 0.5 / len(df) * days  # Коэффициент можно регулировать
    
    for idx, row in df.iterrows():
        # Определение типа свечи и цветов
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
        
        # Рисуем тень (фитиль) - цвет зависит от типа свечи
        ax.plot([mdates.date2num(idx), mdates.date2num(idx)], 
                [row['low'], row['high']], 
                color=shadow_color, 
                linewidth=1.5,
                alpha=0.7)
        
        # Рисуем тело свечи
        ax.bar(mdates.date2num(idx), 
               body_top - body_bottom, 
               bottom=body_bottom, 
               width=candle_width, 
               color=body_color, 
               edgecolor='black',
               linewidth=1)
    
    # Настройки графика
    ax.set_title(f'Свечной график {symbol} за последние {days} дней ({exchange_id})', fontsize=16)
    ax.set_xlabel('Дата', fontsize=12)
    ax.set_ylabel('Цена (USDT)', fontsize=12)
    
    # Форматирование даты
    date_format = DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    
    legend_elements = [
        Line2D([0], [0], color='green', lw=4, label='Рост (бычья)'),
        Line2D([0], [0], color='red', lw=4, label='Падение (медвежья)'),
        Line2D([0], [0], color='green', linestyle='-', lw=2, label='Бычья тень'),
        Line2D([0], [0], color='red', linestyle='-', lw=2, label='Медвежья тень')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    try:
        data = fetch_ohlcv_data()
        print("Последние 5 строк данных:")
        print(data.tail())
        plot_candlestick_colored_shadows(data)
    except Exception as e:
        print(f"Произошла ошибка: {e}")