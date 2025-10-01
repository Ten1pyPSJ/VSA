import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
from collections import deque

# Настройки
SYMBOL = 'BTC/USDT'  # Пара для анализа
TIMEFRAME = '4h'     # Таймфрейм
EXCHANGE = 'binance' # Биржа
LOOKBACK = 100       # Количество свечей для анализа
ORDERBOOK_DEPTH = 10 # Глубина стакана для анализа

class MarketRadar:
    def __init__(self):
        # Инициализация подключения к бирже
        self.exchange = getattr(ccxt, EXCHANGE)()
        self.exchange.load_markets()
        self.orderbook_history = deque(maxlen=20)  # История стаканов
        self.market_state = {
            'trend': None,
            'strength': None,
            'phase': None,
            'activity': None,
            'volatility': None,
            'context': None
        }
        
    def fetch_ohlcv(self):
        """Загрузка OHLCV данных"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LOOKBACK)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            return None
    
    def fetch_orderbook(self):
        """Загрузка стакана ордеров"""
        try:
            orderbook = self.exchange.fetch_order_book(SYMBOL, limit=ORDERBOOK_DEPTH)
            self.orderbook_history.append(orderbook)
            return orderbook
        except Exception as e:
            print(f"Ошибка при загрузке стакана: {e}")
            return None
    
    def calculate_indicators(self, df):
        """Расчет технических индикаторов для VSA"""
        # Объемные индикаторы
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Анализ свечей
        df['range'] = df['high'] - df['low']
        df['avg_range'] = df['range'].rolling(window=20).mean()
        df['close_position'] = (df['close'] - df['low']) / df['range']
        df['body_ratio'] = abs(df['close'] - df['open']) / df['range']
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['range']
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['range']
        
        # Волатильность
        df['volatility'] = df['range'].rolling(window=14).mean() / df['close'].rolling(window=14).mean() * 100
        
        # Уровни поддержки/сопротивления
        df['support'] = df['low'].rolling(window=5).min()
        df['resistance'] = df['high'].rolling(window=5).max()
        
        return df
    
    def analyze_market_state(self, df):
        """Анализ общего состояния рынка на основе VSA"""
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Определение тренда по объему и закрытию
        if last['close'] > prev['close'] and last['volume_ratio'] > 1.2:
            self.market_state['trend'] = 'Восходящий'
        elif last['close'] < prev['close'] and last['volume_ratio'] > 1.2:
            self.market_state['trend'] = 'Нисходящий'
        else:
            self.market_state['trend'] = 'Боковой'
        
        # Определение силы тренда
        if last['volume_ratio'] > 2.0:
            self.market_state['strength'] = 'Сильный'
        elif last['volume_ratio'] > 1.5:
            self.market_state['strength'] = 'Умеренный'
        else:
            self.market_state['strength'] = 'Слабый'
        
        # Определение фазы рынка
        if last['close'] > last['open'] and last['volume_ratio'] > 1.5 and last['close_position'] > 0.7:
            self.market_state['phase'] = 'Аккумуляция'
        elif last['close'] < last['open'] and last['volume_ratio'] > 1.5 and last['close_position'] < 0.3:
            self.market_state['phase'] = 'Распределение'
        else:
            self.market_state['phase'] = 'Нейтральная'
        
        # Определение активности
        if last['volume_ratio'] > 2.0:
            self.market_state['activity'] = 'Высокая активность'
        elif last['volume_ratio'] > 1.5:
            self.market_state['activity'] = 'Повышенная активность'
        else:
            self.market_state['activity'] = 'Низкая активность'
            
        # Определение волатильности
        if last['volatility'] > 3:
            self.market_state['volatility'] = 'Высокая'
        elif last['volatility'] > 1.5:
            self.market_state['volatility'] = 'Умеренная'
        else:
            self.market_state['volatility'] = 'Низкая'
            
        # Определение рыночного контекста
        if (self.market_state['trend'] == 'Восходящий' and 
            self.market_state['phase'] == 'Распределение'):
            self.market_state['context'] = 'Коррекция после роста'
        elif (self.market_state['trend'] == 'Нисходящий' and 
              self.market_state['phase'] == 'Аккумуляция'):
            self.market_state['context'] = 'Отскок после падения'
        elif self.market_state['trend'] == 'Боковой':
            self.market_state['context'] = 'Консолидация'
        else:
            self.market_state['context'] = 'Трендовое движение'
    
    def analyze_orderbook(self, orderbook):
        """Улучшенный анализ стакана ордеров"""
        if not orderbook or len(self.orderbook_history) < 5:
            return []
        
        bids = orderbook['bids']
        asks = orderbook['asks']
        
        # Расчет суммарного объема в стакане
        total_bid_volume = sum([bid[1] for bid in bids])
        total_ask_volume = sum([ask[1] for ask in asks])
        
        # Анализ ликвидности
        liquidity_ratio = total_bid_volume / total_ask_volume
        
        signals = []
        
        # Определение силы/слабости по стакану
        if liquidity_ratio > 2.0:
            signals.append(f"Очень сильный спрос в стакане (соотношение {liquidity_ratio:.1f})")
        elif liquidity_ratio > 1.5:
            signals.append(f"Сильный спрос в стакане (соотношение {liquidity_ratio:.1f})")
        elif liquidity_ratio < 0.5:
            signals.append(f"Очень сильное предложение в стакане (соотношение {liquidity_ratio:.1f})")
        elif liquidity_ratio < 0.67:
            signals.append(f"Сильное предложение в стакане (соотношение {liquidity_ratio:.1f})")
        
        # Анализ изменения стакана
        prev_orderbook = self.orderbook_history[-2] if len(self.orderbook_history) > 1 else None
        if prev_orderbook:
            prev_bid = prev_orderbook['bids'][0][0] if prev_orderbook['bids'] else 0
            prev_ask = prev_orderbook['asks'][0][0] if prev_orderbook['asks'] else 0
            current_bid = bids[0][0] if bids else 0
            current_ask = asks[0][0] if asks else 0
            
            if current_bid > prev_ask:
                signals.append("Агрессивные покупки - цена пробила предыдущий ask")
            elif current_ask < prev_bid:
                signals.append("Агрессивные продажи - цена пробила предыдущий bid")
        
        return signals
    
    def vsa_analysis(self, df):
        """Улучшенный анализ VSA (Volume Spread Analysis)"""
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        prev_prev_candle = df.iloc[-3]
        
        signals = []
        details = []
        
        # 1. Сильный спрос (восходящий тренд)
        if (last_candle['close'] > last_candle['open'] and 
            last_candle['volume_ratio'] > 1.5 and 
            last_candle['range'] > last_candle['avg_range'] and
            last_candle['close_position'] > 0.7 and
            last_candle['body_ratio'] > 0.5 and
            last_candle['lower_shadow'] < 0.2):
            signals.append(f"📈 Сильный спрос (объем x{last_candle['volume_ratio']:.1f}, крупные покупатели активны)")
            details.append("🔹 Цена закрылась в верхней части диапазона с большим объемом")
            details.append("🔹 Широкая свеча с маленькой нижней тенью")
            details.append("🔹 Покупатели контролируют ситуацию")
            details.append("🔹 Сигнал подтверждается, если предыдущая свеча была нисходящей")
        
        # 2. Слабость на подъеме (возможный разворот)
        elif (last_candle['close'] > last_candle['open'] and 
              last_candle['volume_ratio'] < 0.7 and 
              last_candle['range'] > last_candle['avg_range'] and
              last_candle['body_ratio'] < 0.3 and
              last_candle['upper_shadow'] > 0.5):
            signals.append(f"⚠️ Слабость на подъеме (объем x{last_candle['volume_ratio']:.1f}, покупатели выдыхаются)")
            details.append("🔹 Цена выросла, но объем ниже среднего")
            details.append("🔹 Большой диапазон с маленьким телом и длинной верхней тенью")
            details.append("🔹 Покупатели не подтверждают движение")
            details.append("🔹 Особенно сильный сигнал после длительного роста")
        
        # 3. Сильное предложение (нисходящий тренд)
        elif (last_candle['close'] < last_candle['open'] and 
              last_candle['volume_ratio'] > 1.5 and 
              last_candle['range'] > last_candle['avg_range'] and
              last_candle['close_position'] < 0.3 and
              last_candle['body_ratio'] > 0.5 and
              last_candle['upper_shadow'] < 0.2):
            signals.append(f"📉 Сильное предложение (объем x{last_candle['volume_ratio']:.1f}, крупные продавцы активны)")
            details.append("🔹 Цена закрылась в нижней части диапазона с большим объемом")
            details.append("🔹 Широкая медвежья свеча с маленькой верхней тенью")
            details.append("🔹 Продавцы контролируют ситуацию")
            details.append("🔹 Сигнал подтверждается, если предыдущая свеча была восходящей")
        
        # 4. Слабость на падении (возможный разворот)
        elif (last_candle['close'] < last_candle['open'] and 
              last_candle['volume_ratio'] < 0.7 and 
              last_candle['range'] > last_candle['avg_range'] and
              last_candle['body_ratio'] < 0.3 and
              last_candle['lower_shadow'] > 0.5):
            signals.append(f"⚠️ Слабость на падении (объем x{last_candle['volume_ratio']:.1f}, продавцы выдыхаются)")
            details.append("🔹 Цена упала, но объем ниже среднего")
            details.append("🔹 Большой диапазон с маленьким телом и длинной нижней тенью")
            details.append("🔹 Продавцы не подтверждают движение")
            details.append("🔹 Особенно сильный сигнал после длительного падения")
        
        # 5. Тест на предложение
        elif (last_candle['close'] < prev_candle['close'] and
              last_candle['volume'] > last_candle['volume_sma_20'] * 1.5 and
              last_candle['close'] > (last_candle['high'] + last_candle['low']) / 2 and
              last_candle['range'] > last_candle['avg_range'] and
              last_candle['upper_shadow'] > 0.3):
            signals.append("🔍 Тест на предложение - крупные игроки проверяют уровень")
            details.append("🔹 Большой объем при тестировании уровня сопротивления")
            details.append("🔹 Цена закрылась выше середины диапазона")
            details.append("🔹 Длинная верхняя тень показывает отказ от уровня")
            details.append("🔹 Возможна остановка нисходящего движения")
        
        # 6. Тест на спрос
        elif (last_candle['close'] > prev_candle['close'] and
              last_candle['volume'] > last_candle['volume_sma_20'] * 1.5 and
              last_candle['close'] < (last_candle['high'] + last_candle['low']) / 2 and
              last_candle['range'] > last_candle['avg_range'] and
              last_candle['lower_shadow'] > 0.3):
            signals.append("🔍 Тест на спрос - крупные игроки проверяют уровень")
            details.append("🔹 Большой объем при тестировании уровня поддержки")
            details.append("🔹 Цена закрылась ниже середины диапазона")
            details.append("🔹 Длинная нижняя тень показывает отказ от уровня")
            details.append("🔹 Возможна остановка восходящего движения")
        
        # 7. Абсорбция покупателей
        elif (last_candle['close'] < last_candle['open'] and
              last_candle['volume'] > last_candle['volume_sma_20'] * 2.5 and
              last_candle['close'] == last_candle['low'] and
              last_candle['body_ratio'] > 0.8 and
              prev_candle['close'] > prev_candle['open']):
            signals.append("🛑 Абсорбция покупателей - сильное давление продавцов")
            details.append("🔹 Очень большой объем на нисходящей свече")
            details.append("🔹 Цена закрылась на минимуме без нижней тени")
            details.append("🔹 Продавцы поглотили всех покупателей")
            details.append("🔹 Особенно сильный сигнал после восходящего движения")
        
        # 8. Абсорбция продавцов
        elif (last_candle['close'] > last_candle['open'] and
              last_candle['volume'] > last_candle['volume_sma_20'] * 2.5 and
              last_candle['close'] == last_candle['high'] and
              last_candle['body_ratio'] > 0.8 and
              prev_candle['close'] < prev_candle['open']):
            signals.append("🛑 Абсорбция продавцов - сильное давление покупателей")
            details.append("🔹 Очень большой объем на восходящей свече")
            details.append("🔹 Цена закрылась на максимуме без верхней тени")
            details.append("🔹 Покупатели поглотили всех продавцов")
            details.append("🔹 Особенно сильный сигнал после нисходящего движения")
        
        # 9. Накопление (улучшенные условия)
        elif (last_candle['range'] < last_candle['avg_range'] * 0.6 and
              last_candle['volume'] > last_candle['volume_sma_20'] * 1.8 and
              abs(last_candle['close'] - last_candle['open']) / last_candle['range'] < 0.2 and
              last_candle['upper_shadow'] > 0.3 and
              last_candle['lower_shadow'] > 0.3 and
              prev_candle['volume'] < prev_candle['volume_sma_20']):
            signals.append("🔄 Накопление - крупные игроки набирают позиции")
            details.append("🔹 Очень маленький диапазон с очень большим объемом")
            details.append("🔹 Длинные тени с обеих сторон (доджи или маленькое тело)")
            details.append("🔹 Предыдущий объем был ниже среднего")
            details.append("🔹 Крупные игроки аккумулируют позиции")
        
        # 10. Распределение (улучшенные условия)
        elif (last_candle['range'] < last_candle['avg_range'] * 0.6 and
              last_candle['volume'] > last_candle['volume_sma_20'] * 1.8 and
              abs(last_candle['close'] - last_candle['open']) / last_candle['range'] < 0.2 and
              last_candle['upper_shadow'] > 0.3 and
              last_candle['lower_shadow'] > 0.3 and
              prev_candle['volume'] > prev_candle['volume_sma_20'] * 1.5 and
              prev_candle['close'] > prev_candle['open']):
            signals.append("🔄 Распределение - крупные игроки закрывают позиции")
            details.append("🔹 Очень маленький диапазон с очень большим объемом")
            details.append("🔹 Длинные тени с обеих сторон (доджи или маленькое тело)")
            details.append("🔹 Предыдущая свеча была восходящей с большим объемом")
            details.append("🔹 Крупные игроки распродают позиции")
        
        # 11. Усиление спроса (улучшенные условия)
        elif (last_candle['close'] > last_candle['open'] and
              last_candle['volume_ratio'] > 1.5 and
              last_candle['volume_ratio'] > prev_candle['volume_ratio'] and
              last_candle['close'] > prev_candle['close'] and
              prev_candle['close'] > prev_candle['open'] and
              prev_candle['volume_ratio'] > 1.2):
            signals.append("📈 Усиление спроса - объемы растут вместе с ценой")
            details.append("🔹 Последовательное увеличение объема и цены")
            details.append("🔹 Текущая и предыдущая свечи восходящие с объемом выше среднего")
            details.append("🔹 Подтверждение восходящего движения")
            details.append("🔹 Вход в позицию на откатах")
        
        # 12. Усиление предложения (улучшенные условия)
        elif (last_candle['close'] < last_candle['open'] and
              last_candle['volume_ratio'] > 1.5 and
              last_candle['volume_ratio'] > prev_candle['volume_ratio'] and
              last_candle['close'] < prev_candle['close'] and
              prev_candle['close'] < prev_candle['open'] and
              prev_candle['volume_ratio'] > 1.2):
            signals.append("📉 Усиление предложения - объемы растут вместе с падением цены")
            details.append("🔹 Последовательное увеличение объема при падении цены")
            details.append("🔹 Текущая и предыдущая свечи нисходящие с объемом выше среднего")
            details.append("🔹 Подтверждение нисходящего движения")
            details.append("🔹 Вход в позицию на откатах")
        
        return signals, details
    
    def candle_patterns(self, df):
        """Улучшенный анализ свечных моделей с учетом объема"""
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        prev_prev_candle = df.iloc[-3]
        
        patterns = []
        details = []
        
        # Молот (улучшенные условия)
        if (last_candle['close'] > last_candle['open'] and 
            last_candle['body_ratio'] < 0.3 and 
            last_candle['lower_shadow'] > 0.6 and
            last_candle['upper_shadow'] < 0.1 and
            last_candle['volume_ratio'] > 1.2 and
            last_candle['low'] == min(last_candle['low'], prev_candle['low'], prev_prev_candle['low'])):
            patterns.append("🔨 Молот с подтверждением объема - сильный сигнал разворота вверх")
            details.append("🔹 Маленькое тело в верхней части диапазона")
            details.append("🔹 Длинная нижняя тень (минимум 2/3 диапазона)")
            details.append("🔹 Почти нет верхней тени")
            details.append("🔹 Объем выше среднего подтверждает сигнал")
            details.append("🔹 Появился после нисходящего движения")
        
        # Повешенный (улучшенные условия)
        elif (last_candle['close'] < last_candle['open'] and 
              last_candle['body_ratio'] < 0.3 and 
              last_candle['upper_shadow'] > 0.6 and
              last_candle['lower_shadow'] < 0.1 and
              last_candle['volume_ratio'] > 1.2 and
              last_candle['high'] == max(last_candle['high'], prev_candle['high'], prev_prev_candle['high'])):
            patterns.append("🪢 Повешенный с подтверждением объема - сильный сигнал разворота вниз")
            details.append("🔹 Маленькое тело в нижней части диапазона")
            details.append("🔹 Длинная верхняя тень (минимум 2/3 диапазона)")
            details.append("🔹 Почти нет нижней тени")
            details.append("🔹 Объем выше среднего подтверждает сигнал")
            details.append("🔹 Появился после восходящего движения")
        
        # Поглощение (улучшенные условия)
        if (last_candle['close'] > prev_candle['open'] and 
            last_candle['open'] < prev_candle['close'] and 
            last_candle['close'] > prev_candle['close'] and 
            last_candle['open'] < prev_candle['open'] and
            last_candle['volume_ratio'] > 1.5 and
            prev_candle['volume_ratio'] < 0.8):
            patterns.append("🟢 Бычье поглощение с подтверждением объема - очень сильный сигнал")
            details.append("🔹 Текущая свеча полностью 'поглотила' предыдущую")
            details.append("🔹 Закрытие выше открытия предыдущей свечи")
            details.append("🔹 Большой объем на текущей свече и малый на предыдущей")
            details.append("🔹 Предыдущая свеча была нисходящей")
            details.append("🔹 Лучший сигнал после нисходящего движения")
        elif (last_candle['close'] < prev_candle['open'] and 
              last_candle['open'] > prev_candle['close'] and 
              last_candle['close'] < prev_candle['close'] and 
              last_candle['open'] > prev_candle['open'] and
              last_candle['volume_ratio'] > 1.5 and
              prev_candle['volume_ratio'] < 0.8):
            patterns.append("🔴 Медвежье поглощение с подтверждением объема - очень сильный сигнал")
            details.append("🔹 Текущая свеча полностью 'поглотила' предыдущую")
            details.append("🔹 Закрытие ниже открытия предыдущей свечи")
            details.append("🔹 Большой объем на текущей свече и малый на предыдущей")
            details.append("🔹 Предыдущая свеча была восходящей")
            details.append("🔹 Лучший сигнал после восходящего движения")
        
        # Утренняя звезда (улучшенные условия)
        if (prev_candle['close'] < prev_candle['open'] and
            prev_candle['body_ratio'] > 0.7 and
            last_candle['close'] > last_candle['open'] and
            last_candle['open'] < prev_candle['close'] and
            last_candle['body_ratio'] > 0.5 and
            prev_prev_candle['close'] < prev_prev_candle['open'] and
            last_candle['volume_ratio'] > 1.5 and
            prev_candle['volume_ratio'] < 0.7):
            patterns.append("🌟 Утренняя звезда с подтверждением объема - сильный бычий разворот")
            details.append("🔹 После нисходящей свечи идет свеча с маленьким телом и малым объемом")
            details.append("🔹 Завершается сильной восходящей свечой с большим объемом")
            details.append("🔹 Все три свечи подтверждают разворот")
            details.append("🔹 Лучший сигнал после четкого нисходящего движения")
        
        # Вечерняя звезда (улучшенные условия)
        elif (prev_candle['close'] > prev_candle['open'] and
              prev_candle['body_ratio'] > 0.7 and
              last_candle['close'] < last_candle['open'] and
              last_candle['open'] > prev_candle['close'] and
              last_candle['body_ratio'] > 0.5 and
              prev_prev_candle['close'] > prev_prev_candle['open'] and
              last_candle['volume_ratio'] > 1.5 and
              prev_candle['volume_ratio'] < 0.7):
            patterns.append("🌙 Вечерняя звезда с подтверждением объема - сильный медвежий разворот")
            details.append("🔹 После восходящей свечи идет свеча с маленьким телом и малым объемом")
            details.append("🔹 Завершается сильной нисходящей свечой с большим объемом")
            details.append("🔹 Все три свечи подтверждают разворот")
            details.append("🔹 Лучший сигнал после четкого восходящего движения")
        
        # Пин-бар (улучшенные условия)
        if (last_candle['body_ratio'] < 0.3 and
            last_candle['upper_shadow'] > 0.7 and
            last_candle['lower_shadow'] < 0.1 and
            last_candle['volume_ratio'] > 1.3 and
            prev_candle['close'] < prev_candle['open']):
            patterns.append("📌 Медвежий пин-бар с подтверждением объема - сильный сигнал")
            details.append("🔹 Очень маленькое тело в нижней части диапазона")
            details.append("🔹 Очень длинная верхняя тень (70% и более диапазона)")
            details.append("🔹 Почти нет нижней тени")
            details.append("🔹 Объем выше среднего подтверждает сигнал")
            details.append("🔹 Появился после нисходящего движения")
        elif (last_candle['body_ratio'] < 0.3 and
              last_candle['lower_shadow'] > 0.7 and
              last_candle['upper_shadow'] < 0.1 and
              last_candle['volume_ratio'] > 1.3 and
              prev_candle['close'] > prev_candle['open']):
            patterns.append("📌 Бычий пин-бар с подтверждением объема - сильный сигнал")
            details.append("🔹 Очень маленькое тело в верхней части диапазона")
            details.append("🔹 Очень длинная нижняя тень (70% и более диапазона)")
            details.append("🔹 Почти нет верхней тени")
            details.append("🔹 Объем выше среднего подтверждает сигнал")
            details.append("🔹 Появился после восходящего движения")
        
        return patterns, details
    
    def generate_market_commentary(self):
        """Генерация комментария о состоянии рынка на основе VSA"""
        commentary = []
        details = []
        
        # Общее состояние
        commentary.append(f"\n📌 Текущее состояние рынка (VSA анализ):")
        commentary.append(f"📊 Тренд: {self.market_state['trend']}")
        commentary.append(f"💪 Сила: {self.market_state['strength']}")
        commentary.append(f"🔄 Фаза: {self.market_state['phase']}")
        commentary.append(f"🏃 Активность: {self.market_state['activity']}")
        commentary.append(f"🌪️ Волатильность: {self.market_state['volatility']}")
        commentary.append(f"📌 Контекст: {self.market_state['context']}")
        
        # Детали состояния
        details.append("\n🔍 Детали текущего состояния:")
        
        if self.market_state['trend'] == 'Восходящий':
            details.append("🔹 Рынок находится в восходящем тренде по VSA")
            if self.market_state['strength'] == 'Сильный':
                details.append("🔹 Сильные объемы подтверждают тренд")
                details.append("🔹 Ищите возможности для покупки на откатах")
            elif self.market_state['strength'] == 'Умеренный':
                details.append("🔹 Умеренные объемы поддерживают тренд")
                details.append("🔹 Будьте осторожны с новыми входами")
            else:
                details.append("🔹 Слабые объемы - тренд может быть под угрозой")
                details.append("🔹 Ожидайте подтверждения перед входом")
        elif self.market_state['trend'] == 'Нисходящий':
            details.append("🔹 Рынок находится в нисходящем тренде по VSA")
            if self.market_state['strength'] == 'Сильный':
                details.append("🔹 Сильные объемы подтверждают тренд")
                details.append("🔹 Ищите возможности для продажи на откатах")
            elif self.market_state['strength'] == 'Умеренный':
                details.append("🔹 Умеренные объемы поддерживают тренд")
                details.append("🔹 Будьте осторожны с новыми входами")
            else:
                details.append("🔹 Слабые объемы - тренд может быть под угрозой")
                details.append("🔹 Ожидайте подтверждения перед входом")
        else:
            details.append("🔹 Рынок находится в боковом движении")
            details.append("🔹 Нет четкого направления по объему")
            details.append("🔹 Ищите пробои уровней с подтверждением объема")
        
        # Интерпретация фазы
        if self.market_state['phase'] == 'Аккумуляция':
            commentary.append("\nℹ️ VSA показывает фазу аккумуляции. Крупные игроки накапливают позиции.")
            details.append("🔹 Ожидайте потенциального роста после завершения фазы")
            details.append("🔹 Ищите подтверждающие сигналы для входа")
        elif self.market_state['phase'] == 'Распределение':
            commentary.append("\nℹ️ VSA показывает фазу распределения. Крупные игроки распродают позиции.")
            details.append("🔹 Ожидайте потенциального падения после завершения фазы")
            details.append("🔹 Ищите подтверждающие сигналы для входа в шорт")
        else:
            commentary.append("\nℹ️ Рынок в нейтральной фазе. Нет четких признаков аккумуляции или распределения.")
            details.append("🔹 Ожидайте более четких сигналов")
            details.append("🔹 Торгуйте осторожно")
        
        # Рекомендации по активности
        if self.market_state['activity'] == 'Высокая активность':
            commentary.append("\n💡 Высокая торговая активность указывает на сильное участие крупных игроков.")
            details.append("🔹 Большие объемы - активность профессионалов")
            details.append("🔹 Сильные движения более вероятны")
        elif self.market_state['activity'] == 'Повышенная активность':
            commentary.append("\n💡 Повышенная активность может указывать на накопление или распределение.")
            details.append("🔹 Объемы выше среднего - возможны манипуляции")
            details.append("🔹 Анализируйте контекст для понимания ситуации")
        else:
            commentary.append("\n💡 Низкая активность означает отсутствие интереса крупных игроков.")
            details.append("🔹 Маленькие объемы - слабое участие")
            details.append("🔹 Возможны ложные движения")
        
        return commentary, details
    
    def generate_recommendation(self, vsa, vsa_details, orderbook, patterns, patterns_details):
        """Генерация рекомендации на основе VSA и свечного анализа"""
        recommendation = []
        details = []
        confidence = 0
        
        # Оценка силы сигналов
        bullish_signals = 0
        bearish_signals = 0
        
        # Анализ VSA (увеличиваем вес VSA сигналов)
        for signal in vsa:
            if "спрос" in signal.lower() or "покуп" in signal.lower() or "быч" in signal.lower():
                bullish_signals += 3  # Увеличиваем вес VSA сигналов
            elif "предложение" in signal.lower() or "продаж" in signal.lower() or "медвеж" in signal.lower():
                bearish_signals += 3
        
        # Анализ стакана
        for signal in orderbook:
            if "спрос" in signal.lower():
                bullish_signals += 2
            elif "предложение" in signal.lower():
                bearish_signals += 2
        
        # Анализ свечных моделей
        for pattern in patterns:
            if "быч" in pattern.lower() or "вверх" in pattern.lower():
                bullish_signals += 2
            elif "медвеж" in pattern.lower() or "вниз" in pattern.lower():
                bearish_signals += 2
        
        # Учет рыночного состояния
        if self.market_state['trend'] == 'Восходящий':
            bullish_signals += 1.5
        elif self.market_state['trend'] == 'Нисходящий':
            bearish_signals += 1.5
            
        if self.market_state['strength'] == 'Сильный':
            if self.market_state['trend'] == 'Восходящий':
                bullish_signals += 1
            else:
                bearish_signals += 1
        
        # Формирование рекомендации
        signal_diff = bullish_signals - bearish_signals
        
        if signal_diff > 6:
            recommendation.append("🟢🟢🟢 ОЧЕНЬ СИЛЬНЫЙ СИГНАЛ НА ПОКУПКУ (VSA)")
            confidence = min(95, 75 + signal_diff * 3)
        elif signal_diff > 4:
            recommendation.append("🟢🟢 СИЛЬНЫЙ СИГНАЛ НА ПОКУПКУ (VSA)")
            confidence = min(85, 65 + signal_diff * 4)
        elif signal_diff > 2:
            recommendation.append("🟢 Умеренный сигнал на покупку (VSA)")
            confidence = 60 + signal_diff * 5
        elif signal_diff > 0:
            recommendation.append("🟢 Слабый сигнал на покупку (VSA)")
            confidence = 55 + signal_diff * 5
        elif signal_diff < -6:
            recommendation.append("🔴🔴🔴 ОЧЕНЬ СИЛЬНЫЙ СИГНАЛ НА ПРОДАЖУ (VSA)")
            confidence = min(95, 75 + abs(signal_diff) * 3)
        elif signal_diff < -4:
            recommendation.append("🔴🔴 СИЛЬНЫЙ СИГНАЛ НА ПРОДАЖУ (VSA)")
            confidence = min(85, 65 + abs(signal_diff) * 4)
        elif signal_diff < -2:
            recommendation.append("🔴 Умеренный сигнал на продажу (VSA)")
            confidence = 60 + abs(signal_diff) * 5
        elif signal_diff < 0:
            recommendation.append("🔴 Слабый сигнал на продажу (VSA)")
            confidence = 55 + abs(signal_diff) * 5
        else:
            recommendation.append("⚪ Нет четкого сигнала по VSA")
            confidence = 50
        
        recommendation.append(f"📊 Уверенность: {confidence}%")
        
        # Конкретные рекомендации
        if confidence > 80:
            recommendation.append("\n💡 РЕКОМЕНДАЦИЯ: Сильный VSA сигнал. Рассмотрите вход в сделку с нормальным размером позиции.")
            details.append("🔹 Вход по текущей цене или на откате")
            details.append("🔹 Используйте нормальный размер позиции")
            details.append("🔹 Установите стоп-лосс за ближайший значимый уровень")
        elif confidence > 70:
            recommendation.append("\n💡 Рекомендация: Хороший VSA сигнал. Возможен вход в сделку с умеренным размером позиции.")
            details.append("🔹 Вход на подтверждении сигнала")
            details.append("🔹 Умеренный размер позиции")
            details.append("🔹 Установите стоп-лосс за ближайший значимый уровень")
        elif confidence > 60:
            recommendation.append("\n💡 Рекомендация: Умеренный VSA сигнал. Вход с уменьшенным размером позиции.")
            details.append("🔹 Ожидайте дополнительных подтверждений")
            details.append("🔹 Уменьшенный размер позиции")
            details.append("🔹 Установите более широкий стоп-лосс")
        elif confidence > 50:
            recommendation.append("\n💡 Рекомендация: Слабый VSA сигнал. Вход с минимальным размером позиции или ожидание.")
            details.append("🔹 Ожидайте четких подтверждений")
            details.append("🔹 Минимальный размер позиции или ожидание")
        else:
            recommendation.append("\n💡 Рекомендация: Оставайтесь вне рынка. Ожидайте более четких VSA сигналов.")
            details.append("🔹 Нет четких сигналов для входа")
            details.append("🔹 Лучше оставаться вне рынка")
        
        # Управление рисками с учетом волатильности
        recommendation.append("\n⚠️ УПРАВЛЕНИЕ РИСКАМИ:")
        recommendation.append("- Всегда используйте стоп-лосс")
        
        if self.market_state['volatility'] == 'Высокая':
            recommendation.append("- Рискуйте не более 0.5-1% капитала на сделку из-за высокой волатильности")
            recommendation.append("- Увеличьте расстояние до стоп-лосса")
            details.append("🔹 Высокая волатильность - уменьшите риск")
            details.append("🔹 Большие стопы из-за широких диапазонов")
        elif self.market_state['volatility'] == 'Умеренная':
            recommendation.append("- Рискуйте 1-1.5% капитала на сделку")
            details.append("🔹 Умеренная волатильность - стандартный риск")
        else:
            recommendation.append("- Рискуйте 1-2% капитала на сделку")
            details.append("🔹 Низкая волатильность - можно рисковать больше")
        
        # Дополнительные рекомендации по VSA
        if any("спрос" in s.lower() for s in vsa) and confidence > 60:
            recommendation.append("\n📈 VSA показывает активность покупателей. Ищите возможности для входа в лонг.")
            details.append("🔹 Крупные игроки покупают - присоединяйтесь")
            details.append("🔹 Ищите точки входа на откатах к уровням поддержки")
        elif any("предложение" in s.lower() for s in vsa) and confidence > 60:
            recommendation.append("\n📉 VSA показывает активность продавцов. Ищите возможности для входа в шорт.")
            details.append("🔹 Крупные игроки продают - присоединяйтесь")
            details.append("🔹 Ищите точки входа на откатах к уровням сопротивления")
        
        # Детали сигналов
        details.append("\n🔍 Детали сигналов:")
        
        if vsa:
            details.append("\nVSA сигналы:")
            for signal, detail in zip(vsa, vsa_details):
                details.append(f"- {signal}")
                for d in detail.split('\n'):
                    details.append(f"  {d}")
        
        if patterns:
            details.append("\nСвечные модели:")
            for pattern, pattern_detail in zip(patterns, patterns_details):
                details.append(f"- {pattern}")
                for d in pattern_detail.split('\n'):
                    details.append(f"  {d}")
        
        return recommendation, details
    
    def plot_chart(self, df, orderbook_signals):
        """Визуализация данных с акцентом на VSA"""
        plt.figure(figsize=(14, 12))
        
        # Цена и объемы
        plt.subplot(2, 1, 1)
        
        # Восходящие свечи
        up = df[df['close'] >= df['open']]
        plt.bar(up.index, up['high']-up['low'], bottom=up['low'], color='green', width=0.8, alpha=0.7)
        plt.bar(up.index, up['close']-up['open'], bottom=up['open'], color='darkgreen', width=0.8)
        
        # Нисходящие свечи
        down = df[df['close'] < df['open']]
        plt.bar(down.index, down['high']-down['low'], bottom=down['low'], color='red', width=0.8, alpha=0.7)
        plt.bar(down.index, down['open']-down['close'], bottom=down['close'], color='darkred', width=0.8)
        
        # Подсвечиваем последнюю свечу
        last_index = df.index[-1]
        last_close = df['close'].iloc[-1]
        plt.scatter(last_index, last_close, color='gold', s=100, zorder=5, label='Текущая цена')
        
        # Добавляем уровни поддержки/сопротивления
        support = df['support'].iloc[-1]
        resistance = df['resistance'].iloc[-1]
        plt.axhline(y=support, color='green', linestyle=':', alpha=0.7, label='Поддержка')
        plt.axhline(y=resistance, color='red', linestyle=':', alpha=0.7, label='Сопротивление')
        
        plt.title(f'{SYMBOL} - VSA анализ | {datetime.now().strftime("%Y-%m-%d %H:%M")}')
        plt.ylabel('Цена')
        plt.legend()
        plt.grid()
        
        # Объем (компактный вариант)
        plt.subplot(2, 1, 2)
        
        # Фильтрация только значительных объемов
        significant_volumes = df[df['volume_ratio'] > 1.0]
        
        # Цвета для объемов
        colors = np.where(significant_volumes['close'] >= significant_volumes['open'], 'green', 'red')
        
        # Отображение только значительных объемов
        bars = plt.bar(significant_volumes.index, significant_volumes['volume_ratio'], 
                      color=colors, width=0.8, alpha=0.7)
        
        # Добавляем подписи для очень больших объемов
        for bar in bars:
            height = bar.get_height()
            if height > 2.0:
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}x',
                        ha='center', va='bottom', fontsize=8)
        
        # Линия среднего объема
        plt.axhline(y=1, color='blue', linestyle='--', linewidth=1)
        
        plt.ylabel('Объем (отношение к среднему)')
        plt.legend(['Средний объем', 'Покупки', 'Продажи'])
        plt.grid(axis='y')
        
        plt.tight_layout()
        plt.show()
    
    def run_analysis(self):
        """Запуск полного анализа"""
        print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - VSA анализ {SYMBOL} на {TIMEFRAME} таймфрейме")
        
        # Загрузка данных
        df = self.fetch_ohlcv()
        if df is None or len(df) < 50:
            print("Недостаточно данных для анализа")
            return
        
        # Загрузка стакана
        orderbook = self.fetch_orderbook()
        
        # Расчет индикаторов
        df = self.calculate_indicators(df)
        
        # Анализ состояния рынка
        self.analyze_market_state(df)
        
        # Анализ
        vsa_signals, vsa_details = self.vsa_analysis(df)
        orderbook_signals = self.analyze_orderbook(orderbook) if orderbook else []
        candle_patterns, patterns_details = self.candle_patterns(df)
        market_commentary, market_details = self.generate_market_commentary()
        recommendation, recommendation_details = self.generate_recommendation(
            vsa_signals, vsa_details, 
            orderbook_signals, 
            candle_patterns, patterns_details
        )
        
        # Вывод результатов
        print("\n=== РЫНОЧНОЕ СОСТОЯНИЕ (VSA) ===")
        for line in market_commentary:
            print(line)
        
        print("\n=== ДЕТАЛИ РЫНОЧНОГО СОСТОЯНИЯ ===")
        for line in market_details:
            print(line)
        
        print("\n=== РЕЗУЛЬТАТЫ АНАЛИЗА ===")
        
        print("\nVSA СИГНАЛЫ:")
        for signal, detail in zip(vsa_signals, vsa_details):
            print(f"- {signal}")
            for d in detail.split('\n'):
                print(f"  {d}")
        
        if orderbook_signals:
            print("\nСИГНАЛЫ СТАКАНА:")
            for signal in orderbook_signals:
                print(f"- {signal}")
        
        print("\nСВЕЧНЫЕ МОДЕЛИ:")
        for pattern, detail in zip(candle_patterns, patterns_details):
            print(f"- {pattern}")
            for d in detail.split('\n'):
                print(f"  {d}")
        
        print("\n=== ТОРГОВЫЕ РЕКОМЕНДАЦИИ (VSA) ===")
        for rec in recommendation:
            print(rec)
        
        print("\n=== ДЕТАЛИ РЕКОМЕНДАЦИЙ ===")
        for detail in recommendation_details:
            print(detail)
        
        # Визуализация
        self.plot_chart(df, orderbook_signals)

if __name__ == "__main__":
    radar = MarketRadar()
    
    # Бесконечный цикл с обновлением каждый час
    while True:
        radar.run_analysis()
        print("\nСледующее обновление через 1 час...")
        time.sleep(3600)  # Пауза 1 час