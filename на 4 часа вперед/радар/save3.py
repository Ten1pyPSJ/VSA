import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
from collections import deque

# Настройки
SYMBOL = 'BTC/USDT'  # Пара для анализа
TIMEFRAME = '15m'     # Часовой таймфрейм
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
        """Расчет технических индикаторов"""
        # Простые скользящие средние
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Объемные индикаторы
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Дополнительные индикаторы для VSA
        df['range'] = df['high'] - df['low']
        df['avg_range'] = df['range'].rolling(window=20).mean()
        df['close_position'] = (df['close'] - df['low']) / df['range']
        df['body_ratio'] = abs(df['close'] - df['open']) / df['range']
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['range']
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['range']
        
        # Индекс силы тренда
        df['trend_strength'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5) * 100
        
        # Волатильность
        df['volatility'] = df['range'].rolling(window=14).mean() / df['close'].rolling(window=14).mean() * 100
        
        # Уровни поддержки/сопротивления
        df['support'] = df['low'].rolling(window=5).min()
        df['resistance'] = df['high'].rolling(window=5).max()
        
        return df
    
    def analyze_market_state(self, df):
        """Анализ общего состояния рынка"""
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Определение тренда
        if last['sma_20'] > last['sma_50'] and last['close'] > last['sma_20']:
            self.market_state['trend'] = 'Сильный восходящий'
        elif last['sma_20'] > last['sma_50']:
            self.market_state['trend'] = 'Восходящий'
        elif last['sma_20'] < last['sma_50'] and last['close'] < last['sma_20']:
            self.market_state['trend'] = 'Сильный нисходящий'
        elif last['sma_20'] < last['sma_50']:
            self.market_state['trend'] = 'Нисходящий'
        else:
            self.market_state['trend'] = 'Боковой'
        
        # Определение силы тренда
        if abs(last['trend_strength']) > 5:
            self.market_state['strength'] = 'Сильный'
        elif abs(last['trend_strength']) > 2:
            self.market_state['strength'] = 'Умеренный'
        else:
            self.market_state['strength'] = 'Слабый'
        
        # Определение фазы рынка
        if last['rsi'] > 70 and last['close'] > last['sma_50']:
            self.market_state['phase'] = 'Перекупленность в тренде'
        elif last['rsi'] < 30 and last['close'] < last['sma_50']:
            self.market_state['phase'] = 'Перепроданность в тренде'
        elif last['rsi'] > 70:
            self.market_state['phase'] = 'Перекупленность'
        elif last['rsi'] < 30:
            self.market_state['phase'] = 'Перепроданность'
        else:
            self.market_state['phase'] = 'Нейтральная'
        
        # Определение активности
        if last['volume_ratio'] > 2:
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
        if (self.market_state['trend'] in ['Сильный восходящий', 'Восходящий'] and 
            self.market_state['phase'] == 'Перекупленность'):
            self.market_state['context'] = 'Коррекция после роста'
        elif (self.market_state['trend'] in ['Сильный нисходящий', 'Нисходящий'] and 
              self.market_state['phase'] == 'Перепроданность'):
            self.market_state['context'] = 'Отскок после падения'
        elif self.market_state['trend'] == 'Боковой':
            self.market_state['context'] = 'Консолидация'
        else:
            self.market_state['context'] = 'Трендовое движение'
    
    def analyze_orderbook(self, orderbook):
        """Анализ стакана ордеров"""
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
        """Расширенный анализ VSA (Volume Spread Analysis)"""
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
            last_candle['body_ratio'] > 0.5):
            signals.append(f"📈 Сильный спрос (объем x{last_candle['volume_ratio']:.1f}, крупные покупатели активны)")
            details.append("🔹 Цена закрылась в верхней части диапазона с большим объемом")
            details.append("🔹 Широкая свеча с маленькой верхней тенью")
            details.append("🔹 Покупатели контролируют ситуацию")
        
        # 2. Слабость на подъеме (возможный разворот)
        elif (last_candle['close'] > last_candle['open'] and 
              last_candle['volume_ratio'] < 0.7 and 
              last_candle['range'] > last_candle['avg_range'] and
              last_candle['body_ratio'] < 0.3):
            signals.append(f"⚠️ Слабость на подъеме (объем x{last_candle['volume_ratio']:.1f}, покупатели выдыхаются)")
            details.append("🔹 Цена выросла, но объем ниже среднего")
            details.append("🔹 Большой диапазон с маленьким телом")
            details.append("🔹 Покупатели не подтверждают движение")
        
        # 3. Сильное предложение (нисходящий тренд)
        elif (last_candle['close'] < last_candle['open'] and 
              last_candle['volume_ratio'] > 1.5 and 
              last_candle['range'] > last_candle['avg_range'] and
              last_candle['close_position'] < 0.3 and
              last_candle['body_ratio'] > 0.5):
            signals.append(f"📉 Сильное предложение (объем x{last_candle['volume_ratio']:.1f}, крупные продавцы активны)")
            details.append("🔹 Цена закрылась в нижней части диапазона с большим объемом")
            details.append("🔹 Широкая медвежья свеча с маленькой нижней тенью")
            details.append("🔹 Продавцы контролируют ситуацию")
        
        # 4. Слабость на падении (возможный разворот)
        elif (last_candle['close'] < last_candle['open'] and 
              last_candle['volume_ratio'] < 0.7 and 
              last_candle['range'] > last_candle['avg_range'] and
              last_candle['body_ratio'] < 0.3):
            signals.append(f"⚠️ Слабость на падении (объем x{last_candle['volume_ratio']:.1f}, продавцы выдыхаются)")
            details.append("🔹 Цена упала, но объем ниже среднего")
            details.append("🔹 Большой диапазон с маленьким телом")
            details.append("🔹 Продавцы не подтверждают движение")
        
        # 5. Тест на предложение
        elif (last_candle['close'] < prev_candle['close'] and
              last_candle['volume'] > last_candle['volume_sma_20'] * 1.2 and
              last_candle['close'] > (last_candle['high'] + last_candle['low']) / 2 and
              last_candle['range'] > last_candle['avg_range']):
            signals.append("🔍 Тест на предложение - крупные игроки проверяют уровень")
            details.append("🔹 Большой объем при тестировании уровня")
            details.append("🔹 Цена закрылась выше середины диапазона")
            details.append("🔹 Возможна остановка нисходящего движения")
        
        # 6. Тест на спрос
        elif (last_candle['close'] > prev_candle['close'] and
              last_candle['volume'] > last_candle['volume_sma_20'] * 1.2 and
              last_candle['close'] < (last_candle['high'] + last_candle['low']) / 2 and
              last_candle['range'] > last_candle['avg_range']):
            signals.append("🔍 Тест на спрос - крупные игроки проверяют уровень")
            details.append("🔹 Большой объем при тестировании уровня")
            details.append("🔹 Цена закрылась ниже середины диапазона")
            details.append("🔹 Возможна остановка восходящего движения")
        
        # 7. Абсорбция покупателей
        elif (last_candle['close'] < last_candle['open'] and
              last_candle['volume'] > last_candle['volume_sma_20'] * 2 and
              last_candle['close'] == last_candle['low'] and
              last_candle['body_ratio'] > 0.7):
            signals.append("🛑 Абсорбция покупателей - сильное давление продавцов")
            details.append("🔹 Очень большой объем на нисходящей свече")
            details.append("🔹 Цена закрылась на минимуме")
            details.append("🔹 Продавцы поглотили всех покупателей")
        
        # 8. Абсорбция продавцов
        elif (last_candle['close'] > last_candle['open'] and
              last_candle['volume'] > last_candle['volume_sma_20'] * 2 and
              last_candle['close'] == last_candle['high'] and
              last_candle['body_ratio'] > 0.7):
            signals.append("🛑 Абсорбция продавцов - сильное давление покупателей")
            details.append("🔹 Очень большой объем на восходящей свече")
            details.append("🔹 Цена закрылась на максимуме")
            details.append("🔹 Покупатели поглотили всех продавцов")
        
        # 9. Накопление
        elif (last_candle['range'] < last_candle['avg_range'] * 0.7 and
              last_candle['volume'] > last_candle['volume_sma_20'] * 1.5 and
              abs(last_candle['close'] - last_candle['open']) / last_candle['range'] < 0.3):
            signals.append("🔄 Накопление - крупные игроки набирают позиции")
            details.append("🔹 Маленький диапазон с большим объемом")
            details.append("🔹 Небольшое тело свечи (доджи или маленькое тело)")
            details.append("🔹 Крупные игроки аккумулируют позиции")
        
        # 10. Распределение
        elif (last_candle['range'] < last_candle['avg_range'] * 0.7 and
              last_candle['volume'] > last_candle['volume_sma_20'] * 1.5 and
              abs(last_candle['close'] - last_candle['open']) / last_candle['range'] < 0.3 and
              prev_candle['volume'] > prev_candle['volume_sma_20'] * 1.5):
            signals.append("🔄 Распределение - крупные игроки закрывают позиции")
            details.append("🔹 Маленький диапазон с большим объемом")
            details.append("🔹 Небольшое тело свечи (доджи или маленькое тело)")
            details.append("🔹 Крупные игроки распродают позиции")
        
        # 11. Усиление спроса
        elif (last_candle['close'] > last_candle['open'] and
              last_candle['volume_ratio'] > 1.2 and
              last_candle['volume_ratio'] > prev_candle['volume_ratio'] and
              last_candle['close'] > prev_candle['close']):
            signals.append("📈 Усиление спроса - объемы растут вместе с ценой")
            details.append("🔹 Рост цены сопровождается увеличением объема")
            details.append("🔹 Последовательное усиление покупателей")
            details.append("🔹 Подтверждение восходящего движения")
        
        # 12. Усиление предложения
        elif (last_candle['close'] < last_candle['open'] and
              last_candle['volume_ratio'] > 1.2 and
              last_candle['volume_ratio'] > prev_candle['volume_ratio'] and
              last_candle['close'] < prev_candle['close']):
            signals.append("📉 Усиление предложения - объемы растут вместе с падением цены")
            details.append("🔹 Падение цены сопровождается увеличением объема")
            details.append("🔹 Последовательное усиление продавцов")
            details.append("🔹 Подтверждение нисходящего движения")
        
        return signals, details
    
    def candle_patterns(self, df):
        """Анализ свечных моделей"""
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        prev_prev_candle = df.iloc[-3]
        
        patterns = []
        details = []
        
        # Молот
        if (last_candle['close'] > last_candle['open'] and 
            (last_candle['close'] - last_candle['open']) / (last_candle['high'] - last_candle['low']) < 0.3 and 
            last_candle['low'] == min(last_candle['low'], prev_candle['low'], prev_prev_candle['low'])):
            patterns.append("🔨 Молот - возможен разворот вверх")
            details.append("🔹 Маленькое тело в верхней части диапазона")
            details.append("🔹 Длинная нижняя тень (минимум 2-3 раза больше тела)")
            details.append("🔹 Появился после нисходящего движения")
        
        # Повешенный
        elif (last_candle['close'] < last_candle['open'] and 
              (last_candle['open'] - last_candle['close']) / (last_candle['high'] - last_candle['low']) < 0.3 and 
              last_candle['high'] == max(last_candle['high'], prev_candle['high'], prev_prev_candle['high'])):
            patterns.append("🪢 Повешенный - возможен разворот вниз")
            details.append("🔹 Маленькое тело в нижней части диапазона")
            details.append("🔹 Длинная верхняя тень (минимум 2-3 раза больше тела)")
            details.append("🔹 Появился после восходящего движения")
        
        # Поглощение
        if (last_candle['close'] > prev_candle['open'] and 
            last_candle['open'] < prev_candle['close'] and 
            last_candle['close'] > prev_candle['close'] and 
            last_candle['open'] < prev_candle['open']):
            patterns.append("🟢 Бычье поглощение - сильный сигнал на покупку")
            details.append("🔹 Текущая свеча полностью 'поглотила' предыдущую")
            details.append("🔹 Закрытие выше открытия предыдущей свечи")
            details.append("🔹 Лучше работает с увеличенным объемом")
        elif (last_candle['close'] < prev_candle['open'] and 
              last_candle['open'] > prev_candle['close'] and 
              last_candle['close'] < prev_candle['close'] and 
              last_candle['open'] > prev_candle['open']):
            patterns.append("🔴 Медвежье поглощение - сильный сигнал на продажу")
            details.append("🔹 Текущая свеча полностью 'поглотила' предыдущую")
            details.append("🔹 Закрытие ниже открытия предыдущей свечи")
            details.append("🔹 Лучше работает с увеличенным объемом")
        
        # Утренняя звезда
        if (prev_candle['close'] < prev_candle['open'] and
            last_candle['close'] > last_candle['open'] and
            last_candle['open'] < prev_candle['close'] and
            prev_prev_candle['close'] < prev_prev_candle['open'] and
            prev_candle['body_ratio'] > 0.7 and
            last_candle['body_ratio'] > 0.5):
            patterns.append("🌟 Утренняя звезда - сильный бычий разворот")
            details.append("🔹 После нисходящей свечи идет свеча с маленьким телом")
            details.append("🔹 Завершается сильной восходящей свечой")
            details.append("🔹 Лучше работает с подтверждением объема")
        
        # Вечерняя звезда
        elif (prev_candle['close'] > prev_candle['open'] and
              last_candle['close'] < last_candle['open'] and
              last_candle['open'] > prev_candle['close'] and
              prev_prev_candle['close'] > prev_prev_candle['open'] and
              prev_candle['body_ratio'] > 0.7 and
              last_candle['body_ratio'] > 0.5):
            patterns.append("🌙 Вечерняя звезда - сильный медвежий разворот")
            details.append("🔹 После восходящей свечи идет свеча с маленьким телом")
            details.append("🔹 Завершается сильной нисходящей свечой")
            details.append("🔹 Лучше работает с подтверждением объема")
        
        return patterns, details
    
    def trend_analysis(self, df):
        """Анализ тренда"""
        last_close = df.iloc[-1]['close']
        sma_20 = df.iloc[-1]['sma_20']
        sma_50 = df.iloc[-1]['sma_50']
        
        trend = []
        details = []
        
        # Определение тренда
        if sma_20 > sma_50 and last_close > sma_20:
            trend.append("📈 Сильный восходящий тренд")
            details.append("🔹 Цена выше обеих скользящих средних")
            details.append("🔹 Короткая MA (20) выше длинной MA (50)")
            details.append("🔹 Явное восходящее движение")
        elif sma_20 > sma_50 and last_close > sma_50:
            trend.append("📈 Восходящий тренд")
            details.append("🔹 Цена выше длинной скользящей средней (50)")
            details.append("🔹 Короткая MA (20) выше длинной MA (50)")
            details.append("🔹 Общее восходящее направление")
        elif sma_20 < sma_50 and last_close < sma_20:
            trend.append("📉 Сильный нисходящий тренд")
            details.append("🔹 Цена ниже обеих скользящих средних")
            details.append("🔹 Короткая MA (20) ниже длинной MA (50)")
            details.append("🔹 Явное нисходящее движение")
        elif sma_20 < sma_50 and last_close < sma_50:
            trend.append("📉 Нисходящий тренд")
            details.append("🔹 Цена ниже длинной скользящей средней (50)")
            details.append("🔹 Короткая MA (20) ниже длинной MA (50)")
            details.append("🔹 Общее нисходящее направление")
        else:
            trend.append("↔️ Боковик или неопределенный тренд")
            details.append("🔹 Цена колеблется вокруг скользящих средних")
            details.append("🔹 Нет четкого направления")
            details.append("🔹 Возможна консолидация")
        
        # Пересечение SMA
        if df['sma_20'].iloc[-2] < df['sma_50'].iloc[-2] and df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1]:
            trend.append("🌟 Золотое пересечение - возможен разворот вверх")
            details.append("🔹 Короткая MA (20) пересекла длинную MA (50) снизу вверх")
            details.append("🔹 Классический бычий сигнал")
            details.append("🔹 Требует подтверждения объемом")
        elif df['sma_20'].iloc[-2] > df['sma_50'].iloc[-2] and df['sma_20'].iloc[-1] < df['sma_50'].iloc[-1]:
            trend.append("💀 Мертвое пересечение - возможен разворот вниз")
            details.append("🔹 Короткая MA (20) пересекла длинную MA (50) сверху вниз")
            details.append("🔹 Классический медвежий сигнал")
            details.append("🔹 Требует подтверждения объемом")
        
        return trend, details
    
    def rsi_analysis(self, df):
        """Анализ RSI"""
        last_rsi = df.iloc[-1]['rsi']
        
        rsi_signal = []
        details = []
        
        if last_rsi > 70:
            rsi_signal.append("🔴 RSI > 70 - перекупленность, возможен откат")
            details.append("🔹 Индикатор показывает перекупленность")
            details.append("🔹 Возможна коррекция или консолидация")
            details.append("🔹 В тренде может долго оставаться в зоне перекупленности")
        elif last_rsi < 30:
            rsi_signal.append("🟢 RSI < 30 - перепроданность, возможен откат")
            details.append("🔹 Индикатор показывает перепроданность")
            details.append("🔹 Возможен отскок или консолидация")
            details.append("🔹 В тренде может долго оставаться в зоне перепроданности")
        elif last_rsi > 50:
            rsi_signal.append("🟢 RSI > 50 - бычий импульс")
            details.append("🔹 Индикатор выше средней линии")
            details.append("🔹 Преобладает бычья динамика")
            details.append("🔹 Восходящий импульс")
        else:
            rsi_signal.append("🔴 RSI < 50 - медвежий импульс")
            details.append("🔹 Индикатор ниже средней линии")
            details.append("🔹 Преобладает медвежья динамика")
            details.append("🔹 Нисходящий импульс")
        
        # Дивергенции
        if len(df) > 14:
            prices = df['close'].tail(14).values
            rsis = df['rsi'].tail(14).values
            
            # Бычья дивергенция
            if (prices[-1] < prices[-3] < prices[-5] and 
                rsis[-1] > rsis[-3] > rsis[-5]):
                rsi_signal.append("🟢 Бычья дивергенция RSI - возможен разворот вверх")
                details.append("🔹 Цена делает более низкие минимумы")
                details.append("🔹 RSI делает более высокие минимумы")
                details.append("🔹 Медвежий импульс ослабевает")
            
            # Медвежья дивергенция
            elif (prices[-1] > prices[-3] > prices[-5] and 
                  rsis[-1] < rsis[-3] < rsis[-5]):
                rsi_signal.append("🔴 Медвежья дивергенция RSI - возможен разворот вниз")
                details.append("🔹 Цена делает более высокие максимумы")
                details.append("🔹 RSI делает более низкие максимумы")
                details.append("🔹 Бычий импульс ослабевает")
        
        return rsi_signal, details
    
    def generate_market_commentary(self):
        """Генерация комментария о состоянии рынка"""
        commentary = []
        details = []
        
        # Общее состояние
        commentary.append(f"\n📌 Текущее состояние рынка:")
        commentary.append(f"📊 Тренд: {self.market_state['trend']}")
        commentary.append(f"💪 Сила: {self.market_state['strength']}")
        commentary.append(f"🔄 Фаза: {self.market_state['phase']}")
        commentary.append(f"🏃 Активность: {self.market_state['activity']}")
        commentary.append(f"🌪️ Волатильность: {self.market_state['volatility']}")
        commentary.append(f"📌 Контекст: {self.market_state['context']}")
        
        # Детали состояния
        details.append("\n🔍 Детали текущего состояния:")
        
        if self.market_state['trend'] in ['Сильный восходящий', 'Восходящий']:
            details.append("🔹 Рынок находится в восходящем тренде")
            if self.market_state['strength'] == 'Сильный':
                details.append("🔹 Тренд имеет сильную динамику")
            elif self.market_state['strength'] == 'Умеренный':
                details.append("🔹 Тренд имеет умеренную силу")
            else:
                details.append("🔹 Тренд слабый, возможен разворот")
        elif self.market_state['trend'] in ['Сильный нисходящий', 'Нисходящий']:
            details.append("🔹 Рынок находится в нисходящем тренде")
            if self.market_state['strength'] == 'Сильный':
                details.append("🔹 Тренд имеет сильную динамику")
            elif self.market_state['strength'] == 'Умеренный':
                details.append("🔹 Тренд имеет умеренную силу")
            else:
                details.append("🔹 Тренд слабый, возможен разворот")
        else:
            details.append("🔹 Рынок находится в боковом движении")
            details.append("🔹 Нет четкого направления")
            details.append("🔹 Ищите пробои уровней с подтверждением")
        
        # Интерпретация
        if self.market_state['trend'] in ['Сильный восходящий', 'Восходящий']:
            if self.market_state['phase'] == 'Перекупленность':
                commentary.append("\nℹ️ Внимание: рынок в восходящем тренде, но достиг перекупленности. "
                                "Возможна коррекция или консолидация перед продолжением роста.")
                details.append("🔹 Перекупленность в тренде - возможна пауза")
                details.append("🔹 Ищите подтверждение продолжения тренда")
            else:
                commentary.append("\nℹ️ Рынок находится в восходящем тренде. "
                                "Рассматривайте покупки на откатах при подтверждении спроса.")
                details.append("🔹 Тренд здоровый, ищите точки входа")
                details.append("🔹 Покупайте на откатах к поддержкам")
        elif self.market_state['trend'] in ['Сильный нисходящий', 'Нисходящий']:
            if self.market_state['phase'] == 'Перепроданность':
                commentary.append("\nℹ️ Внимание: рынок в нисходящем тренде, но достиг перепроданности. "
                                "Возможен отскок или консолидация перед продолжением падения.")
                details.append("🔹 Перепроданность в тренде - возможна пауза")
                details.append("🔹 Ищите подтверждение продолжения тренда")
            else:
                commentary.append("\nℹ️ Рынок находится в нисходящем тренде. "
                                "Рассматривайте продажи на откатах при подтверждении предложения.")
                details.append("🔹 Тренд здоровый, ищите точки входа")
                details.append("🔹 Продавайте на откатах к сопротивлениям")
        else:
            commentary.append("\nℹ️ Рынок находится в боковом движении. "
                            "Ищите пробои уровней с объемом для определения направления.")
            details.append("🔹 Торгуйте от границ диапазона")
            details.append("🔹 Ищите пробои с подтверждением объема")
        
        # Рекомендации по активности
        if self.market_state['activity'] == 'Высокая активность':
            commentary.append("\n💡 Высокая торговая активность указывает на сильное участие крупных игроков. "
                            "Следите за VSA-сигналами для определения их намерений.")
            details.append("🔹 Большие объемы - активность профессионалов")
            details.append("🔹 Анализируйте VSA для понимания их действий")
        elif self.market_state['activity'] == 'Повышенная активность':
            commentary.append("\n💡 Повышенная активность может указывать на накопление или распределение. "
                            "Анализируйте контекст для понимания ситуации.")
            details.append("🔹 Объемы выше среднего - возможны манипуляции")
            details.append("🔹 Определите, кто контролирует рынок")
        else:
            commentary.append("\n💡 Низкая активность может означать отсутствие интереса крупных игроков. "
                            "Будьте осторожны с входами - возможны ложные движения.")
            details.append("🔹 Маленькие объемы - слабое участие")
            details.append("🔹 Возможны резкие движения на новостях")
        
        # Рекомендации по волатильности
        if self.market_state['volatility'] == 'Высокая':
            commentary.append("\n⚠️ Высокая волатильность: рынок может совершать резкие движения. "
                            "Увеличьте стоп-лоссы и уменьшите размер позиции.")
            details.append("🔹 Большие дневные диапазоны")
            details.append("🔹 Будьте готовы к резким движениям")
        elif self.market_state['volatility'] == 'Умеренная':
            commentary.append("\nℹ️ Умеренная волатильность: рынок движется предсказуемо. "
                            "Ищите торговые возможности в направлении тренда.")
            details.append("🔹 Стабильные движения")
            details.append("🔹 Хорошие условия для торговли")
        else:
            commentary.append("\nℹ️ Низкая волатильность: рынок находится в спокойном состоянии. "
                            "Рассмотрите стратегии торговли в диапазоне.")
            details.append("🔹 Маленькие дневные диапазоны")
            details.append("🔹 Возможен резкий пробой")
        
        return commentary, details
    
    def generate_recommendation(self, vsa, vsa_details, orderbook, patterns, patterns_details, trend, trend_details, rsi, rsi_details):
        """Генерация рекомендации на основе анализа"""
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
        
        # Анализ тренда
        for t in trend:
            if "восходящий" in t.lower():
                bullish_signals += 1.5
            elif "нисходящий" in t.lower():
                bearish_signals += 1.5
            elif "золотое" in t.lower():
                bullish_signals += 2
            elif "мертвое" in t.lower():
                bearish_signals += 2
        
        # Анализ RSI
        for r in rsi:
            if "перепроданность" in r.lower() or "быч" in r.lower():
                bullish_signals += 1.5
            elif "перекупленность" in r.lower() or "медвеж" in r.lower():
                bearish_signals += 1.5
        
        # Учет рыночного состояния
        if self.market_state['trend'] in ['Сильный восходящий', 'Восходящий']:
            bullish_signals += 1.5
        elif self.market_state['trend'] in ['Сильный нисходящий', 'Нисходящий']:
            bearish_signals += 1.5
            
        if self.market_state['strength'] == 'Сильный':
            if self.market_state['trend'] in ['Сильный восходящий', 'Восходящий']:
                bullish_signals += 1
            else:
                bearish_signals += 1
        
        # Формирование рекомендации
        signal_diff = bullish_signals - bearish_signals
        
        if signal_diff > 6:
            recommendation.append("🟢🟢🟢 ОЧЕНЬ СИЛЬНЫЙ СИГНАЛ НА ПОКУПКУ")
            confidence = min(95, 75 + signal_diff * 3)
        elif signal_diff > 4:
            recommendation.append("🟢🟢 СИЛЬНЫЙ СИГНАЛ НА ПОКУПКУ")
            confidence = min(85, 65 + signal_diff * 4)
        elif signal_diff > 2:
            recommendation.append("🟢 Умеренный сигнал на покупку")
            confidence = 60 + signal_diff * 5
        elif signal_diff > 0:
            recommendation.append("🟢 Слабый сигнал на покупку")
            confidence = 55 + signal_diff * 5
        elif signal_diff < -6:
            recommendation.append("🔴🔴🔴 ОЧЕНЬ СИЛЬНЫЙ СИГНАЛ НА ПРОДАЖУ")
            confidence = min(95, 75 + abs(signal_diff) * 3)
        elif signal_diff < -4:
            recommendation.append("🔴🔴 СИЛЬНЫЙ СИГНАЛ НА ПРОДАЖУ")
            confidence = min(85, 65 + abs(signal_diff) * 4)
        elif signal_diff < -2:
            recommendation.append("🔴 Умеренный сигнал на продажу")
            confidence = 60 + abs(signal_diff) * 5
        elif signal_diff < 0:
            recommendation.append("🔴 Слабый сигнал на продажу")
            confidence = 55 + abs(signal_diff) * 5
        else:
            recommendation.append("⚪ Нет четкого сигнала")
            confidence = 50
        
        recommendation.append(f"📊 Уверенность: {confidence}%")
        
        # Конкретные рекомендации
        if confidence > 80:
            recommendation.append("\n💡 РЕКОМЕНДАЦИЯ: Сильный сигнал. Рассмотрите вход в сделку с нормальным размером позиции. "
                                "Используйте стоп-лосс для управления риском.")
            details.append("🔹 Вход по текущей цене или на откате")
            details.append("🔹 Используйте нормальный размер позиции")
        elif confidence > 70:
            recommendation.append("\n💡 Рекомендация: Хороший сигнал. Возможен вход в сделку с умеренным размером позиции. "
                                "Дождитесь частичного подтверждения.")
            details.append("🔹 Вход на подтверждении сигнала")
            details.append("🔹 Умеренный размер позиции")
        elif confidence > 60:
            recommendation.append("\n💡 Рекомендация: Умеренный сигнал. Вход с уменьшенным размером позиции. "
                                "Требуется дополнительное подтверждение.")
            details.append("🔹 Ожидайте дополнительных подтверждений")
            details.append("🔹 Уменьшенный размер позиции")
        elif confidence > 50:
            recommendation.append("\n💡 Рекомендация: Слабый сигнал. Вход с минимальным размером позиции или ожидание. "
                                "Необходимо четкое подтверждение.")
            details.append("🔹 Ожидайте четких подтверждений")
            details.append("🔹 Минимальный размер позиции или ожидание")
        else:
            recommendation.append("\n💡 Рекомендация: Оставайтесь вне рынка. "
                                "Ожидайте более четких сигналов.")
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
        
        recommendation.append("- Подтверждайте сигналы на нескольких таймфреймах")
        details.append("🔹 Проверяйте сигналы на старших таймфреймах")
        
        # Дополнительные рекомендации по VSA
        if any("спрос" in s.lower() for s in vsa) and confidence > 60:
            recommendation.append("\n📈 VSA показывает активность покупателей. Ищите возможности для входа в лонг.")
            details.append("🔹 Крупные игроки покупают - присоединяйтесь")
            details.append("🔹 Ищите точки входа на откатах")
        elif any("предложение" in s.lower() for s in vsa) and confidence > 60:
            recommendation.append("\n📉 VSA показывает активность продавцов. Ищите возможности для входа в шорт.")
            details.append("🔹 Крупные игроки продают - присоединяйтесь")
            details.append("🔹 Ищите точки входа на откатах")
        
        # Детали сигналов
        details.append("\n🔍 Детали сигналов:")
        
        if vsa:
            details.append("\nVSA сигналы:")
            for signal, detail in zip(vsa, vsa_details):
                details.append(f"- {signal}")
                details.append(f"  {detail}")
        
        if patterns:
            details.append("\nСвечные модели:")
            for pattern, pattern_detail in zip(patterns, patterns_details):
                details.append(f"- {pattern}")
                details.append(f"  {pattern_detail}")
        
        if trend:
            details.append("\nАнализ тренда:")
            for t, trend_detail in zip(trend, trend_details):
                details.append(f"- {t}")
                details.append(f"  {trend_detail}")
        
        if rsi:
            details.append("\nАнализ RSI:")
            for r, rsi_detail in zip(rsi, rsi_details):
                details.append(f"- {r}")
                details.append(f"  {rsi_detail}")
        
        return recommendation, details
    
    def plot_chart(self, df, orderbook_signals):
        """Визуализация данных"""
        plt.figure(figsize=(14, 12))
        
        # Цена
        plt.subplot(3, 1, 1)
        plt.plot(df.index, df['close'], label='Цена', color='blue', linewidth=2)
        plt.plot(df.index, df['sma_20'], label='SMA 20', color='orange', linestyle='--', linewidth=1.5)
        plt.plot(df.index, df['sma_50'], label='SMA 50', color='red', linestyle='--', linewidth=1.5)
        
        # Подсвечиваем последнюю свечу
        last_index = df.index[-1]
        last_close = df['close'].iloc[-1]
        plt.scatter(last_index, last_close, color='gold', s=100, zorder=5, label='Текущая цена')
        
        # Добавляем уровни поддержки/сопротивления
        support = df['support'].iloc[-1]
        resistance = df['resistance'].iloc[-1]
        plt.axhline(y=support, color='green', linestyle=':', alpha=0.5, label='Поддержка')
        plt.axhline(y=resistance, color='red', linestyle=':', alpha=0.5, label='Сопротивление')
        
        plt.title(f'{SYMBOL} - Часовой график | {datetime.now().strftime("%Y-%m-%d %H:%M")}')
        plt.ylabel('Цена')
        plt.legend()
        plt.grid()
        
        # Объем (компактный вариант)
        plt.subplot(3, 1, 2)
        
        # Фильтрация только значительных объемов
        significant_volumes = df[df['volume_ratio'] > 1.2]
        
        # Цвета для объемов
        colors = np.where(significant_volumes['close'] >= significant_volumes['open'], 'green', 'red')
        
        # Отображение только значительных объемов
        bars = plt.bar(significant_volumes.index, significant_volumes['volume_ratio'], 
                      color=colors, width=0.8, alpha=0.7)
        
        # Добавляем подписи для очень больших объемов
        for bar in bars:
            height = bar.get_height()
            if height > 2.5:
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}x',
                        ha='center', va='bottom', fontsize=8)
        
        # Линия среднего объема
        plt.axhline(y=1, color='blue', linestyle='--', linewidth=1)
        
        plt.ylabel('Объем (отношение к среднему)')
        plt.legend(['Средний объем', 'Покупки', 'Продажи'])
        plt.grid(axis='y')
        
        # Стакан ордеров и рыночное состояние
        plt.subplot(3, 1, 3)
        
        # Рыночное состояние
        state_text = (f"Тренд: {self.market_state['trend']}\n"
                     f"Сила: {self.market_state['strength']}\n"
                     f"Фаза: {self.market_state['phase']}\n"
                     f"Активность: {self.market_state['activity']}\n"
                     f"Волатильность: {self.market_state['volatility']}\n"
                     f"Контекст: {self.market_state['context']}")
        
        if orderbook_signals:
            orderbook_text = "СИГНАЛЫ СТАКАНА:\n" + "\n".join(orderbook_signals)
        else:
            orderbook_text = "Нет значительных сигналов в стакане"
        
        plt.text(0.02, 0.8, state_text, 
                ha='left', va='center', fontsize=10,
                bbox=dict(facecolor='lightblue', alpha=0.5))
        
        plt.text(0.5, 0.8, orderbook_text, 
                ha='center', va='center', fontsize=10,
                bbox=dict(facecolor='lightyellow', alpha=0.5))
        
        plt.axis('off')
        plt.title('Рыночное состояние и анализ стакана')
        
        plt.tight_layout()
        plt.show()
    
    def run_analysis(self):
        """Запуск полного анализа"""
        print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Анализ {SYMBOL} на {TIMEFRAME} таймфрейме")
        
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
        trend, trend_details = self.trend_analysis(df)
        rsi, rsi_details = self.rsi_analysis(df)
        market_commentary, market_details = self.generate_market_commentary()
        recommendation, recommendation_details = self.generate_recommendation(
            vsa_signals, vsa_details, 
            orderbook_signals, 
            candle_patterns, patterns_details, 
            trend, trend_details, 
            rsi, rsi_details
        )
        
        # Вывод результатов
        print("\n=== РЫНОЧНОЕ СОСТОЯНИЕ ===")
        for line in market_commentary:
            print(line)
        
        print("\n=== ДЕТАЛИ РЫНОЧНОГО СОСТОЯНИЯ ===")
        for line in market_details:
            print(line)
        
        print("\n=== РЕЗУЛЬТАТЫ АНАЛИЗА ===")
        
        print("\nVSA СИГНАЛЫ:")
        for signal, detail in zip(vsa_signals, vsa_details):
            print(f"- {signal}")
            print(f"  {detail}")
        
        if orderbook_signals:
            print("\nСИГНАЛЫ СТАКАНА:")
            for signal in orderbook_signals:
                print(f"- {signal}")
        
        print("\nСВЕЧНЫЕ МОДЕЛИ:")
        for pattern, detail in zip(candle_patterns, patterns_details):
            print(f"- {pattern}")
            print(f"  {detail}")
        
        print("\nТРЕНД:")
        for t, detail in zip(trend, trend_details):
            print(f"- {t}")
            print(f"  {detail}")
        
        print("\nRSI:")
        for r, detail in zip(rsi, rsi_details):
            print(f"- {r}")
            print(f"  {detail}")
        
        print("\n=== ТОРГОВЫЕ РЕКОМЕНДАЦИИ ===")
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