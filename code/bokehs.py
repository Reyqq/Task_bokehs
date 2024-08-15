"""
Модуль bokehs предоставляет набор функций для создания и настройки финансовых графиков
с использованием библиотеки Bokeh.

Основные функции:
- create_data_sources: Подготовка данных для свечных графиков
- create_candlestick_chart: Создание базового свечного графика
- style_plot: Настройка стиля графика
"""

from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from bokeh.plotting import figure, ColumnDataSource


def customize_plot_styles(p: figure) -> None:
    """
    .. :no-index:


    Настраивает стили графика Bokeh для улучшения его внешнего вида.

    Эта функция применяет ряд стилистических изменений к объекту Figure из Bokeh,
    включая цвета фона, осей, сетки, а также настройки отображения меток и линий.

    Args:
        p (bokeh.plotting.Figure): Объект Figure из Bokeh, который нужно настроить.

    Returns:
        None: Функция изменяет переданный объект Figure и ничего не возвращает.

    Note:
        Эта функция модифицирует переданный объект Figure напрямую.
        Основные изменения включают:
        - Установку темного фона (#181c27)
        - Настройку цветов и стилей осей
        - Настройку сетки (пунктирные линии, прозрачность)
        - Удаление линий тиков
        - Автоскрытие панели инструментов
    """
    # Настройка фона
    p.background_fill_color = "#181c27"
    p.background_fill_alpha = 1
    p.border_fill_color = "#181c27"

    # Настройка оси X
    p.xaxis.major_label_orientation = "horizontal"
    p.xaxis.axis_label_text_color = "#b2b5be"
    p.xaxis.major_label_text_color = "#b2b5be"
    p.xaxis.axis_line_width = 1.0
    p.xaxis.axis_line_color = "#2a2e39"
    p.xaxis.minor_tick_line_color = None
    p.xaxis.major_tick_line_color = None  # Убрать линии тиков по оси X

    # Настройка оси Y
    p.yaxis.axis_label_text_color = "#b2b5be"
    p.yaxis.major_label_text_color = "#b2b5be"
    p.yaxis.axis_line_width = 1.0
    p.yaxis.axis_line_color = "#2a2e39"
    p.yaxis.minor_tick_line_color = None
    p.yaxis.major_tick_line_color = None  # Убрать линии тиков по оси Y

    # Настройка сетки
    p.grid.grid_line_color = "#2a2e39"
    p.outline_line_color = "#2a2e39"
    p.xgrid.grid_line_dash = [2, 2]
    p.ygrid.grid_line_dash = [2, 2]
    p.xgrid.grid_line_alpha = 0
    p.ygrid.grid_line_alpha = 0

    # Дополнительные настройки
    p.toolbar.autohide = True


def create_data_sources(df: pd.DataFrame) -> Tuple[pd.DataFrame, ColumnDataSource, ColumnDataSource, ColumnDataSource]:
    """
    .. :no-index:


    Создает источники данных для графиков свечей.

    Эта функция обрабатывает входной датафрейм, добавляя новые столбцы для визуализации свечей,
    и создает отдельные источники данных для восходящих и нисходящих свечей.

    Args:
        df (pd.DataFrame): Исходный датафрейм с данными. Должен содержать столбцы 'open', 'close'.

    Returns:
        Tuple[pd.DataFrame, ColumnDataSource, ColumnDataSource, ColumnDataSource]: Кортеж, содержащий:
            - Обработанный датафрейм с дополнительными столбцами.
            - ColumnDataSource для восходящих свечей.
            - ColumnDataSource для нисходящих свечей.
            - ColumnDataSource со всеми данными.

    Note:
        Функция создает следующие дополнительные столбцы в датафрейме:
        - 'middle': среднее значение между 'open' и 'close'.
        - 'height_inc': высота восходящей свечи.
        - 'height_dec': высота нисходящей свечи.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'open': [10, 12, 9], 'close': [11, 10, 11]})
        >>> df, inc_source, dec_source, all_source = create_data_sources(df)
    """
    df = df.copy()

    # Вычисление дополнительных данных для свечей
    df['middle'] = (df['open'] + df['close']) / 2
    df['height_inc'] = df['close'] - df['open']
    df['height_dec'] = df['open'] - df['close']

    # Разделение данных на восходящие и нисходящие свечи
    inc = df['close'] > df['open']
    dec = df['open'] > df['close']

    # Создание источников данных
    df_inc = ColumnDataSource(df[inc])
    df_dec = ColumnDataSource(df[dec])
    df_source = ColumnDataSource(df)

    return df, df_inc, df_dec, df_source


def process_positions(
        df: pd.DataFrame,
        positions_df: Optional[pd.DataFrame] = None,
        open_position_timestamps: Optional[List[int]] = None,
        close_position_timestamps: Optional[List[int]] = None,
        stop_losses: Optional[List[float]] = None,
        marker_sell_stop_color: str = 'red',
        marker_sell_color: str = 'green'
) -> Tuple[
    List[int], List[Tuple[Optional[int], str]], List[int], List[Optional[float]], List[float], List[Optional[float]]]:
    """
    .. :no-index:

    Обрабатывает позиции и создает источники данных для графиков.

    Args:
        df (pd.DataFrame): Датафрейм с данными.
        positions_df (Optional[pd.DataFrame]): Датафрейм с позициями. По умолчанию None.
        open_position_timestamps (Optional[List[int]]): Список временных меток открытия позиций. По умолчанию None.
        close_position_timestamps (Optional[List[int]]): Список временных меток закрытия позиций. По умолчанию None.
        stop_losses (Optional[List[float]]): Список стоп-лоссов. По умолчанию None.
        marker_sell_stop_color (str): Цвет маркера для позиций с достижением стоп-лосса. По умолчанию 'red'.
        marker_sell_color (str): Цвет маркера для обычных позиций. По умолчанию 'green'.

    Returns:
        Tuple[List[int], List[Tuple[Optional[int], str]], List[int], List[Optional[float]], List[float], List[Optional[float]]]:
        Кортеж, содержащий:
        - Индексы покупок
        - Кортежи с индексами продаж и цветами маркеров
        - Количество свечей в сделке
        - Процентные изменения
        - Цены открытия
        - Цены закрытия

    Raises:
        ValueError: Если не предоставлены необходимые данные о позициях.
    """

    if positions_df is None:
        if any(arg is None for arg in [open_position_timestamps, close_position_timestamps, stop_losses]):
            raise ValueError(
                "Необходимо передать либо positions_df, либо все списки: open_position_timestamps, close_position_timestamps и stop_losses")
        positions_df = pd.DataFrame({
            'open_position_timestamp': open_position_timestamps,
            'close_position_timestamp': close_position_timestamps,
            'stop_loss': stop_losses
        })
    else:
        required_columns = ['open_position_timestamp', 'close_position_timestamp', 'stop_loss']
        if not all(col in positions_df.columns for col in required_columns):
            raise ValueError(f"DataFrame с метками позиций должен содержать столбцы: {', '.join(required_columns)}")

    positions_df['open_position_timestamp'] = pd.to_datetime(positions_df['open_position_timestamp'], unit='s')
    positions_df['close_position_timestamp'] = pd.to_datetime(positions_df['close_position_timestamp'], unit='s')

    indices_buy = df.index[df['datetime'].isin(positions_df['open_position_timestamp'])].tolist()
    indices_sell = df.index[df['datetime'].isin(positions_df['close_position_timestamp'])].tolist()

    num_candles_in_trade: List[int] = []
    adjusted_sell_indices: List[Tuple[Optional[int], str]] = []
    percent_moves: List[Optional[float]] = []
    open_prices: List[float] = []
    close_prices: List[Optional[float]] = []

    for i, buy_idx in enumerate(indices_buy):
        if i < len(indices_sell):
            sell_idx = indices_sell[i]
            stop_loss = positions_df['stop_loss'].iloc[i]

            if stop_loss is not None:
                stop_loss_hit = any(df['close'].iloc[j] < stop_loss for j in range(buy_idx, sell_idx))
                if stop_loss_hit:
                    sell_idx = next(j for j in range(buy_idx, sell_idx) if df['close'].iloc[j] < stop_loss)
                    color = marker_sell_stop_color
                else:
                    color = marker_sell_color
            else:
                color = marker_sell_color

            num_candles_in_trade.append(sell_idx - buy_idx)
            percent_move = ((df['close'].iloc[sell_idx] - df['open'].iloc[buy_idx]) / df['open'].iloc[buy_idx]) * 100
            percent_moves.append(percent_move)
            adjusted_sell_indices.append((sell_idx, color))
            open_prices.append(df['open'].iloc[buy_idx])
            close_prices.append(df['close'].iloc[sell_idx])
        else:
            num_candles_in_trade.append(len(df) - buy_idx)
            adjusted_sell_indices.append((None, marker_sell_color))
            percent_moves.append(None)
            open_prices.append(df['open'].iloc[buy_idx])
            close_prices.append(None)

    return indices_buy, adjusted_sell_indices, num_candles_in_trade, percent_moves, open_prices, close_prices


def create_markers_data(
    df: pd.DataFrame,
    indices_buy: List[int],
    adjusted_sell_indices: List[Tuple[int, str]],
    num_candles_in_trade: List[int],
    percent_moves: List[float],
    open_prices: List[float],
    close_prices: List[float],
    distance_buy: float,
    distance_sell: float,
    marker_buy_color: str,
    marker_sell_color: str,
    marker_sell_stop_color: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Создает данные для маркеров покупок и продаж на графике.

    Эта функция обрабатывает данные о сделках (покупках и продажах) и возвращает словари с информацией, необходимой
    для визуализации маркеров на графике. Маркеры покупок располагаются ниже минимального значения свечи,
    а маркеры продаж - выше максимального значения свечи.

    Args:
        df (pd.DataFrame): Датафрейм с данными, включающий информацию о ценах и временных метках.
        indices_buy (List[int]): Индексы, соответствующие временным меткам открытых позиций (покупок).
        adjusted_sell_indices (List[Tuple[int, str]]): Список кортежей, содержащих индексы закрытых позиций (продаж) и цвет маркера.
        num_candles_in_trade (List[int]): Список с количеством свечей, которые были частью сделки (от покупки до продажи).
        percent_moves (List[float]): Процентное изменение цены между открытием и закрытием позиции.
        open_prices (List[float]): Цены открытия для каждой покупки.
        close_prices (List[float]): Цены закрытия для каждой продажи.
        distance_buy (float): Расстояние для позиционирования маркера покупки по оси Y.
        distance_sell (float): Расстояние для позиционирования маркера продажи по оси Y.
        marker_buy_color (str): Цвет маркера покупки.
        marker_sell_color (str): Цвет маркера продажи.
        marker_sell_stop_color (str): Цвет маркера продажи при достижении стоп-лосса.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: Два словаря с данными для маркеров покупок и продаж.
    """

    marker_data_buy = {
        'datetime': df['datetime'][indices_buy],
        'y': df['low'][indices_buy] - distance_buy,
        'labels': np.arange(1, len(indices_buy) + 1).astype(str),
        'color': [marker_buy_color] * len(indices_buy),
        'num_candles': num_candles_in_trade
    }

    marker_data_sell = {
        'datetime': [],
        'y': [],
        'labels': [],
        'color': [],
        'num_candles': [],
        'percent_move': [],
        'open_price': [],
        'close_price': []
    }

    for i, buy_idx in enumerate(indices_buy):
        sell_idx, color = adjusted_sell_indices[i]
        if sell_idx is not None:
            marker_data_sell['datetime'].append(df['datetime'].iloc[sell_idx])
            marker_data_sell['y'].append(df['high'].iloc[sell_idx] + distance_sell)
            marker_data_sell['labels'].append(str(i + 1))
            marker_data_sell['num_candles'].append(sell_idx - buy_idx)
            marker_data_sell['color'].append(color)
            marker_data_sell['percent_move'].append(percent_moves[i])
            marker_data_sell['open_price'].append(df['open'].iloc[buy_idx])
            marker_data_sell['close_price'].append(df['close'].iloc[sell_idx])

    return marker_data_buy, marker_data_sell

