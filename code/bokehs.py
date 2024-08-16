"""
Модуль bokehs предоставляет набор функций для создания и настройки финансовых графиков
с использованием библиотеки Bokeh.

Основные функции:
- create_data_sources: Подготовка данных для свечных графиков
- create_candlestick_chart: Создание базового свечного графика
- style_plot: Настройка стиля графика
"""

from typing import Optional, List, Tuple, Dict, Any, Union

import numpy as np
import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import Span, LabelSet, HoverTool, Slider, DatePicker, Toggle, CrosshairTool, FreehandDrawTool
from bokeh.models.callbacks import CustomJS
from bokeh.models.formatters import DatetimeTickFormatter
from bokeh.plotting import figure, show, ColumnDataSource


def customize_plot_styles(p: figure) -> None:
    """


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


def plot_trading_signals(
        df: pd.DataFrame,
        positions_df: Optional[pd.DataFrame] = None,
        open_position_timestamps: Optional[List[int]] = None,
        close_position_timestamps: Optional[List[int]] = None,
        stop_losses: Optional[List[float]] = None,
        distance_buy: int = 5,
        distance_sell: int = 5,
        marker_buy_color: str = 'blue',
        marker_sell_color: str = 'green',
        marker_sell_stop_color: str = 'red',
        buy_marker: str = 'triangle',
        sell_marker: str = 'inverted_triangle',
        chart_width: int = 1000,
        chart_height: int = 400,
        profit_candle_color: str = '#0072B2',
        loss_candle_color: str = 'white',
        marker_size: int = 10,
        labels_color: str = 'white',
        candle_spacing: str = '35sec',
        width_slider_start: int = 200,
        width_slider_end: int = 1400,
        width_slider_step: int = 10,
        width_slider_value: int = 400,
        height_slider_value: int = 300,
        height_slider_start: int = 200,
        height_slider_end: int = 600,
        height_slider_step: int = 10,
        tools: str = 'wheel_zoom,pan,box_zoom,reset,save,undo',
        color_segment: str = '#787b86',
        y_axis_location: str = 'right',
        toolbar_location: str = 'above',
        line_width_curr: float = 0.5,
        line_color_curr: str = '#787b86',
        line_dash_curr: Union[str, List[int]] = 'dashed',
        horizontal_line_dash: Union[str, List[int]] = [4, 4],
        horizontal_line_color: str = '#2a2e39',
        horizontal_line_width: float = 1,
        show_horizontal_line: bool = True,
        title: Optional[str] = None,
        start_date_picker_title: str = 'Начальная дата',
        end_date_picker_title: str = 'Конечная дата',
        legend_location: str = 'top_right',
        legend_title: Optional[str] = None,
        legend_title_text_font_style: str = 'italic',
        legend_title_text_font_size: str = '10px',
        legend_label_text_color: str = "#b2b5be",
        legend_label_text_font: str = 'Helvetica',
        legend_label_text_font_style: str = 'normal',
        legend_label_text_font_size: str = '12px',
        legend_background_fill_color: str = '#181c27',
        legend_border_line_color: str = '#2a2e39',
        legend_border_line_width: float = 1,
        legend_border_line_alpha: float = 0.5,
        legend_background_fill_alpha: float = 0.5,
        title_text_color: str = '#b2b5be',
        marker_legend_label_buy: str = 'buy_signal',
        marker_legend_label_sell: str = 'sell_signal',
) -> figure:
    """
    Создает интерактивный график с торговыми сигналами на основе данных свечей и информации о позициях.

    Args:
        df (pd.DataFrame): DataFrame с данными свечей. Должен содержать столбцы 'datetime', 'open', 'high', 'low', 'close'.
        positions_df (Optional[pd.DataFrame]): DataFrame с информацией о позициях. Должен содержать столбцы 'open_position_timestamp', 'close_position_timestamp', 'stop_loss'. По умолчанию None.
        open_position_timestamps (Optional[List[int]]): Список временных меток открытия позиций. По умолчанию None.
        close_position_timestamps (Optional[List[int]]): Список временных меток закрытия позиций. По умолчанию None.
        stop_losses (Optional[List[float]]): Список стоп-лоссов для каждой сделки. По умолчанию None.
        distance_buy (int): Дистанция от свечи для маркеров покупок. По умолчанию 5.
        distance_sell (int): Дистанция от свечи для маркеров продаж. По умолчанию 5.
        marker_buy_color (str): Цвет маркеров покупок. По умолчанию 'blue'.
        marker_sell_color (str): Цвет маркеров продаж. По умолчанию 'green'.
        marker_sell_stop_color (str): Цвет маркеров продаж при срабатывании стоп-лосса. По умолчанию 'red'.
        buy_marker (str): Форма маркеров покупок. По умолчанию 'triangle'.
        sell_marker (str): Форма маркеров продаж. По умолчанию 'inverted_triangle'.
        chart_width (int): Ширина графика. По умолчанию 1000.
        chart_height (int): Высота графика. По умолчанию 400.
        profit_candle_color (str): Цвет свечей при возрастании цены. По умолчанию '#0072B2'.
        loss_candle_color (str): Цвет свечей при убытке цены. По умолчанию 'white'.
        marker_size (int): Размер маркеров. По умолчанию 10.
        labels_color (str): Цвет меток. По умолчанию 'white'.
        candle_spacing (str): Расстояние между свечами. По умолчанию '35sec'.
        width_slider_start (int): Начальное значение ширины графика в виджете Slider. По умолчанию 200.
        width_slider_end (int): Конечное значение ширины графика в виджете Slider. По умолчанию 1400.
        width_slider_step (int): Шаг изменения ширины графика в виджете Slider. По умолчанию 10.
        width_slider_value (int): Значение по умолчанию для ширины графика в виджете Slider. По умолчанию 400.
        height_slider_value (int): Значение по умолчанию для высоты графика в виджете Slider. По умолчанию 300.
        height_slider_start (int): Начальное значение высоты графика в виджете Slider. По умолчанию 200.
        height_slider_end (int): Конечное значение высоты графика в виджете Slider. По умолчанию 600.
        height_slider_step (int): Шаг изменения высоты графика в виджете Slider. По умолчанию 10.
        tools (str): Набор инструментов для графика. По умолчанию "wheel_zoom,pan,box_zoom,reset,save,undo".
        color_segment (str): Цвет усиков свечей. По умолчанию '#787b86'.
        y_axis_location (str): Расположение оси y. По умолчанию 'right'.
        toolbar_location (str): Расположение набора инструментов. По умолчанию 'above'.
        line_width_curr (float): Ширина указательной линии. По умолчанию 0.5.
        line_color_curr (str): Цвет указательной линии. По умолчанию '#787b86'.
        line_dash_curr (Union[str, List[int]]): Стиль линии. По умолчанию 'dashed'.
        horizontal_line_dash (Union[str, List[int]]): Стиль горизонтальной линии. По умолчанию [4, 4].
        horizontal_line_color (str): Цвет горизонтальной линии. По умолчанию '#2a2e39'.
        horizontal_line_width (float): Ширина горизонтальной линии. По умолчанию 1.
        show_horizontal_line (bool): Показывать ли горизонтальную линию. По умолчанию True.
        title (Optional[str]): Заголовок графика. По умолчанию None.
        start_date_picker_title (str): Название кнопки для выбора начальной даты. По умолчанию 'Начальная дата'.
        end_date_picker_title (str): Название кнопки для выбора конечной даты. По умолчанию 'Конечная дата'.
        legend_location (str): Расположение легенды. По умолчанию 'top_right'.
        legend_title (Optional[str]): Заголовок легенды. По умолчанию None.
        legend_title_text_font_style (str): Стиль шрифта заголовка легенды. По умолчанию "italic".
        legend_title_text_font_size (str): Размер шрифта заголовка легенды. По умолчанию "10px".
        legend_label_text_color (str): Цвет меток легенды. По умолчанию "#b2b5be".
        legend_label_text_font (str): Шрифт меток легенды. По умолчанию "Helvetica".
        legend_label_text_font_style (str): Стиль шрифта меток легенды. По умолчанию "normal".
        legend_label_text_font_size (str): Размер шрифта меток легенды. По умолчанию "12px".
        legend_background_fill_color (str): Цвет фона легенды. По умолчанию '#181c27'.
        legend_border_line_color (str): Цвет границы легенды. По умолчанию '#2a2e39'.
        legend_border_line_width (float): Ширина границы легенды. По умолчанию 1.
        legend_border_line_alpha (float): Прозрачность границы легенды. По умолчанию 0.5.
        legend_background_fill_alpha (float): Прозрачность фона легенды. По умолчанию 0.5.
        title_text_color (str): Цвет заголовка. По умолчанию '#b2b5be'.
        marker_legend_label_buy (str): Название маркера покупки в легенде. По умолчанию 'buy_signal'.
        marker_legend_label_sell (str): Название маркера продажи в легенде. По умолчанию 'sell_signal'.

    Returns:
        figure: Объект Bokeh figure с интерактивным графиком торговых сигналов.

    Raises:
        ValueError: Если не предоставлены необходимые данные о позициях (positions_df или open_position_timestamps, close_position_timestamps, stop_losses).
        KeyError: Если в df отсутствуют необходимые столбцы ('datetime', 'open', 'high', 'low', 'close').

    Note:
        Эта функция создает сложный интерактивный график с использованием библиотеки Bokeh.
        График включает свечной график, маркеры сигналов покупки и продажи, а также различные
        настройки внешнего вида и интерактивности. Пользователь может настроить практически
        все аспекты графика, включая цвета, размеры, стили и расположение элементов.
    """
    # Проверка типов данных
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Параметр 'df' должен быть pandas DataFrame.")
    if positions_df is not None and not isinstance(positions_df, pd.DataFrame):
        raise TypeError("Параметр 'positions_df' должен быть pandas DataFrame.")
    if positions_df is None and (
            open_position_timestamps is None or close_position_timestamps is None or stop_losses is None):
        raise ValueError(
            "Необходимо передать либо positions_df, либо списки open_position_timestamps, close_position_timestamps и stop_losses")

    # Создаем источники данных
    df, df_inc, df_dec, df_source = create_data_sources(df)

    # Обрабатываем позиции
    indices_buy, adjusted_sell_indices, num_candles_in_trade, percent_moves, open_prices, close_prices = process_positions(
        df, positions_df, open_position_timestamps, close_position_timestamps, stop_losses)

    # Создаем данные для маркеров
    marker_data_buy, marker_data_sell = create_markers_data(
        df, indices_buy, adjusted_sell_indices, num_candles_in_trade, percent_moves, open_prices, close_prices,
        distance_buy, distance_sell, marker_buy_color, marker_sell_color, marker_sell_stop_color)

    # Создаем график
    width1 = Span(dimension="width", line_dash=line_dash_curr, line_width=line_width_curr, line_color=line_color_curr)
    height1 = Span(dimension="height", line_dash=line_dash_curr, line_width=line_width_curr, line_color=line_color_curr)

    p = figure(width=chart_width,
               height=chart_height,
               tools=[tools, CrosshairTool(overlay=[width1, height1])],
               y_axis_location=y_axis_location,
               toolbar_location=toolbar_location,
               active_scroll='wheel_zoom',
               active_drag='pan',
               x_axis_type="datetime",
               toolbar_sticky=False,
               title=title,
               )

    customize_plot_styles(p)

    # Создаем мультилинию, используя источник данных
    source = ColumnDataSource(data=dict(xs=[], ys=[]))
    multi_line = p.multi_line('xs', 'ys', source=source)

    tool = FreehandDrawTool(renderers=[multi_line])
    p.add_tools(tool)

    # Добавление горизонтальной линии
    average_price = df['close'].mean()
    horizontal_line = Span(location=average_price, dimension='width', line_dash=horizontal_line_dash,
                           line_color=horizontal_line_color, line_width=horizontal_line_width)
    p.add_layout(horizontal_line)

    # Добавление свечей и сегментов
    p.segment(x0='datetime', y0='high', x1='datetime', y1='low', source=df_source, color=color_segment)
    p.rect(x='datetime', y='middle', width=pd.Timedelta(candle_spacing), height='height_inc', source=df_inc,
           fill_color=profit_candle_color, line_color=profit_candle_color)
    p.rect(x='datetime', y='middle', width=pd.Timedelta(candle_spacing), height='height_dec', source=df_dec,
           fill_color=loss_candle_color, line_color=loss_candle_color)

    # Добавление маркеров на график
    marker_source_buy = ColumnDataSource(data=marker_data_buy)
    marker_source_sell = ColumnDataSource(data=marker_data_sell)
    buy_glyph = p.scatter(x='datetime', y='y', source=marker_source_buy, size=marker_size, color='color',
                          marker=buy_marker,
                          legend_label=marker_legend_label_buy)
    sell_glyph = p.scatter(x='datetime', y='y', source=marker_source_sell, size=marker_size, color='color',
                           marker=sell_marker,
                           legend_label=marker_legend_label_sell)

    # Добавление меток для маркеров
    labels_buy = LabelSet(x='datetime', y='y', text='labels', level='glyph', x_offset=0, y_offset=0,
                          source=marker_source_buy, text_color=labels_color, text_font_size='5pt', text_align='center',
                          text_baseline='middle')
    labels_sell = LabelSet(x='datetime', y='y', text='labels', level='glyph', x_offset=0, y_offset=0,
                           source=marker_source_sell, text_color=labels_color, text_font_size='5pt',
                           text_align='center',
                           text_baseline='middle')
    p.add_layout(labels_buy)
    p.add_layout(labels_sell)

    # Добавление HoverTool для маркеров
    hover_buy = HoverTool(renderers=[buy_glyph], tooltips=[("Кол-во свечей в сделке", "@num_candles")])
    hover_sell = HoverTool(renderers=[sell_glyph],
                           tooltips=[
                               ("Дата и время", "@datetime{%F %T}"),
                               ("Кол-во свечей в сделке", "@num_candles"),
                               ("Процент движения", "@percent_move{0.00} %"),
                               ("Цена открытия", "@open_price{0,0}"),
                               ("Цена закрытия", "@close_price{0,0}")
                           ],
                           formatters={'@datetime': 'datetime', '@{adj close}': 'printf'})
    p.add_tools(hover_buy)
    p.add_tools(hover_sell)

    # Настройка формата даты на оси x
    p.xaxis.formatter = DatetimeTickFormatter(
        hours="%H:%M",
        minutes="%H:%M",
        days="%d %b",
        months="%b %Y"
    )

    hover = HoverTool(mode='vline', renderers=[p.renderers[1]])
    hover.tooltips = [
        ("datetime", "@datetime{%F %T}"),
        ("open", "@open{0,0}"),
        ("high", "@high{0,0}"),
        ("low", "@low{0,0}"),
        ("close", "@close{0,0}"),
        ("volume", "@volume{0.00 a} m"),
        ("index", "@index")
    ]
    hover.formatters = {'@datetime': 'datetime', '@{adj close}': 'printf'}
    hover.mode = "mouse"
    p.add_tools(hover)

    # Создание виджетов и добавление их к графику
    width_slider = Slider(start=width_slider_start, end=width_slider_end, step=width_slider_step,
                          value=width_slider_value)
    width_slider.js_link('value', p, 'width')

    height_slider = Slider(start=height_slider_start, end=height_slider_end, step=height_slider_step,
                           value=height_slider_value)
    height_slider.js_link('value', p, 'height')

    start_date_str = df['datetime'].iloc[0].strftime('%Y-%m-%d')
    end_date_str = df['datetime'].iloc[-1].strftime('%Y-%m-%d')

    start_date_picker = DatePicker(title=start_date_picker_title, value=start_date_str)
    end_date_picker = DatePicker(title=end_date_picker_title, value=end_date_str)

    callback = CustomJS(args=dict(p=p, start_date_picker=start_date_picker, end_date_picker=end_date_picker), code="""
        var start_date = new Date(start_date_picker.value).getTime();
        var end_date = new Date(end_date_picker.value).getTime();
        p.x_range.start = start_date;
        p.x_range.end = end_date;
    """)

    start_date_picker.js_on_change('value', callback)
    end_date_picker.js_on_change('value', callback)

    toggle1 = Toggle(label="horizontal_line", button_type="success", active=True)
    toggle1.js_link('active', horizontal_line, 'visible')

    toggle_buy = Toggle(label="markers_buy", button_type="success", active=True)
    toggle_buy.js_link('active', buy_glyph, 'visible')
    toggle_buy.js_link('active', labels_buy, 'visible')

    toggle_sell = Toggle(label="markers_sell", button_type="success", active=True)
    toggle_sell.js_link('active', sell_glyph, 'visible')
    toggle_sell.js_link('active', labels_sell, 'visible')

    p.legend.location = legend_location
    p.legend.title = legend_title
    p.legend.title_text_font_style = legend_title_text_font_style
    p.legend.title_text_font_size = legend_title_text_font_size
    p.legend.label_text_color = legend_label_text_color
    p.legend.label_text_font = legend_label_text_font
    p.legend.label_text_font_style = legend_label_text_font_style
    p.legend.label_text_font_size = legend_label_text_font_size
    p.legend.background_fill_color = legend_background_fill_color
    p.legend.background_fill_alpha = legend_background_fill_alpha
    p.legend.border_line_width = legend_border_line_width
    p.legend.border_line_color = legend_border_line_color
    p.legend.border_line_alpha = legend_border_line_alpha

    if title is not None:
        p.title.text_color = title_text_color

    layout = column(row(start_date_picker, end_date_picker), width_slider, height_slider,
                    row(toggle1, toggle_buy, toggle_sell), p)
    show(layout)
