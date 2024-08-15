.. _functions_overview:

=============
Обзор функций
=============


В этом разделе представлен краткий обзор основных функций, используемых в проекте.

- :func:`customize_plot_styles(p: Figure) <bokehs.customize_plot_styles>`: Настраивает стили графиков, такие как линии, цвета и легенды.
- :func:`create_data_sources(df: pd.DataFrame) <bokehs.create_data_sources>`: Создает источники данных для графиков на основе предоставленного DataFrame.
- :func:`process_positions(df: pd.DataFrame, positions_df: Optional[pd.DataFrame] = None, ...) <bokehs.process_positions>`: Обрабатывает и визуализирует позиции на графике, включая покупку и продажу, а также добавляет маркеры.
- :func:`create_markers_data(df: pd.DataFrame,indices_buy: List[int], ...) <bokehs.create_markers_data>`: Создает данные для маркеров покупок и продаж на графике.


Подробное описание каждой функции вы найдете в разделе ":ref:`api_reference/index:Справочник API`".
