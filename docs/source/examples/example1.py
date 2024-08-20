import sys
import os
import pandas as pd

# Добавляем путь к папке code в системный путь Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'code')))

# Теперь можно импортировать функции из bokehs.py
from bokehs import plot_trading_signals

# Здесь вы можете использовать импортированные функции
# Пример использования функции
open_timestamps = [1577103060, 1577104980, 1577107680]
close_timestamps = [1577104020, 1577105820, None]
stop_losses = [7543, 7541, None]

# Загрузка данных
df = pd.read_parquet('/content/drive/MyDrive/Colab Notebooks/Vlados_work/draft/base_data.parquet', )

# Переименование столбцов
df.rename(columns={'open_futures': 'open',
                   'high_futures': 'high',
                   'low_futures': 'low',
                   'close_futures': 'close',
                   'first_volume_futures': 'volume',
                   'timestamp': 'datetime'},
                   inplace=True)

# Преобразование столбца 'datetime' в тип данных datetime
df['date'] = df.datetime.dt.date


df = df[:500]


# Предполагается, что `df` уже создан и содержит необходимые данные
plot_trading_signals(df, open_position_timestamps=open_timestamps, close_position_timestamps=close_timestamps, stop_losses=stop_losses)
