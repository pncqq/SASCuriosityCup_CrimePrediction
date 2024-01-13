import pandas as pd

#### Crime prediction

# Nazwa pliku .data
nazwa_pliku_data = 'dane.data'

# Określenie separatora (np. ',', '\t', ';')
separator = ','

# Wczytywanie danych bez nagłówków
data = pd.read_csv(nazwa_pliku_data, sep=separator, header=None)

# Zapis do pliku CSV
data.to_csv('dane.csv', index=False, header=False)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
