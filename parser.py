
# coding: utf-8

import pandas as pd

# Здесь конвертер strip убирает лишние пробелы в конце строки. Применили только для колонки name,
# тк только ее дальше будем использовать.
def strip(text):
    try:
        return text.strip()
    except AttributeError:
        return text

df_names = pd.read_csv('G2014.TTL', sep=',', encoding='cp1251', skiprows=8,
                       names=['number', 'name', 'type', 'length'],
                       converters={'name': strip})

col_names = df_names['name']
df = pd.read_csv('2014.csv', sep=';', encoding='cp1251', nrows=20, names=col_names)
