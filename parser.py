
# coding: utf-8

# In[66]:

import pandas as pd


# In[67]:

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


# In[68]:

col_names = df_names['name']


# In[69]:

df = pd.read_csv('2014.csv', sep=';', encoding='cp1251', nrows=20, names=col_names)

