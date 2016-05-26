import neuro
import parse
import pandas as pd
import numpy as np

rows = 5 # Количество примеров на вход
columns = 3 # Количество чисел в примере
out_markers = [1, 2, 3, 5, 10, 30, 60] # На каких значениях цеплять нейроны

RTS = parse.Instrument('SPFB.RTS_150101_160101.csv',rows + columns + max(out_markers) + 10)
#print (RTS.size)
df = np.array(RTS.dataframe).T[6]
b = np.empty((rows, columns))
c = np.empty((rows, len(out_markers)))
#print(RTS.size - 12, 10)
#print("bshape", b.shape)
#print("df", df.T[3][0:10])
#b.resize([RTS.size-12,10])
for i in range(rows):
	b[i] = df[i:i+columns]/df[i+columns-1]-1
	c[i] = df[i+columns+np.array(out_markers)-1]/df[i+columns-1]-1
print(b)
print(b.shape)
print(c)
print(c.shape)

#a = np.array(RTS.dataframe["<CLOSE>"][3:5])
#print (a / a[len(a)-1] - 1)
n = neuro.Perceptron([columns,columns * 2,len(out_markers)])












#print(1)

#if __name__ == '__main__': print(2)