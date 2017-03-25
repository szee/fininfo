import pandas as pd


path=('D:/code/slobby/data-20151027t000000-structure-20131231t000000.csv')
path2=('G2014.TTL')
df2=pd.read_csv(path2,encoding='windows-1251',skiprows=7,index_col=False,header=1)
names=df2.columns[1]
print(names)
df=pd.read_csv(path,nrows=20,encoding='windows-1251',sep=';',header=-1,index_col=False)
df.columns=df2[names]
df.columns=[x.strip(' ') for x in df.columns]
print(df.head(5))    
