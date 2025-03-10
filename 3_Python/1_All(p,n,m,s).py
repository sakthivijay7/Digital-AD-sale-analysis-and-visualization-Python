   ###  LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import label
from sympy.printing.pretty.pretty_symbology import line_width
    ### DATASET LOAD PANDAS
path="C:/Users/ELCOT/Desktop/New folder/Python area,sale/Dataset.csv"
df=pd.read_csv(path)
#print(df.head())
#print(df.shape)
#print(df.columns)
##print(df.split())
##x=pd.get_dummies(df[' '])
a=df['Id']
b=df['LotArea']
c=df['SalePrice']
A=a[0:100]
B=b[0:100]
C=c[0:100]
    ###MATPLOTLIB
#plt.plot(A,B,C,'d',ls='-',color='b',ms=10,mec='r',mfc='g')
#plt.bar(B,c,color='b',label='Id')
d=np.array(b,c)  ##NUMPY
D=np.array(B,C)
#plt.hist(d,label=a)
#plt.scatter(B,C,marker='o',color='r',label='SalePrice')
#plt.pie(B,autopct='%1.1f%%')
#plt.subplot(2,1,1)
#plt.plot(b,c,'r')
#plt.subplot(2,1,2)
#plt.plot(B,C,'b')
#plt.title('Land')
#plt.xlabel('LotArea')
#plt.ylabel('SalePrice')
#plt.legend()
#plt.grid()
       ### SEABORN
#sns.relplot(data=df,x='Id',y='LotArea',hue='SalePrice')
#sns.displot(df,x='Id',y='LotArea',hue='SalePrice')
#sns.pairplot(df)
#sns.catplot(data=df,x='Id',y='LotArea',kind='box',hue='SalePrice')
#sns.regplot(data=df,x='Id',y='LotArea');
#sns.lmplot(data=df,x='Id',y='LotArea',hue='SalePrice',legend=bool)

plt.show()
