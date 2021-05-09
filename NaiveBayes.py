import pandas as pd
import numpy as np

df =  pd.read_table('iris.csv', sep=',', header=None, names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'])

n_setosa= df['class'][df['class']=='Iris-setosa'].count()
n_versicolor= df['class'][df['class']=='Iris-versicolor'].count()
n_virginica= df['class'][df['class']=='Iris-virginica'].count()

n_total= df['class'].count()

means= df.groupby('class').mean()

print(means)