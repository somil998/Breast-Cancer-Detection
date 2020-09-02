import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

from google.colab import files
upload = files.upload()
df = pd.read_csv('data.csv')
df.head(7)

#count rows and coloumns
df.shape

#count no of empty row and  coloumn
df.isna().sum()

#drop column with missing value
df = df.dropna(axis=1)

df.shape

df['diagnosis'].value_counts()

#visualize the count
sns.countplot(df['diagnosis'],label='count')


df.dtypes

#Encode the categorical  data value
from sklearn.preprocessing import LabelEncoder
labelencoder_Y=LabelEncoder()
df.iloc[:,1] = labelencoder_Y.fit_transform(df.iloc[:,1].values)
#df.iloc[:,1]

#create a pair plot
sns.pairplot(df.iloc[:,1:5], hue='diagnosis')

#next 5 rows
df.head(5)

df.iloc[:,1:12].corr()

#visualize the correlation
sns.heatmap(df.iloc[:,1:12].corr())