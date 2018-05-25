import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv("dataset/train.csv")

x_train = dataset.iloc[:, 1:-1]
y_train = dataset.iloc[:, -1]

le_a = LabelEncoder()
x_train[['A', 'D', 'E', 'F', 'G', 'I', 'J', 'L', 'M']] = le_a.fit_transform(x_train[['A', 'D', 'E', 'F', 'G', 'I', 'J', 'L', 'M']])

from collections import defaultdict
d = defaultdict(LabelEncoder)

fit = x_train[['A', 'D', 'E', 'F', 'G', 'I', 'J', 'L', 'M']].apply(lambda x: d[x.name].fit_transform(x))
x_train.isnull().any()
x_train.count()

x_train = x_train.fillna(x_train.mode()[0], inplace=True)
cols = x_train[['A', 'D', 'E', 'F', 'G', 'I', 'J', 'L', 'M']]
x_train[cols]=x_train[cols].fillna(x_train.mode().iloc[0])

x_train[['M']] = le_a.fit_transform(x_train[['M']])


x_train[['B']] = x_train[['B']].fillna(x_train[['B']].mean())

# fitting the random forest classifier to the training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(x_train.values, y_train)

test = pd.read_csv("dataset/test.csv")