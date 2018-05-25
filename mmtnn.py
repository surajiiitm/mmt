import pandas as pd
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv("dataset/train.csv")

x_train = dataset.iloc[:, 1:-1]
y_train = dataset.iloc[:, -1]
test = pd.read_csv("dataset/test.csv")
id = test.iloc[:, 0]
test = test.iloc[:, 1:]

le_a = LabelEncoder()
x_train[['A']] = x_train[['A']].fillna('b')
x_train[['A']] = le_a.fit_transform(x_train[['A']])
test[['A']] = test[['A']].fillna('b')
test[['A']] = le_a.transform(test[['A']])

x_train[['B']] = x_train[['B']].fillna(x_train[['B']].mean())
test[['B']] = test[['B']].fillna(x_train[['B']].mean())

le_d = LabelEncoder()
x_train[['D']] = x_train[['D']].fillna('u')
x_train[['D']] = le_d.fit_transform(x_train[['D']])
test[['D']] = test[['D']].fillna('u')
test[['D']] = le_d.transform(test[['D']])


le_e = LabelEncoder()
mode = x_train[['E']].mode()
x_train[['E']] = x_train[['E']].fillna('g')
x_train[['E']] = le_e.fit_transform(x_train[['E']])
test[['E']] = test[['E']].fillna('g')
test[['E']] = le_e.transform(test[['E']])

le_f = LabelEncoder()
x_train[['F']].mode()
x_train[['F']] = x_train[['F']].fillna('c')
x_train[['F']] = le_f.fit_transform(x_train[['F']])
test[['F']] = test[['F']].fillna('c')
test[['F']] = le_f.transform(test[['F']])


le_g = LabelEncoder()
x_train[['G']].mode()
x_train[['G']] = x_train[['G']].fillna('v')
x_train[['G']] = le_g.fit_transform(x_train[['G']])
test[['G']] = test[['G']].fillna('v')
test[['G']] = le_g.transform(test[['G']])

le_i = LabelEncoder()
x_train[['I']] = le_i.fit_transform(x_train[['I']])
test[['I']] = le_i.transform(test[['I']])


le_j = LabelEncoder()
x_train[['J']] = le_j.fit_transform(x_train[['J']])
test[['J']] = le_j.transform(test[['J']])

le_l = LabelEncoder()
x_train[['L']] = le_l.fit_transform(x_train[['L']])
test[['L']] = le_l.transform(test[['L']])


le_m = LabelEncoder()
x_train[['M']] = le_m.fit_transform(x_train[['M']])
test[['M']] = le_m.transform(test[['M']])


x_train[['N']] = x_train[['N']].fillna(x_train[['N']].mean())
test[['N']] = test[['N']].fillna(x_train[['N']].mean())


# define model
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(225, input_shape=(15,), activation='relu'))
classifier.add(Dense(15, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(x_train, y_train, batch_size=100, epochs=12)

res = classifier.predict(test.values)
res = classifier.predict_classes(test.values)
res = res.reshape(138)
    
from collections import OrderedDict
df = pd.DataFrame(OrderedDict({'id':id, 'P':res}))
    
df.to_csv('submission3.csv', index=False)
