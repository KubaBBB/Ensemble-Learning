from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def divide_into_train_test(df):
    X = df.drop(['id', 'date', 'price', 'type'], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = df['type'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y)
    return X_train, y_train, X_test, y_test


df = pd.read_csv('./data_with_classes.csv')
x_train, y_train, x_test, y_test = divide_into_train_test(df)

model = LinearRegression(n_jobs=-1)
model.fit(x_train, y_train.values.ravel())
y_predicted = model.predict(x_test)
y_predicted_classes = [round(elem) for elem in y_predicted]
print("Lin reg")
print(accuracy_score(y_test, y_predicted_classes))


model = LogisticRegression(n_jobs=-1, solver='lbfgs', multi_class='multinomial')
model.fit(x_train, y_train.values.ravel())
y_predicted = model.predict(x_test)
y_predicted_classes = [round(elem) for elem in y_predicted]
print("Log reg")
print(accuracy_score(y_test, y_predicted_classes))


model = DecisionTreeClassifier()
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)
print("Dec tree")
print(accuracy_score(y_test, y_predicted))

model = MLPClassifier(hidden_layer_sizes=(50))
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)
y_predicted_classes = [round(elem) for elem in y_predicted]
print("MLP")
print(accuracy_score(y_test, y_predicted_classes))

import random
c = list(zip(x_train, y_train))

random.shuffle(c)
import numpy as np
x_train, y_train = zip(*c)
x_train = np.array(x_train)
y_train = np.array(y_train)


model = LinearRegression(n_jobs=-1)
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)
y_predicted_classes = [round(elem) for elem in y_predicted]
print("Lin reg")
print(accuracy_score(y_test, y_predicted_classes))


model = LogisticRegression(n_jobs=-1, verbose=0, solver='lbfgs', multi_class='multinomial')
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)
y_predicted_classes = [round(elem) for elem in y_predicted]
print("Log reg")
print(accuracy_score(y_test, y_predicted_classes))


model = DecisionTreeClassifier()
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)
print("Dec tree")
print(accuracy_score(y_test, y_predicted))

model = MLPClassifier(hidden_layer_sizes=(50))
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)
y_predicted_classes = [round(elem) for elem in y_predicted]
print("MLP")
print(accuracy_score(y_test, y_predicted_classes))
