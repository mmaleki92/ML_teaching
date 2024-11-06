import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Dataset
ds = pd.read_csv('Data.csv')
X = ds.iloc[:, :-1].values
y = ds.iloc[:, 3].values

# Missing data
imputer = SimpleImputer()
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

# Categorical data
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# The new OneHotEncoder automatically detects categorical features
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
