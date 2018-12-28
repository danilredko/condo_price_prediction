
import numpy as np
import pandas as pd
import re
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn import ensemble
from scipy import stats

from scipy.stats import norm
from scipy.stats import skew

import sklearn as sklearn

from numpy import genfromtxt
import matplotlib.pyplot as plt


df= pd.read_csv('condos.csv', sep='!', names=['street_no', 'street', 'unit', 'price', 'sqft', 'beds', 'baths', 'parking', 'locker'], false_values=['No'], true_values=['Yes'])

df.drop(['street_no', 'street', 'unit'], axis=1, inplace=True)

df['den'] = np.zeros(len(df['beds']))



#Conver all sqft intervals to one number

def sqft_average(sqft):

    if sqft=='0':

        return 0

    first_number, second_number = re.findall('\d+', sqft)

    return (int(first_number)+int(second_number))/2


df = df[df['beds']!='UNKNOWN']



df = df[df['price']<5000000]

#df = df[df['price']>100000]

print(df.loc[(df['beds']=='Studio&1')][:10])

df['sqft'] = df['sqft'].map(sqft_average)

#print(df.loc[df['beds']=='Studio'][:5])

df['den'] = np.where(df['beds'].str.contains('+1', regex=False) | df['beds'].str.contains('&1', regex=False) , 1.0, 0.0)

df['beds'] = np.where(df['beds'].str.contains('Studio', regex=False), '0', df['beds'])

#print(df.loc[(df['beds']=='0')][:5])

df['beds'] = np.where(df['beds'].str[-2:]=='+1', df['beds'].str[0], df['beds'])

df['beds'] = df['beds'].apply(int)

t = df['price']
t = np.array(t)

print(t.shape)

df.drop(['price'], axis=1, inplace=True)

X = df

X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.3, shuffle=True)

print(X_train.shape)
print(t_train.shape)
print(X_test.shape)
print(t_test.shape)


scaler = StandardScaler()

#scaler.fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

reg = LinearRegression()

reg.fit(X_train, t_train)

print(reg.score(X_test, t_test))

clf = ensemble.GradientBoostingRegressor(n_estimators=400, max_depth=5,min_samples_split=2, learning_rate=0.1)

clf.fit(X_train, t_train)

print(clf.score(X_test, t_test))



for i in range(10):
    clf = MLPClassifier(hidden_layer_sizes=(i+1, 4))
    clf.fit(X_train, t_train)
    print(clf.score(X_test, t_test))