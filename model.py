
import numpy as np
import pandas as pd
import re
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
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


print(df.head())
df['den'] = np.zeros(len(df['beds']))

df['floor'] = np.zeros(len(df['beds']))


#Conver all sqft intervals to one number

def sqft_average(sqft):

    if sqft=='0':

        return 0

    elif '-' in sqft:

        first_number, second_number = re.findall('\d+', sqft)
        return (int(first_number)+int(second_number))/2

    else:

        return int(sqft)


def get_floor(unit):

    if len(unit)==4:
        return int(unit[0:2])
    elif len(unit)==3 or len(unit)==1:
        return int(unit[0])




df = df[df['beds']!='UNKNOWN']
df = df[df['sqft']!=0]
#df = df[df['price']>100000]


df = df[df['unit'].str.isdecimal()]
#print(df.loc[(df['beds']=='Studio&1')][:10])

df['sqft'] = df['sqft'].map(sqft_average)
df['floor'] = df['unit'].map(get_floor)

print(df.head())


#print(df.loc[df['beds']=='Studio'][:5])

df['den'] = np.where(df['beds'].str.contains('+1', regex=False) | df['beds'].str.contains('&1', regex=False) , 1, 0)

df['beds'] = np.where(df['beds'].str.contains('Studio', regex=False), '0', df['beds'])

#print(df.loc[(df['beds']=='0')][:5])

df['beds'] = np.where(df['beds'].str[-2:]=='+1', df['beds'].str[0], df['beds'])

df['beds'] = df['beds'].apply(int)

t = df['price']
t = np.array(t)


correlation_matrix = df.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()

df.drop(['street_no', 'street', 'unit'], axis=1, inplace=True)


features = ['sqft']
target=df['price']

#X = pd.DataFrame(df['sqft'], columns=['sqft'])
X = np.concatenate((df['sqft'].values.reshape(1350,1), df['beds'].values.reshape(1350,1), df['baths'].values.reshape(1350,1)), axis=1).reshape(1350,3)
Y = df['price']



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, shuffle=True)

lin_model = LinearRegression()

lin_model.fit(X_train, Y_train)
'''
predictions = lin_model.predict(X_test)

plt.scatter(X_train, Y_train)
plt.plot(X_test, predictions, color='blue')
plt.show()
'''
print(lin_model.score(X_test, Y_test))

regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X_train, Y_train)

# Score model
print(regressor.score(X_test, Y_test))

#print(regressor.predict(np.array([1000]).reshape(1,-1)))
