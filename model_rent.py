
import numpy as np
import pandas as pd
import re
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn import ensemble
from scipy import stats

from scipy.stats import norm
from scipy.stats import skew

import sklearn as sklearn

from numpy import genfromtxt
import matplotlib.pyplot as plt


df= pd.read_csv('condos_rent.csv', sep='!', names=['street_no', 'street', 'unit', 'price', 'sqft', 'beds', 'baths', 'parking', 'locker'], false_values=['No'], true_values=['Yes'])



df['den'] = np.zeros(len(df['beds']))



#Conver all sqft intervals to one number

def sqft_average(sqft):

    if sqft=='0':

        return 0

    first_number, second_number = re.findall('\d+', sqft)

    return (int(first_number)+int(second_number))/2


df = df[df['beds']!='UNKNOWN']
df = df[df['sqft']!=0]
df = df[df['price']>1000]



print(df.loc[(df['beds']=='Studio&1')][:10])

df['sqft'] = df['sqft'].map(sqft_average)

#print(df.loc[df['beds']=='Studio'][:5])

df['den'] = np.where(df['beds'].str.contains('+1', regex=False) | df['beds'].str.contains('&1', regex=False) , 1, 0)

df['beds'] = np.where(df['beds'].str.contains('Studio', regex=False), '0', df['beds'])

#print(df.loc[(df['beds']=='0')][:5])

df['beds'] = np.where(df['beds'].str[-2:]=='+1', df['beds'].str[0], df['beds'])

df['beds'] = df['beds'].apply(int)

t = df['price']
t = np.array(t)

df['parking'] = np.where(df['parking']==True, 1, 0)
df['locker'] = np.where(df['locker']==True, 1, 0)


correlation_matrix = df.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()

df.drop(['street_no', 'street', 'unit'], axis=1, inplace=True)

features = ['sqft']
target=df['price']

X = pd.DataFrame(df['sqft'], columns = ['sqft'])
Y = df['price']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, shuffle=True, random_state=60)

print(X_train.loc[0])
lin_model = LinearRegression()

lin_model.fit(X_train, Y_train)

print(lin_model.score(X_test, Y_test))

predictions = lin_model.predict(np.array([560]).reshape(1, -1))

print(predictions)