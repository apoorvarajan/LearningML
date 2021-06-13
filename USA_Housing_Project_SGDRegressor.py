import numpy as np
import pandas as pd
df = pd.read_csv('USA_Housing.csv')
df.corr().Price.sort_values()
X=df[['Avg. Area Income','Avg. Area House Age','Area Population','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms']]
y=df.Price
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X2 = X
scaler.fit(X2)
scaled_X2 =scaler.transform(X2)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(scaled_X2,y,test_size=0.3,random_state=101)
from sklearn.linear_model import SGDRegressor
model = SGDRegressor()
model.fit(X_train,y_train)
pred = model.predict(X_test)
from sklearn import metrics
print("mean absolute error = ", metrics.mean_absolute_error(y_test,pred))
print("mean squared error = ", metrics.mean_squared_error(y_test,pred))
print("root mean squared error = ", np.sqrt(metrics.mean_squared_error(y_test,pred)))