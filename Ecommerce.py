import numpy as np
import pandas as pd
df = pd.read_csv("Ecommerce Customers")
df.head()
reduced_data =  df[['Avg. Session Length','Time on App','Time on Website','Length of Membership','Yearly Amount Spent']]
X=reduced_data[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y=reduced_data['Yearly Amount Spent']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1.fit(X_train,y_train)
pred1 = model1.predict(X_test)
from sklearn import metrics
print("mean absolute error = ", metrics.mean_absolute_error(y_test,pred1))
print("mean squared error = ", metrics.mean_squared_error(y_test,pred1))
print("root mean squared error = ", np.sqrt(metrics.mean_squared_error(y_test,pred1)))
