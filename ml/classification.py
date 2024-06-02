
import matplotlib.pyplot as plt
from numpy import asarray
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from db_querries import DbClass
from sklearn import metrics

import seaborn as sns

from ml.MyLinearRegression import My_Linear_Regression 
db= DbClass()


db.connect()
data=db.getTable('knjige_filtered',columns=['id','sifra','title','autor','kategorija','izdavac','povez','godina','format','strana','opis','cena'])
db.close()



l=LabelEncoder()
data["povez"]=l.fit_transform(data["povez"])
data["kategorija"]=l.fit_transform(data["kategorija"])
data["godina"]=l.fit_transform(data["godina"])
data["izdavac"]=l.fit_transform(data["izdavac"])
data["autor"]=l.fit_transform(data["autor"])
data['format']=np.array(map(float, data['format']))


scaler= MinMaxScaler()
data['strana'] = scaler.fit_transform(data[['strana']])
data['cena'] = scaler.fit_transform(data[['cena']])
data['format'] = scaler.fit_transform(data[['format']])




x=data[['povez','format','strana','autor','kategorija','izdavac','godina']]
y=data['cena']


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,shuffle=True)


print(x_train)
print(y_train)


model2 = LinearRegression()
model2.fit(x_train, y_train)
predictions2=model2.predict(x_test)
importance2 = model2.coef_

for i,v in enumerate(importance2):
 print('Feature: %0d, Score: %.5f' % (i,v))


print(model2.score(x_test, y_test))

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions2))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions2))
print('R-squared:', metrics.r2_score(y_test, predictions2))


model=My_Linear_Regression()
model.fit_model(x_train, y_train)
predictions=model.predict(x_test)
importance = model.weights


for i,v in enumerate(importance):
 print('Feature: %0d, Score: %.5f' % (i,v))


print(model.score(x_test, y_test))

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))
print('R-squared:', metrics.r2_score(y_test, predictions))


plt.scatter(x_train['format'], y_train)  
plt.plot(x_train['format'], model.predict(x_train), color='blue', linewidth=3)
plt.show()







# xxx = []
# for i in range(len(xx)):
#     xxx.append(xx[i]*model.coef_[0] + model.intercept_)
# plt.plot(xx, xxx, color='orange', linewidth=1)
# plt.show()

# print("Coeff:", model.score(x_test.to_numpy().reshape(-1,1), y_test))
# from sklearn.metrics import r2_score
# print("Coeff2:", r2_score(y_test, model.predict(x_test.to_numpy().reshape(-1,1))))

