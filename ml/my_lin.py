
import pickle
import matplotlib.pyplot as plt
from numpy import asarray
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression,LogisticRegression, SGDRegressor

from sklearn.preprocessing import LabelEncoder,MinMaxScaler, OneHotEncoder,StandardScaler
from db_querries import DbClass
from sklearn import metrics
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score

from ml.MyLinearRegression import My_Linear_Regression


db= DbClass()


db.connect()
dataORIG=db.getTable('knjige_filtered',columns=['id','sifra','title','autor','kategorija','izdavac','povez','godina','format','strana','opis','cena'])
db.close()


data = dataORIG.copy().drop(columns=['opis','autor','sifra','title','id'])


#one-hot-encoding za kategoriju

data = pd.get_dummies(data, columns=['povez','godina','kategorija','izdavac']) # jos i format strana 

y = data['cena']
x = data.drop(columns=['cena'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)



best_model= My_Linear_Regression(alpha=0.0001,epsilon=0.1,eta0=0.1,tol=0.0001,max_iter=2000)

best_model.fit(x_train_scaled,y_train.to_numpy().reshape(-1, 1))

predictions = best_model.predict(x_test_scaled)

feature_importance = best_model.weights
feature_names = x_train.columns


for feature_name, importance in zip(feature_names, feature_importance):
    print(f'Feature: {feature_name}, Importance: {importance}')


predictions_train = best_model.predict(x_train_scaled)
print('R-squared (Test):', metrics.r2_score(y_test,predictions))
print('R-squared (Train):', metrics.r2_score(y_train,predictions_train))


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))


model_pkl_file = "linear_reg_model.pkl"  

with open(model_pkl_file, 'wb') as file:  
   pickle.dump((best_model, scaler,feature_names), file)



plt.scatter(range(y_test.shape[0]),y_test, color='red', label='Predicted Regression Line') 

plt.scatter(range(predictions.shape[0]),predictions, color='blue', label='Predicted Regression Line') 
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Linear Regression')
plt.show()

best_model.plotLoss()

