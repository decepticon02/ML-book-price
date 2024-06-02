
import matplotlib.pyplot as plt
from numpy import asarray
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression,LogisticRegression, SGDRegressor

from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from db_querries import DbClass
from sklearn import metrics
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score


db= DbClass()


db.connect()
dataORIG=db.getTable('knjige_filtered',columns=['id','sifra','title','autor','kategorija','izdavac','povez','godina','format','strana','opis','cena'])
db.close()


data = dataORIG.copy().drop(columns=['opis','autor','sifra','title','id'])

# label_encoder = LabelEncoder()
# data['povez'] = label_encoder.fit_transform(data['povez'])

#one-hot-encoding za kategoriju
data = pd.get_dummies(data, columns=['povez','izdavac','godina','kategorija'])

y = data['cena']
x = data.drop(columns=['cena'])


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)
y_test=scaler.fit_transform(y_test.to_numpy().reshape(-1, 1))
y_train=scaler.fit_transform(y_train.to_numpy().reshape(-1, 1))


###############GRID HYPER PARAMS OPTIMIZING#############
# param_grid = {
#     'max_iter': [7000, 10000, 12000],
#     'tol': [0.0001, 0.001, 0.00005],
#     'eta0': [0.0001, 0.0002, 0.01],
#     'learning_rate': ['optimal', 'adaptive']
# }

#model = SGDRegressor()
#grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
# grid_search.fit(x_train_scaled, y_train)

# best_model = grid_search.best_estimator_
# print('Best model params ', best_model.get_params())
# predictions = best_model.predict(x_test_scaled)
################## OPTIMIZED ######################

best_model= SGDRegressor(alpha=0.0001,epsilon=0.1,eta0=0.0001,learning_rate='adaptive',tol=0.0001,max_iter=8000,penalty='l2',verbose=1)
best_model.fit(x_train_scaled,y_train)

predictions = best_model.predict(x_test_scaled)

feature_importance = best_model.coef_
feature_names = x_train.columns


for feature_name, importance in zip(feature_names, feature_importance):
    print(f'Feature: {feature_name}, Importance: {importance}')

print('R-squared (Test):', best_model.score(x_test_scaled, y_test))
print('R-squared (Train):', best_model.score(x_train_scaled, y_train))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))

cv_scores = cross_val_score(best_model, np.concatenate((x_train_scaled, x_test_scaled), axis=0),  np.concatenate((y_train, y_test), axis=0), cv=10, scoring='r2')
print('Cross-validation R-squared scores:', cv_scores)
print('Mean Cross-validation R-squared:', np.mean(cv_scores))

print(x_test_scaled.shape,y_test.shape,predictions.shape)



plt.scatter(range(y_test.shape[0]),y_test, color='red', label='Predicted Regression Line') 

plt.scatter(range(predictions.shape[0]),predictions, color='blue', label='Predicted Regression Line') 
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Linear Regression')
plt.show()

