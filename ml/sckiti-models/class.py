
import matplotlib.pyplot as plt
from numpy import asarray
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression,LogisticRegression, SGDRegressor
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from db_querries import DbClass
from sklearn import metrics
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score

from ml.MyLogisticRegressionOvsR import My_Logistic_Regression


db= DbClass()


db.connect()
dataORIG=db.getTable('knjige_filtered',columns=['id','sifra','title','autor','kategorija','izdavac','povez','godina','format','strana','opis','cena'])
db.close()


data = dataORIG.copy().drop(columns=['opis','autor','sifra','title','id'])

# label_encoder = LabelEncoder()
# data['povez'] = label_encoder.fit_transform(data['povez'])

#one-hot-encoding za kategoriju
data = pd.get_dummies(data, columns=['povez','godina','kategorija','izdavac'])

y = data['cena']
x = data.drop(columns=['cena'])


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

def classify(i):
    if i<=500:
        return 1
    if i>500  and i<=1500:
        return 2
    if i>1500 and i<=3000:
        return 3
    if i>3000 and i<=5000:
        return 4
    if i>5000 and i<=15000:
        return 5    
    if i>15000:
        return 6      
   
applyall = np.vectorize(classify)
y_test = applyall(y_test)
y_train = applyall(y_train)

model = OneVsRestClassifier(LogisticRegression(penalty='l2',max_iter=100))
model.fit(x_train_scaled,y_train)


predictions = model.predict(x_test_scaled)
predictions_train = model.predict(x_train_scaled)

print('F1-micro (Test):', metrics.f1_score(y_test, predictions,average='micro'))
print('F1-micro (Train):', metrics.f1_score(y_train, predictions_train,average='micro'))



# cv_scores = cross_val_score(model, np.concatenate((x_train_scaled, x_test_scaled), axis=0),  np.concatenate((y_train, y_test), axis=0), cv=3, scoring='f1_micro' )
# print('Cross-validation F1-micro scores:', cv_scores)
# print('Mean Cross-validation F1-micro:', np.mean(cv_scores))




plt.scatter(range(y_test.shape[0]),y_test, color='red', label='Predicted Regression Line') 

plt.scatter(range(predictions.shape[0]),predictions, color='blue', label='Predicted Regression Line') 
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Logistic Regression')
plt.show()

conf_matrix = metrics.confusion_matrix(y_test, predictions)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()