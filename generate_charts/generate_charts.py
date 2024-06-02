import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

#NAME='3-3 Po godinama.csv'

NAME='3-5 Po cenama.csv'

data = pd.read_csv(f'{NAME}')

print(data)
#################### Za generisanje 3-3 ##################################

# current_year = datetime.datetime.now().year
# data_chart=data.groupby(pd.cut(data['godina'], np.arange(1961,current_year +11,10),right=False),observed=False)['count'].sum()

# data_chart.to_csv('genratesd.csv')
# print(data_chart)

# plt.figure(figsize=(10, 10))
# data_chart.plot(kind='bar')
# plt.xlabel('Godine')
# plt.ylabel('Broj')
# plt.title('Broj izdatih knjiga po godinama')
# plt.xticks(rotation=45)
# plt.yticks(np.arange(0,data_chart.max(),300))  
# plt.savefig('{NAME}.png')

#################### Za generisanje 3-5 ##################################


data_chart=data.groupby(pd.cut(data['cena'], [0,501,1501,3001,5001,10001,15001,np.inf],right=False),observed=False)['count'].sum()

data_chart.to_csv('genratesd.csv')
print(data_chart)


plt.figure(figsize=(30, 20))
data_chart.plot(kind='bar')
plt.xlabel('Cene')
plt.ylabel('Broj')
plt.title('Broj knjiga za prodaju po cenama')
plt.xticks()
plt.yticks(np.arange(0,data_chart.max()+500,500))  
plt.grid(visible=True)
plt.savefig(f'{NAME}.png')

