import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("data/important.csv")
print(data.shape)
print(data.head(10))
print(data.dtypes)
data=data.drop(['date','street','city','statezip','country'],axis=1)
print(data)
print(data.isnull().sum())
print(data['view'].value_counts())

#Handling the bedroom, bathroom and floors values to its ceiling
data['bedrooms']=np.ceil(data['bedrooms'])
data['bathrooms']=np.ceil(data['bathrooms'])
data['floors']=np.ceil(data['floors'])
print(data.ndim)

#corelation matrix
cor_matrix=data.corr()
print(cor_matrix)
print(cor_matrix['price'].sort_values())
x=data.drop(['price'],axis=1)
y=data['price']
print(y.max())

#plotting the scatter plot of the different attributes vs price
plt.figure()
fig, ax = plt.subplots(3, 4,figsize=(15,15))
fig.suptitle('scattered plot of the attributes')
ax[0,0].scatter(x['bedrooms'],y,c='tab:red',alpha=0.1,edgecolors='red')
ax[0,0].set_title('bedrooms vs price')

ax[0,1].scatter(x['bathrooms'],y,color='gold')
ax[0,1].set_title('bathrooms vs price')

ax[0,2].scatter(x['sqft_living'],y,color='silver')
ax[0,2].set_title('sqft_living vs price')

ax[0,3].scatter(x['sqft_lot'],y,color='blue')
ax[0,3].set_title('sqft_lot vs price')

ax[1,0].scatter(x['floors'],y,color='green')
ax[1,0].set_title('floors vs price')

ax[1,1].scatter(x['waterfront'],y,color='skyblue')
ax[1,1].set_title('waterfront vs price')

ax[1,2].scatter(x['view'],y,color='pink')
ax[1,2].set_title('view vs price')

ax[1,3].scatter(x['condition'],y,color='brown')
ax[1,3].set_title('condition vs price')

ax[2,0].scatter(x['sqft_above'],y,color='purple')
ax[2,0].set_title('sqft_above vs price')

ax[2,1].scatter(x['sqft_basement'],y,color='orange')
ax[2,1].set_title('sqft_basement vs price')

ax[2,2].scatter(x['yr_built'],y,color='cyan')
ax[2,2].set_title('yr_built vs price')

ax[2,3].scatter(x['yr_renovated'],y,c='tab:blue')
ax[2,3].set_title('yr_renovated vs price')

plt.show()


#scalling and the normalization of the data
from sklearn.preprocessing import scale,StandardScaler,MinMaxScaler
x=MinMaxScaler().fit_transform(x)
#x=StandardScaler().fit_transform(x)
#y=MinMaxScaler().fit_transform(y)

#spliting the data into test ,train datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=3)
print(y_train.mean())
print(y_test.mean())
print(y_train.index)
print(y_test.index)

#applying the linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

print('coefficients=',regressor.coef_,'intercept=',regressor.intercept_)

#predictions and accuracy for train and test data
y_pred_train=regressor.predict(x_train)
y_pred_test=regressor.predict(x_test)

print(y_pred_train.mean())
print(y_pred_test.mean())

sco1=regressor.score(x_train,y_train)
sco2=regressor.score(x_test,y_test)
print('train_score=',sco1)
print('test_score=',sco2)

df1 = pd.DataFrame({'Actual': y_train, 'Predicted': y_pred_train})
df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})

plt.plot(range(0,3680),y_pred_train)
plt.plot(range(0,920),y_pred_test)
plt.show()

df1 = df1.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
df2 = df2.head(25)
df2.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

#difference between the y_train and y_predicted
df1['difference']=y_train-y_pred_train
df2['difference']=y_test-y_pred_test
print('max_error in train data:',df1['difference'].max(),'min_error in train data:',df1['difference'].min())
print('max_error in test data:',df2['difference'].max(),'min_error in text data:',df2['difference'].min())

#error measurment
from sklearn.metrics import mean_absolute_error, max_error, mean_squared_error

y_mean = y.mean()
print(y_mean)
print((y_mean / 100) * 60)

# errors for train data
print('mean_absolute_error=', mean_absolute_error(y_train, y_pred_train))
print('max_error=', max_error(y_train, y_pred_train))
print('Mean Squared Error:', mean_squared_error(y_train, y_pred_train))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_train, y_pred_train)))

# errors for test data
print('mean_absolute_error=', mean_absolute_error(y_test, y_pred_test))
print('max_error=', max_error(y_test, y_pred_test))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred_test))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred_test)))

#applying decision Tree alog
from sklearn.tree import DecisionTreeRegressor
DT_regressor=DecisionTreeRegressor(max_depth=1)
DT_regressor.fit(x_train,y_train)
y_pred_train=DT_regressor.predict(x_train)
y_pred_test=DT_regressor.predict(x_test)
df1=pd.DataFrame({'real':y_train,'pred':y_pred_train})
print(df1.head(30))
df1 = df1.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
print('score_train=',DT_regressor.score(x_train,y_train))
print('score_test=',DT_regressor.score(x_test,y_test))
df2=pd.DataFrame({'real':y_test,'pred':y_pred_test})
df2.head(30)
df2 = df2.head(25)
df2.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
#difference between the y_train and y_predicted
df1['difference']=y_train-y_pred_train
df2['difference']=y_test-y_pred_test
print('max_error in train data:',df1['difference'].max(),'min_error in train data:',df1['difference'].min())
print('max_error in test data:',df2['difference'].max(),'min_error in text data:',df2['difference'].min())

#KNeighbors alog
from sklearn.neighbors import KNeighborsRegressor
NN_regressor = KNeighborsRegressor(n_neighbors=1,p=2,n_jobs=None)
NN_regressor.fit(x_train, y_train)
y_pred_train=NN_regressor.predict(x_train)
y_pred_test=NN_regressor.predict(x_test)
print('score_train=',NN_regressor.score(x_train, y_train))
print('score_test=',NN_regressor.score(x_test, y_test))
df1=pd.DataFrame({'real':y_train,'pred':y_pred_train})
df1.head(30)
df1 = df1.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
df2=pd.DataFrame({'real':y_test,'pred':y_pred_test})
df2.head(30)
df2 = df2.head(25)
df2.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
#difference between the y_train and y_predicted
df1['difference']=y_train-y_pred_train
df2['difference']=y_test-y_pred_test
print('max_error in train data:',df1['difference'].max(),'min_error in train data:',df1['difference'].min())
print('max_error in test data:',df2['difference'].max(),'min_error in text data:',df2['difference'].min())

#using SVM algo
from sklearn import svm
regr = svm.SVR(gamma='auto')
regr.fit(x_train, y_train)
regr.score(x_train, y_train)
regr.score(x_test, y_test)

# gaussian_process alog
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
kernel = DotProduct() + WhiteKernel()
GP_regressor = GaussianProcessRegressor(kernel=kernel,random_state=4).fit(x_train, y_train)
GP_regressor.score(x_train, y_train)
