# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# CODE
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

df=pd.read_csv('/content/titanic_dataset.csv')

df.head()

df.isnull().sum()

df.drop('Cabin',axis=1,inplace=True)

df.drop('Name',axis=1,inplace=True)

df.drop('Ticket',axis=1,inplace=True)

df.drop('PassengerId',axis=1,inplace=True)

df.drop('Parch',axis=1,inplace=True)

df

df['Age']=df['Age'].fillna(df['Age'].median())

df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

df.isnull().sum()

plt.title("Dataset with outliers")

df.boxplot()

plt.show()

cols = ['Age','SibSp','Fare']

Q1 = df[cols].quantile(0.25)

Q3 = df[cols].quantile(0.75)

IQR = Q3 - Q1

df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

plt.title("Dataset after removing outliers")

df.boxplot()

plt.show()

from sklearn.preprocessing import OrdinalEncoder

climate = ['C','S','Q']

en= OrdinalEncoder(categories = [climate])

df['Embarked']=en.fit_transform(df[["Embarked"]])

df

climate = ['male','female']

en= OrdinalEncoder(categories = [climate])

df['Sex']=en.fit_transform(df[["Sex"]])

df

from sklearn.preprocessing import RobustScaler

sc=RobustScaler()

df=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])

df

import statsmodels.api as sm

import numpy as np

import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)

df1=pd.DataFrame()

df1["Survived"]=np.sqrt(df["Survived"])

df1["Pclass"],parameters=stats.yeojohnson(df["Pclass"])

df1["Sex"]=np.sqrt(df["Sex"])

df1["Age"]=df["Age"]

df1["SibSp"],parameters=stats.yeojohnson(df["SibSp"])

df1["Fare"],parameters=stats.yeojohnson(df["Fare"])

df1["Embarked"]=df["Embarked"]

df1.skew()

import matplotlib

import seaborn as sns

import statsmodels.api as sm

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

X = df1.drop("Survived",1)

y = df1["Survived"]

plt.figure(figsize=(12,10))

cor = df1.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)

plt.show()

cor_target = abs(cor["Survived"])

relevant_features = cor_target[cor_target>0.5]

relevant_features

X_1 = sm.add_constant(X)

model = sm.OLS(y,X_1).fit()

model.pvalues

cols = list(X.columns)

pmax = 1

while (len(cols)>0):

p= []

X_1 = X[cols]

X_1 = sm.add_constant(X_1)

model = sm.OLS(y,X_1).fit()

p = pd.Series(model.pvalues.values[1:],index = cols)  

pmax = max(p)

feature_with_p_max = p.idxmax()

if(pmax>0.05):

    cols.remove(feature_with_p_max)
    
else:

    break
selected_features_BE = cols

print(selected_features_BE)

model = LinearRegression()

rfe = RFE(model,step= 4)

X_rfe = rfe.fit_transform(X,y)

model.fit(X_rfe,y)

print(rfe.support_)

print(rfe.ranking_)

nof_list=np.arange(1,6)

high_score=0

nof=0

score_list =[]

for n in range(len(nof_list)):

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)

model = LinearRegression()

rfe = RFE(model,step=nof_list[n])

X_train_rfe = rfe.fit_transform(X_train,y_train)

X_test_rfe = rfe.transform(X_test)

model.fit(X_train_rfe,y_train)

score = model.score(X_test_rfe,y_test)

score_list.append(score)

if(score>high_score):

    high_score = score
    
    nof = nof_list[n]
print("Optimum number of features: %d" %nof)

print("Score with %d features: %f" % (nof, high_score))

cols = list(X.columns)

model = LinearRegression()

rfe = RFE(model, step=2)

X_rfe = rfe.fit_transform(X,y)

model.fit(X_rfe,y)

temp = pd.Series(rfe.support_,index = cols)

selected_features_rfe = temp[temp==True].index

print(selected_features_rfe)

reg = LassoCV()

reg.fit(X, y)

print("Best alpha using built-in LassoCV: %f" % reg.alpha_)

print("Best score using built-in LassoCV: %f" %reg.score(X,y))

coef = pd.Series(reg.coef_, index = X.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)) + " variables")

imp_coef = coef.sort_values()

import matplotlib

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Feature importance using Lasso Model")

plt.show()
# OUPUT
![image](https://github.com/ATHDY005/Ex-07-Feature-Selection/assets/84709944/a3e2ccd6-200b-4c26-a3e2-18a87c1f644b)
![image](https://github.com/ATHDY005/Ex-07-Feature-Selection/assets/84709944/19237c63-10e5-4e9f-b040-5c4edd9bc5c9)
![image](https://github.com/ATHDY005/Ex-07-Feature-Selection/assets/84709944/d1b5c6c2-dfc1-4e47-bf0c-dae611567aa5)
![image](https://github.com/ATHDY005/Ex-07-Feature-Selection/assets/84709944/af44f0a6-1661-465b-9014-f9e7a5f3cfb5)
![image](https://github.com/ATHDY005/Ex-07-Feature-Selection/assets/84709944/f1856504-fa3d-4e52-bb9a-f626ca500ae5)
![image](https://github.com/ATHDY005/Ex-07-Feature-Selection/assets/84709944/ce2ca5c3-d03a-4376-b8ce-012c1bc5f887)
![image](https://github.com/ATHDY005/Ex-07-Feature-Selection/assets/84709944/d6c1dd24-9c5f-4965-8251-2ac991b7ea7e)
![image](https://github.com/ATHDY005/Ex-07-Feature-Selection/assets/84709944/7a535c58-023c-4c0a-a671-6f2d84c26d6e)
![image](https://github.com/ATHDY005/Ex-07-Feature-Selection/assets/84709944/1a8c7ddf-4ffd-4968-accf-c12d8ef8704a)
![image](https://github.com/ATHDY005/Ex-07-Feature-Selection/assets/84709944/1efc7a2f-1f39-4f2a-a10f-285e1ce0b7ba)
![image](https://github.com/ATHDY005/Ex-07-Feature-Selection/assets/84709944/0a878b0a-12b0-47ef-8a18-066fa1c83356)
![image](https://github.com/ATHDY005/Ex-07-Feature-Selection/assets/84709944/61f0a9d7-1cd8-42bc-94f5-d5a6298bc9a3)
![image](https://github.com/ATHDY005/Ex-07-Feature-Selection/assets/84709944/4c3b340d-3cee-4963-93d7-82236df6a36e)
![image](https://github.com/ATHDY005/Ex-07-Feature-Selection/assets/84709944/ad691094-5737-4cd6-94f5-bc47fbd415b4)
![image](https://github.com/ATHDY005/Ex-07-Feature-Selection/assets/84709944/20d4f84b-7949-414e-aad0-685fd1dea6c6)
![image](https://github.com/ATHDY005/Ex-07-Feature-Selection/assets/84709944/d2fb7552-e66d-4cb4-b19b-ba6e220c5425)
![image](https://github.com/ATHDY005/Ex-07-Feature-Selection/assets/84709944/38dff88d-55a1-4846-b8b2-bc245fe1077b)
![image](https://github.com/ATHDY005/Ex-07-Feature-Selection/assets/84709944/eb95b60f-ad3f-4833-80b2-bd2cd215bca4)
![image](https://github.com/ATHDY005/Ex-07-Feature-Selection/assets/84709944/776ccfe3-3f9c-414d-a060-63da76076a01)
![image](https://github.com/ATHDY005/Ex-07-Feature-Selection/assets/84709944/1b214e42-ba18-4742-8d7e-f85f7d7c7289)
![image](https://github.com/ATHDY005/Ex-07-Feature-Selection/assets/84709944/d885b0d8-80b1-473a-94ef-1e03cd941bda)


# RESULT 
The various feature selection techniques are performed on a dataset and saved the data to a file.
