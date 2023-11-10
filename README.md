# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: POOJA A
RegisterNumber: 212222240072
*/
```
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())

#assigning hours to X & scores to Y
X = dataset.iloc[:,:-1].values
print(X)
Y = dataset.iloc[:,-1].values
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred =reg.predict(X_test)
print(Y_pred)
print(Y_test)

#Graph plot for training data
plt.scatter(X_train,Y_train,color='blue')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```

## Output:

## 1. df.head()
![image](https://github.com/poojaanbu0/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119390329/1ff25cc4-8388-4b37-8c53-b62de237b3cd)

## 2. df.tail()
![ex2](https://user-images.githubusercontent.com/119390329/229419045-42572467-3df8-4a9f-a348-5a6b35c582fa.png)

## 3. Array value of X
![e2](https://user-images.githubusercontent.com/119390329/229567182-d5c8655b-3689-443b-bd8a-d836e06317ba.png)

## 4. Array value of Y
![X2](https://user-images.githubusercontent.com/119390329/229567328-5320efed-26f7-4794-9e68-6edc6c8a8b33.png)

## 5. Values of Y prediction
![2](https://user-images.githubusercontent.com/119390329/229567400-8109a5b5-cd83-4de7-943c-f7bd3d4079b4.png)

## 6. Array values of Y test
![2X](https://user-images.githubusercontent.com/119390329/229567471-0f76a00a-8a1e-4122-b272-1170a106f4e4.png)

## 7. Training Set Graph
![EP2](https://user-images.githubusercontent.com/119390329/229567588-51242e96-d80c-46dc-aabd-f30d299c3eca.png)

## 8. Test Set Graph
![2(1](https://user-images.githubusercontent.com/119390329/229567626-038980fe-de97-4b39-805e-def215e04b4d.png)

## 9. Values of MSE, MAE and RMSE
![EXP  2(9](https://user-images.githubusercontent.com/119390329/230725849-f4b9c2ea-7a2c-448c-b1be-7b4694764d6e.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
