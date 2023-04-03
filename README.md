# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Upload the file to your cell.
2. Type the required program.
3. Print the program.
4. End the program.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: POOJA A
RegisterNumber: 212222240072
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

X = df.iloc[:,:-1].values
X

y = df.iloc[:,-1].values
y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)

y_pred

y_test

plt.scatter(X_train,y_train,color="green")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,y_test,color="grey")
plt.plot(X_test,regressor.predict(X_test),color="purple")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
*/
```

## Output:
![exp2](https://user-images.githubusercontent.com/119390329/229418695-f10be7f8-148e-4162-a889-8225ae9163fe.png)
![ex2](https://user-images.githubusercontent.com/119390329/229419045-42572467-3df8-4a9f-a348-5a6b35c582fa.png)
![e2](https://user-images.githubusercontent.com/119390329/229567182-d5c8655b-3689-443b-bd8a-d836e06317ba.png)
![X2](https://user-images.githubusercontent.com/119390329/229567328-5320efed-26f7-4794-9e68-6edc6c8a8b33.png)
![2](https://user-images.githubusercontent.com/119390329/229567400-8109a5b5-cd83-4de7-943c-f7bd3d4079b4.png)
![2X](https://user-images.githubusercontent.com/119390329/229567471-0f76a00a-8a1e-4122-b272-1170a106f4e4.png)
![EP2](https://user-images.githubusercontent.com/119390329/229567588-51242e96-d80c-46dc-aabd-f30d299c3eca.png)
![2(1](https://user-images.githubusercontent.com/119390329/229567626-038980fe-de97-4b39-805e-def215e04b4d.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
