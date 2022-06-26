# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.Import the necessary packages using import statement.
2.Read the given csv file and print the number of contents to be displayed.
3.Split the dataset using train_test_split.
4.Calculate Y_Pred and accuracy.
5.Display the result.

## Program:
```

Program to implement the SVM For Spam Mail Detection..
Developed by: RAGUL VK
RegisterNumber:  212221240043


import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["EmailText"].values
y=data["Label"].values
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:
### data.head()

![M1](https://user-images.githubusercontent.com/94154683/173188644-fb111910-cb13-4fcb-8c0a-a6ec86c855b3.png)

### data.info()

![M2](https://user-images.githubusercontent.com/94154683/173188648-a5ff22ad-b4b0-403b-a0cd-20d628e81f11.png)

### Data.isnull().sum()
![M3](https://user-images.githubusercontent.com/94154683/173188651-10d3d90e-1646-481f-8af8-4d2cd73cdedd.png)


### y_pred

![M4](https://user-images.githubusercontent.com/94154683/173188660-97da40d7-3ed8-4520-b95d-cd1df1f7079b.png)

### accuracy


![M5](https://user-images.githubusercontent.com/94154683/173188662-c92f36b2-48b5-439a-9d82-f0b82fad4e23.png)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
