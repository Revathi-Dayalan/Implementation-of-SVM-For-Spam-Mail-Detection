# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Revathi D
RegisterNumber: 212221240045
*/
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/content/Spam.csv',encoding='latin-1')
df = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)

df.head()

df.info()

df.isnull().sum()

x=df["v1"].values
y=df["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:

![image](https://github.com/AkilaMohan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/96000574/38c8cb9e-2c1f-4392-934b-b77f4c1684c0)

![image](https://github.com/AkilaMohan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/96000574/746f92ac-b990-4a75-8cc7-970c82a47f86)

![image](https://github.com/AkilaMohan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/96000574/002d3997-b407-46da-a518-330a82cba570)

![image](https://github.com/AkilaMohan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/96000574/2834ddf9-ebfb-4954-b7af-0a53d9032a28)

![image](https://github.com/AkilaMohan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/96000574/931ccebd-829f-4542-9e14-4e06cdbcb887)




## Result
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
