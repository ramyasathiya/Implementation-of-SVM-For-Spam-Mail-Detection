# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1. Start the Program.

Step 2. Import the necessary packages.

Step 3. Read the given csv file and display the few contents of the data.

Step 4. Assign the features for x and y respectively.

Step 5. Split the x and y sets into train and test sets.

Step 6. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.

Step 7. Find the accuracy of the model.

Step 8. End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: RAMYA S
RegisterNumber:212222040130
*/
```
```
import chardet

file ='spam.csv'
with open (file, 'rb' )as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding = 'Windows-1252')
data.head()

data.info()

data.isnull().sum()

x = data["v2"].values
y = data["v1"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
print("Predicted value : ",y_pred)

from sklearn import metrics 
accuracy = metrics.accuracy_score(y_test,y_pred)
print("Accuracy: ",accuracy)
```




## Output:
# detecting the character encoding:
![image](https://github.com/user-attachments/assets/36c4767f-55bc-4cd1-8c01-eb094cf6ee80)
# head():
![image](https://github.com/user-attachments/assets/9fabf25c-02ba-487a-86ea-51b92804bad7)
# .info()
![image](https://github.com/user-attachments/assets/c3dae0df-81e7-4ab1-a4bb-badb527e4078)
# checking for null values :
![image](https://github.com/user-attachments/assets/fc7fa5af-292e-49b2-8ed3-6407f558440b)
# Predicted values :
![image](https://github.com/user-attachments/assets/5cc0570f-0423-448a-a558-35cf6e6dca08)
# Accuracy :
![image](https://github.com/user-attachments/assets/46f927c6-a579-40d3-aa78-5931056ebedd)






## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
