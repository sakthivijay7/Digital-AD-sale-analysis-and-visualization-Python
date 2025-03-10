      #import library
import pandas as pd
import numpy as np
      #load dataset
path="C:/Users/ELCOT/Desktop/New folder/Python area,sale/DigitalAd_dataset.csv"
dataset=pd.read_csv(path)
#print(dataset.head())
#print(dataset.shape)
#print(dataset.columns)
     #segregate features-input x,labels-output y
x=dataset.iloc[:,:-1].values
#print(x)
y=dataset.iloc[:,-1].values
y=y.reshape(len(y),1)
#print(y)
      # dataset to be train,test,split
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=0)
#print(xtrain.shape)
#print(xtest.shape)
      #Features scalling
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
xtrain=ss.fit_transform(xtrain)
#print(xtrain)
xtest=ss.fit_transform(xtest)
#print(xtest)
     #Tranning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  RandomForestClassifier
      #models
model1=LogisticRegression()
model2=KNeighborsClassifier()
model3=GaussianNB()
model4=SVC()
model5=DecisionTreeClassifier()
model6=RandomForestClassifier()
      #models to fit the trainning
model1.fit(xtrain,ytrain)
model2.fit(xtrain,ytrain)
model3.fit(xtrain,ytrain)
model4.fit(xtrain,ytrain)
model5.fit(xtrain,ytrain)
model6.fit(xtrain,ytrain)
     #prediction
ypred1=model1.predict(xtest)
ypred2=model2.predict(xtest)
ypred3=model3.predict(xtest)
ypred4=model4.predict(xtest)
ypred5=model5.predict(xtest)
ypred6=model6.predict(xtest)
#print(ypred1.shape,ytest.shape)
#print(ypred1,ytest)
#print(np.concatenate((ypred1.reshape(len(ypred1),1), ytest.reshape(len(ytest),1)),1))
     #Evaluating model,confusion martrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ypred1,ytest)
#print("Confsion Matrix:" ,cm)
      #Accuracy score
from sklearn.metrics import accuracy_score
#print("1.Logistic regression    - Accuray Score  : {0}% ".format(accuracy_score(ypred1,ytest)*100))  #81.25% Accuracy
#print("2.K Neighbors classify   - Accuracy Score : {0}%".format(accuracy_score(ypred2,ytest)*100))   #85.0%
#print("3.Gaussion NB            - Accuracy Score : {0}%".format(accuracy_score(ypred3,ytest)*100))   #85.0%
#print("4.Support vector classify- Accuracy Score : {0}%".format(accuracy_score(ypred4,ytest)*100))   #90.0%
#print("5.Decision tree classify - Accuracy Score : {0}%".format(accuracy_score(ypred5,ytest)*100))   #83.75%
#print("6.Random forest classify - Accuracy Score : {0}%".format(accuracy_score(ypred6,ytest)*100))   #83.75%
      #Get input to prediction test
from sklearn.preprocessing import StandardScaler
Age=int(input("Enter New customer Age :"))
Salary=int(input("Entet New customer Salary :"))
newcus=[[Age,Salary]]
result=model4.predict(ss.transform(newcus))
print(result)
if result==1:
    print("Customer will Buy")
else:
    print("Customer won't Buy")

#18,82000,0
#45,22000,1



