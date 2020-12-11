#Import the libraries
import pandas as pd
import numpy as np

#get the data
data=pd.read_csv('Wine data.csv')

#lebel the values in the quality variables
bins=(2,6.5,8)
label=['Bad','Good']
data['quality']=pd.cut(data['quality'],bins=bins,labels=label)

#Encode the values in quality in to 0,1
from sklearn.preprocessing import LabelEncoder
lc=LabelEncoder()
data['quality']=lc.fit_transform(data['quality'])

#rename the columns for better understanding
data.rename(columns={'fixed acidity':'fixed_acidity','volatile acidity':'volatile_acidity',
                     'citric acid':'citric_acid','residual sugar':'residual_sugar',
                     'free sulphur dioxide':'free_sulphur_dioxide','total sulphur dioxide':'total_sulphur_dioxide'
                     },inplace=True)

#split the data in to 'x' and 'y'
x=data.iloc[:,:-1]
y=data['quality']

#split the data in to train and test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=0)


#Model building
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100,max_depth=10,random_state=0)
rf.fit(x_train,y_train)

#predict the model on test_set
rf_pred=rf.predict(x_test)
print(rf_pred[:9])

#get the performance of the model on test data
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print(accuracy_score(y_test,rf_pred))
print(classification_report(y_test,rf_pred))
print(confusion_matrix(y_test,rf_pred))

#dump the model
import pickle
filename='random_model.pickle'
pickle.dump(rf,open(filename, 'wb'))

