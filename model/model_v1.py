import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle

def getcleandata():
    df = pd.read_csv('data/data.csv')
    df.drop(['Unnamed: 32', 'id'], inplace=True, axis=1)
    #print(df.info())
    #sns.heatmap(df.isnull())
    row,col=df.shape
    df.diagnosis=[1 if value=='M' else 0 for value in df.diagnosis]
    #print(df.tail())
    return df

def preprocessing(df):
    X=df.drop(['diagnosis'], axis=1)
    y=df['diagnosis']
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X=X)
    exportMinMaxLabels(X)
    return X,X_scaled,y,scaler

def modelcreate(df):
    X,X_scaled,y,scaler=preprocessing(df)
    X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.25,random_state=69)
    lr=LogisticRegression()
    lr.fit(X_train,y_train)
    pred=lr.predict(X_test)
    print(accuracy_score(y_test,pred), "\n", classification_report(y_test,pred))
    return lr, scaler

def exportMinMaxLabels(df):
    cols=df.agg(['min','max','mean'],axis=0).T#axis 0 signified column wise operations
    print(cols)
    with open('model/minmax.pkl','wb') as f:
        pickle.dump(cols,f)

def main():
    df = getcleandata()
    model, scaler=modelcreate(df)
   
   #now exporting the model, so that there shoudl not be a need to train the model again and again we use picke to do so

    with open('model/modelv1.pkl','wb') as f:
        pickle.dump(model,f)
    with open('model/scalerv1.pkl','wb') as f:
        pickle.dump(scaler,f)


if __name__ == '__main__':
    main()