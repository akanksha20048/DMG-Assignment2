
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier

def runner():
    #Read the file for training
    path1 = "C:/Users/Akanksha/Desktop/Mtech/Mtech_Sem1/DMG/Assignments/DMG_ASSIGNMENT2/dataset/given_dataset.csv"
    
    
    # load dataset
    df = pd.read_csv(path1)
    col_names=list(df.columns)
    
    
    #split dataset in features and target variable
    feature_cols = col_names[0:-1]
    X = df[feature_cols] # Features
    y = df[['T']] # Target variable
    del X['id']
 
    
    #  Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

    
    #Preprocessing: UnderSampling
    # define undersample strategy
    undersample = RandomUnderSampler(sampling_strategy='majority')
    # fit and apply the transform
    X, y = undersample.fit_resample(X_train, y_train)
    
    
    #Model used: XGBoost Classifier
    clf = XGBClassifier( n_estimators=300, max_depth=10)
    clf.fit(X, y)
    
    #predict
    df = pd.read_csv(path2)
    X_test=df
    del X_test['id']
    prediction=clf.predict(X_test)
    df['T']=prediction
    df['id']=pd.read_csv(path2)['id']
    df=df[['id','T']]
    df.to_csv("C:/Users/Akanksha/Desktop/Mtech/Mtech_Sem1/DMG/Assignments/DMG_ASSIGNMENT2/dataset/result_xgboost.csv",index=False)
    
if __name__ == "__main__":
    path2 = "C:/Users/Akanksha/Desktop/Mtech/Mtech_Sem1/DMG/Assignments/DMG_ASSIGNMENT2/dataset/to_predict.csv"
    runner(path2)
    
    

import joblib
    joblib.dump(clf, 'C:/Users/Akanksha/Desktop/Mtech/Mtech_Sem1/DMG/Assignments/DMG_ASSIGNMENT2/dataset/model_rfregressor.pkl') 
    # Load the model from the file 
    data = joblib.load('C:/Users/Akanksha/Desktop/Mtech/Mtech_Sem1/DMG/Assignments/DMG_ASSIGNMENT2/dataset/model_rfrregressor.pkl')