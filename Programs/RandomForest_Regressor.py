
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.ensemble import RandomForestRegressor


def runner(path2):
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
    
    
    #Model used: Random Forest Classifier
    clf=RandomForestRegressor(max_depth=15,max_features=6,min_samples_leaf=3,min_samples_split=25, random_state =0,verbose=2)
    clf.fit(X_train,y_train)
    
    
    #predict
    df = pd.read_csv(path2)
    X_test=df
    del X_test['id']
    prediction=clf.predict(X_test)
    df['T']=prediction
    df['id']=pd.read_csv(path2)['id']
    df=df[['id','T']]
    df.to_csv("C:/Users/Akanksha/Desktop/Mtech/Mtech_Sem1/DMG/Assignments/DMG_ASSIGNMENT2/dataset/result_rfregressor.csv",index=False)
    
if __name__ == "__main__":
        path2 = "C:/Users/Akanksha/Desktop/Mtech/Mtech_Sem1/DMG/Assignments/DMG_ASSIGNMENT2/dataset/to_predict.csv"
        runner(path2)




import joblib
    joblib.dump(clf, 'C:/Users/Akanksha/Desktop/Mtech/Mtech_Sem1/DMG/Assignments/DMG_ASSIGNMENT2/dataset/model_rfregressor.pkl') 
    # Load the model from the file 
    data = joblib.load('C:/Users/Akanksha/Desktop/Mtech/Mtech_Sem1/DMG/Assignments/DMG_ASSIGNMENT2/dataset/model_rfrregressor.pkl')