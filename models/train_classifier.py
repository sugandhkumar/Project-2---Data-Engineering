# import libraries
import sys
import pickle 
import numpy as np
import pandas as pd

import nltk
nltk.download(['punkt', 'wordnet'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline


#Load data from sql database
def load_data(database):
    """ Function to load data from sql database
    
    Inputs:
    1. table - table name in the sql database
    
    Outputs:
    1. X - np arry of predictors
    2. y - np aaray of target variables
    3. y_col_list - list of column names for y
    
    """
    df = pd.read_sql_table("Disaster_Response", 'sqlite:///{}'.format(database))
    X = df.message.values
    y = df.select_dtypes(["int"]).drop("id", axis=1).values
    y_col_list = df.select_dtypes(["int"]).drop("id", axis=1).columns.tolist()
    print("Target column name list - ", y_col_list, "\n")
    return X, y, y_col_list


# Tokenise text
def tokenize(text):
    """
    Function to tokenize
    
    Inputs:
    1. text - string to tokenise
    
    Outputs:
    1. clean tokens - tokanized and lemmatized text list
    """
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens =[]
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens  
    pass


# Instantiate the pipeline model
def build_model():
    """
    Function to execute the model
    
    Input: None
    
    Output: Instantiated model
    """
    
    pipeline = Pipeline([("vect", CountVectorizer(tokenizer = tokenize)),
                     ("tfidf", TfidfTransformer()),
                     ("clf", RandomForestClassifier())
                    ])
      
    parameters = {'clf__min_samples_leaf': [1,2,5],
                  'clf__n_estimators': [50, 100, 200]
                 }

    cv = GridSearchCV(pipeline, param_grid = parameters, n_jobs = -1)
    
    return cv


#create a display function
def show_results(cv, y_test, y_pred, y_col_list):  
    """
    Function to display and output results
    
    Inputs: 
    1. test_data - np array that has the ground truth target test data (default y_test)
    2. pred_data - n array thta has the predicted outcomes (default y_pred)
    3. test_col_list - list of column names for target test data (default y_col_list )
    4. cv - cross validated model (default model)
    
    Outputs:
    1. results_df - category wise weighted precision, recall and f1_scores
    2. results_df - category wise precision, recall and f1_score for each class
    """
    results_class_0 = pd.DataFrame(columns= ["category", "class_0_precision", "class_0_recall", "class_0_f1_score", "class_0_support"])
    results_class_1 = pd.DataFrame(columns= ["category", "class_1_precision", "class_1_recall", "class_1_f1_score", "class_1_support"])
    results_df = pd.DataFrame(columns= ["category","precision","recall", "f1_score", "support"])
    
    for num,cat in enumerate(y_col_list):
        
        precision, recall, f1_score, support = precision_recall_fscore_support(y_test[:,num], y_pred[:,num], average="weighted")
        results_df.loc[len(results_df.index)]=  [cat, precision, recall, f1_score, support]
        
        precision, recall, f1_score, support = precision_recall_fscore_support(y_test[:,num], y_pred[:,num])
        
        if cat == "child_alone":          
            results_class_0.loc[len(results_class_0.index)] = [cat, precision[0], recall[0] ,f1_score[0], support[0]]
            results_class_1.loc[len(results_class_1.index)] = [cat, 0, 0 ,0, 0]
            
        else:
            results_class_0.loc[len(results_class_0.index)] = [cat, precision[0], recall[0] ,f1_score[0], support[0]]
            results_class_1.loc[len(results_class_1.index)] = [cat, precision[1], recall[1] ,f1_score[1], support[1]]     
     
    results_df.round(2)
    results_class_df = results_class_0.merge(results_class_1, on="category", how="inner").round(2)
    results_class_df["class_0_support"] = results_class_df["class_0_support"].astype("int")
    results_class_df["class_1_support"] = results_class_df["class_1_support"].astype("int")
    
    
    print("\nModel Averages - ", "\n", "Average Precision: {}".format(results_df["precision"].mean(), "\n"))
    print("Average Recall: {}".format(results_df["recall"].mean(), "\n"))
    print("Average f1_score: {}".format(results_df["f1_score"].mean(), "\n"))
    
    print("\nClass 0 Averages - ", "\n", "Average Class 0 Precision: {}".format(results_class_df["class_0_precision"].mean(), "\n"))
    print("Average Class 0 Recall: {}".format(results_class_df["class_0_recall"].mean(), "\n"))
    print("Average Class 0 f1_score: {}".format(results_class_df["class_0_f1_score"].mean(), "\n"))
    
    print("\nClass 1 Averages - ", "\n", "Average Class 1 Precision: {}".format(results_class_df["class_1_precision"].mean(), "\n"))
    print("Average Class 1 Recall: {}".format(results_class_df["class_1_recall"].mean(), "\n"))
    print("Average Class 1 f1_score: {}".format(results_class_df["class_1_f1_score"].mean(), "\n"))
    
    print("\n Best parameters: {}".format(cv.best_params_, "\n"))
    print("\n Best estimators: {}".format(cv.best_estimator_, "\n"))
    
    
    return results_df, results_class_df 
    
    
def main():
    """
    Function to execute
    """
    
    if len(sys.argv) == 3:
        database, model_path = sys.argv[1:]       
        
        print('Loading data..\n    Database: {}, Table: Disaster_Response'.format(database))
        
        X, y, y_col_list = load_data(database)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state = 200)
    
    
        print("\nInstantiating the model..")
        
        model = build_model()
        
        print("\nFitting the model..")
        model.fit(X_train, y_train)
        
        print("\nSaving the trained model to the path {}".format(model_path), "\n")
        pickle.dump(model, open(model_path, 'wb'))
        
        print('Trained model saved..!')   
        
        print("\nEvaluating the model..")
        y_pred = model.predict(X_test)
        
        print("\nModel performance metrics..\n")
        results_df, results_class_df = show_results(model, y_test, y_pred, y_col_list)
        
          
        print("\nCategorywise performance metrics", "\n")
        print(results_df)  
                                   
        
#         return results_df, results_class_df
              
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()                 