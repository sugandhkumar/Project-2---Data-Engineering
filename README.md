# Disaster Response Pipeline Project

### Project Summary:

This project entails classification of real time distaster messages into various categories.
The data used for training the model has been provided by Figure Eight.

Using the datasets provided, the ETL pipeline extracts the relevant data and transforms 
and loads it to a sql database.
Using the transformed data, we then train a multi target classifier model to predict the 
categories of a given message

We also create a web app where the model is deployed


### File Descriptions:

There are three files in the project - 

1. process_data.py - This is the file where the ETB pipeline is being executed. Using the 
   datasets provided, the ETL ipeline extracts the relevant data and transforms 
   and loads it to a sql database.
   
2. train_classifier.py - This file has the part where the classifier model is being
   and trained. The trained model output is stored as a pickle file
   
3. run.py - This part has the code where the model is being deployed into a Flask app


## Input datasets - 
1. messages.csv - Dataset with messages and their genres
2. categories.csv - Dataset with message and their classification categories


### Instructions to execute:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/