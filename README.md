# Disaster Response Pipeline Project

The dataset that we'll be working on consists of natural disaster messages that are classified into 36 different classes. 
The dataset was provided by [Figure Eight](https://appen.com/). Example of input messages:
```
['Weather update - a cold front from Cuba that could pass over Haiti',
 'Is the Hurricane over or is it not over',
 'Looking for someone but no name',
 'UN reports Leogane 80-90 destroyed. Only Hospital St. Croix functioning. Needs supplies desperately.',
 'says: west side of Haiti, rest of the country today and tonight']
```

### Required libraries 
[pyproject.toml](https://github.com/elenacramer/Disaster_Response/blob/main/pyproject.toml) lists all required libraries. The install command reads the pyproject.toml file from the current project, resolves the dependencies, and installs them.
```
poetry install
```


### File Descriptions
- [Data](https://github.com/elenacramer/Disaster_Response/tree/main/data): Contains the raw data in csv form as well as the function 'process_data.py'. The function consists of a ETL pipeline that cleans data and stores it in a database.  To run the pipeline: 
    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db` 
    
- [Model](https://github.com/elenacramer/Disaster_Response/tree/main/models): contains the function 'train_classifier' which consists of a ML pipeline that trains a classifier and saves it. To run the pipeline:
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
     
- [App](https://github.com/elenacramer/Disaster_Response/tree/main/app): To run the web app run the following command in the app's directory: 
    `python run.py`  
    Then got to http://0.0.0.0:3001/


