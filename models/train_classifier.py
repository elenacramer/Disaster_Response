import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import pickle

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 
from sklearn.metrics import f1_score, precision_score, recall_score

def load_data(database_filepath):
    '''
    This function loas data from SQL, i.e. sqlite and reads it into pandas DataFrame. 
    After laoding, it splits the dataframe into input and targets. 
    '''
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('df', engine)
    # drop target feature 'child_alone' since it only contains one class
    df = df.drop('child_alone', axis=1)
    # split into targets and feature
    targets = df.columns[4:].tolist()
    X = df["message"]
    Y = df[targets]
    return X, Y, targets


def tokenize(text):
    '''
    Functions that tokenizes, by words, the given text input. Then applies a lemmatizer and converts tokens 
    to lowercase.
    '''
    # tokenize
    tokens = word_tokenize(text)
    # Remove stop words
    words = [w for w in tokens if w not in stopwords.words("english")]
    # instantiate Lemmatizer object
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    '''
    This function builds a Pipeline. 
    Output:
        Pipeline
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('mclf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline 


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function predicts for the test set and computes f1, recall and precision scores.
        Inputs:
            model: model used for predictions
            Y_test: targets from the test set
            Y_pred: predicitons 
       Outputs:
            None
    '''
    Y_pred = model.predict(X_test)
    # Create datafram of Y_pred which is a np.array
    Y_pred_frame = pd.DataFrame(Y_pred, columns = category_names)
    # Create a dataframe to store scores
    result_mclf = pd.DataFrame()
    f1 = []
    precision = []
    recall = []
    # compute scores for each  target feature
    for target in category_names:
        f1.append(f1_score(Y_test[target], Y_pred_frame[target], average='macro'))
        precision.append(precision_score(Y_test[target], Y_pred_frame[target], average='macro'))
        recall.append(recall_score(Y_test[target], Y_pred_frame[target], average='macro'))

    result_mclf["f1"] = f1
    result_mclf["precision"] = precision
    result_mclf["recall"] = recall

    print(result_mclf.mean())


def save_model(model, model_filepath):
    '''
    Save model to model_filepath as pickle. 
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()