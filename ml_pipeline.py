import nltk
nltk.download(['punkt', 'wordnet'])

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import classification_report 
from sklearn.metrics import f1_score, precision_score, recall_score



def load_data():
    '''
    This function loas data from SQL, i.e. sqlite and reads it into pandas DataFrame. 
    After laoding, it splits the dataframe into input and targets. 
    '''
    engine = create_engine('sqlite:///messages.db')
    df = pd.read_sql_table('messages_categories', engine)
    # split into targets and feature
    targets = df.columns[4:].tolist()
    X = df["message"]
    Y = df[targets]
    return X, Y


def tokenize(text):
    '''
    Functions that tokenizes, by words, the given text input. Then applies a lemmatizer and converts tokens 
    to lowercase.
    '''
    # tokenize
    tokens = word_tokenize(text)
    # instantiate Lemmatizer object
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def collect_results(Y_test, Y_pred):
    # Y_pred is np.array, first we make a dataframe of Y_pred
    Y_pred_frame = pd.DataFrame(Y_pred, columns = Y_test.columns.tolist())
    target_names = Y_test.columns.tolist()

    result_mclf = pd.DataFrame()
    f1 = []
    precision = []
    recall = []

    for target in target_names:
        f1.append(f1_score(Y_test[target], Y_pred_frame[target], average='micro'))
        precision.append(precision_score(Y_test[target], Y_pred_frame[target], average='micro'))
        recall.append(recall_score(Y_test[target], Y_pred_frame[target], average='micro'))

    result_mclf["f1"] = f1
    result_mclf["precision"] = precision
    result_mclf["recall"] = recall

    return result_mclf


def main():
    '''
    This functions loads data, splits into input, targets, train and test. 
    It defines a pipeline for a MultiOutputClassifier, fits the train set and predicts
    for the test set. It then displays the f1 score, precision and recall. 
    '''
    X, Y = load_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('mclf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    # train classifier
    pipeline.fit(X_train, Y_train)

    # predict on test data
    Y_pred = pipeline.predict(X_test)

    # display results
    result = display_results(Y_test, Y_pred)
    return collect.mean()

main()