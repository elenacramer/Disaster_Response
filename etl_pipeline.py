import pandas as pd
from sqlalchemy import create_engine

# load messages dataset
messages = pd.read_csv('messages.csv')

# load categories dataset
categories = pd.read_csv('categories.csv')

# merge datasets
df = messages.merge(categories, how='outer', on='id')

# create a dataframe of the 36 individual category columns
categories = df['categories'].str.split(pat=';', expand=True)

# select the first row of the categories dataframe
row = categories.iloc[0]

# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything 
# up to the second to last character of each string with slicing
category_colnames = row.apply(lambda x: x[:-2]).tolist()

# rename the columns of `categories`
categories.columns = category_colnames

for column in categories.columns:
    # set each value to be the last character of the string
    categories[column] = categories[column].apply(lambda x: x[-1])
    
    # convert column from string to numeric
    categories[column] = categories[column].astype('int')

    # drop the original categories column from `df`
df = df.drop('categories', axis=1)

# concatenate the original dataframe with the new `categories` dataframe
for col in categories.columns:
    df[col] = categories[col]


# drop duplicates
df = df.drop_duplicates() 

engine = create_engine('sqlite:///messages.db')
df.to_sql('messages_categories', engine, index=False)