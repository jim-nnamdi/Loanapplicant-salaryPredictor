#! usr/bin/python

import pandas
import pickle
import numpy
import sklearn
from sklearn import tree 
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from yellowbricks.classifier import ClassificationReport

# Get the Dataset and read contents
data = 'static/assets/dataset/adult.csv'
read_data = pandas.read_csv(data)

# check and replace empty columns in the data
for column in read_data:
    read_data[column] = read_data[column].replace('', numpy.NaN)
    read_data = read_data.apply(lambda x:x.fillna(x.value_counts().index[0]))

# Now make the dataset readable by replacing cnf words
read_data.replace(['Divorced','Married-AF-spouse','Married-civ-spouse','Married-spouse-absent','Never-married','Separated','Widowed'],['divorced','married','married','married','not married','not married','not married'], inplace = True)

# Perform some data_reads here and minimal display
read_data.head(7)
read_data.tail(7)

# Label encode the data to properly read the data
data_columns = ['workclass', 'race', 'education','marital-status', 'occupation',               'relationship', 'gender', 'native-country', 'income']

# Call the label Encoder function here for case_use
label_encoder = LabelEncoder.preprocessing()
label_mapping = {}
for column in data_columns:
    data_columns[column] = label_encoder.fit_transform(data_columns[column])
    mapping_dictionary   = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    label_mapping = mapping_dictionary

# Fit the data to be trained and split training data
X = read_data.values[:, :12]
Y = read_data.values[:, :12]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)

# Use logistic regression to run the data testing 
logistic_stack = LogisticRegression()
logistic_stack.fit(X_train, y_train)
logistic_stack_predict = logistic_stack.predict(X_test)

# Convert file to pickle binary and read via Models
pickle.dump(logistic_stack.open('log.pkl', 'wb'))
