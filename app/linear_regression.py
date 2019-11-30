# Author:Samuel Jim Nnamdi
# Project: Prediction using decisionTree Algorithm
# Date: April 26th 2019

# Import Libraries

import pandas
import numpy
import pickle
import sklearn 
from sklearn import tree 
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
#from yellowbrick.classifier import classificationReport

# Get the dataset and read the dataset

get_dataset = "static/assets/dataset/adult.csv"
read_dataset = pandas.read_csv(get_dataset)

# filter the data and check for empty values
# Replace the empty values with the modal value
# of each column

for col in read_dataset:
	read_dataset[col] = read_dataset[col].replace("?",numpy.NaN)
	read_dataset = read_dataset.apply(lambda x:x.fillna(x.value_counts().index[0]))

# Perform Data discretization and make data easy
# to read and Understand more clearly

read_dataset.replace(['Divorced','Married-AF-spouse','Married-civ-spouse','Married-spouse-absent','Never-married','Separated','Widowed'],['divorced','married','married','married','not married','not married','not married'], inplace=True)

# Try to Label Encode the data

category_columns= ['workclass', 'race', 'education','marital-status', 'occupation',               'relationship', 'gender', 'native-country', 'income']

labelEncoder = preprocessing.LabelEncoder()

# Mapping the dictionary of values

mapping_dict = {}
for col in category_columns:
	read_dataset[col] = labelEncoder.fit_transform(read_dataset[col])
	le_name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
	mapping_dict = le_name_mapping
read_dataset=read_dataset.drop(['fnlwgt','educational-num'], axis=1)
read_dataset.head()

# Fitting the data to be trained

X = read_dataset.values[:,:12]
y = read_dataset.values[:, 12]

# Train the data using DecisionTreeAlgorithm
# Using the gini index and entropy functions
# Testdata should be 0.3 -- TrainingData should be 0.7

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.3, random_state=100)

# Train the data using LinearRegression
# Testdata should be 0.3 -- TrainingData should be 0.7

linear_reg = LinearRegression()
linear_reg.fit(X_train,y_train)
prediction_Lreg = linear_reg.predict(X_test)

# Calculate the Accuracy score of our LinearRegression classifier

# print("Accuracy Score using LinearRegression is:", accuracy_score(y_test, prediction_Lreg))

# Train the data using LogisticRegression
# Testdata should be 0.3 -- TrainingData should be 0.7

logistic_reg = LogisticRegression()
logistic_reg.fit(X_train,y_train)
prediction_Loreg = logistic_reg.predict(X_test)

# Calculate the Accuracy score of our LinearRegression classifier

 # print("Accuracy Score using LogisticRegression is:", accuracy_score(y_test, prediction_Loreg))

# Classification report for linearRegression

'''
visualiser = classificationReport(linear_reg,classes = ['won','loss'])
visualiser.fit(X_train,y_train)
visualiser.score(y_test,prediction_Lreg)
result = visualiser.poof()

# Classification report for LogisticRegression

'''
'''
visualiser = classificationReport(logistic_reg,classes = ['won','loss'])
visualiser.fit(X_train,y_train)
visualiser.score(y_test,prediction_Loreg)
result = visualiser.poof()
'''

# Now lets say that after all our analysis we choose
# Logistic regression as our algorithm to work with
# We'll write the model to a pkl file

pickle.dump(logistic_reg,open('model.pkl','wb'))