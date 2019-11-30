# Import the required libraries needed
# for the application to run via Flask
# and the Machine learning Model

import os
import pickle
import numpy as np 
from flask import Flask
from flask import render_template
from flask import url_for
from flask import request

# Initialize the Application
app = Flask(__name__)

# The function below renders the prediction 
# template of the data which gives end-users
# different options to select the required data
@app.route('/')
def prediction():
	return render_template('predict.html')

# The function initializes the numerical python module
# to shape the vals into an array and load our Model in
# the Pickle file
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,12)
    loaded_model = pickle.load(open("model.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

# This route actually checks if the form has been filled
# and then returns the values of the form and pass it to
# the Prediction template to retrieve results.
@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)

        # After the values has been checked by mapping 
        # the values of the form from a dictionary to a
        # List return 1 if condition is true or 0 if false
        if int(result)==1:
            prediction='Income more than 50K'
        else:
            prediction='Income less that 50K'
        return render_template("result.html",prediction=prediction)

# Finally run the main event Loop of the Application
if __name__ == '__main__':
	app.run(debug=True)