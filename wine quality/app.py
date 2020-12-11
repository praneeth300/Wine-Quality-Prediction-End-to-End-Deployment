
#import the requirements
from flask import Flask,render_template,json,jsonify,request
import pickle
import numpy as np
import requests

#initialize the app

app=Flask(__name__)

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        fixed_acidity=float(request.form['fixed_acidity'])
        volatile_acidity=float(request.form['volatile_acidity'])
        citric_acid=float(request.form['citric_acid'])
        residual_sugar=float(request.form['residual_sugar'])
        chlorides=float(request.form['chlorides'])
        free_sulphur_dioxide=float(request.form['free_sulphur_dioxide'])
        total_sulphur_dioxide=float(request.form['total_sulphur_dioxide'])
        density=float(request.form['density'])
        ph=float(request.form['ph'])
        sulphates=float(request.form['sulphates'])
        alcohol=float(request.form['alcohol'])
        #load the pickle file
        filename='random_model.pickle'
        loaded_model=pickle.load(open(filename,'rb'))
        data=np.array([[fixed_acidity,volatile_acidity,citric_acid,residual_sugar,
                        chlorides,free_sulphur_dioxide,
                        total_sulphur_dioxide,density,ph,sulphates,alcohol]])
        my_prediction=loaded_model.predict(data)
        #get the result template
        return render_template('result.html',prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
