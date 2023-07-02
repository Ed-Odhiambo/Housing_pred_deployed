from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import streamlit as st

app = Flask(__name__)

model = pickle.load(open("models/model.pkl", "rb"))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    int_features = [float(x) for x in request.form.values()]
    features = [np.array(int_features)]
    prediction = model.predict(features)
    
    output = round(prediction[0], 2)
    
    return render_template("index.html", prediction_text="The house roughly costs {} dollars".format(output))

if __name__ == '__main__':
    app.run(port=3000, debug=True)
    