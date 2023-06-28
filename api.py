import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
import joblib
import pandas as pd

app = Flask(__name__)
predict_model = joblib.load('trained_model.pkl')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = request.form.getlist('feature')

    column_names = ['City', 'type', 'city_area', 'condition', 'furniture', 'entranceDate', 'Area', 'room_number', 'floor', 'hasParking', 'hasBalcony', 'hasMamad', 'hasStorage']
    df = pd.DataFrame([features], columns=column_names)
    print(df.info())
    
    final_features = pd.DataFrame([features], columns=column_names)
    prediction = predict_model.predict(final_features)[0]
    # risk = predict_model.predict_proba(final_features)[0][1]

    return render_template('index.html', prediction_text='{}'.format(prediction))


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
