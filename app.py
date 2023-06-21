import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route("/predict", methods = ['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template('home.html', 
        prediction_text = "You've burned {} calories.".format(round(prediction[0],2) ))

if __name__=="__main__":
    app.run(debug=True)