import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('call.pkl', 'rb'))

@app.route('/')
def home():
    
    return render_template('cal.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    int_features = [float(x) for x in request.form.values()]
    
    
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction
    print(output)

    return render_template('lovss.html', prediction_text='    You have burned {} calories'.format(output))

if __name__ =="__main__":
    app.run()