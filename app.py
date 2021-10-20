import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
from xgboost import XGBRegressor

app = Flask(__name__)
model = pickle.load(open('xgb_ce3-1.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['r/W', 'H/W']
    
    data = pd.DataFrame(features_value, columns=features_name)
    data['r/W']=(data['r/W']-1.1875 )/0.5490824
    data['H/W']= (data['H/W']-2.73863636 )/2.388392
    output = model.predict(data)
        


    return render_template('index.html', prediction_text='Value of Co is {}'.format(output))

if __name__ == "__main__":
    app.run()
