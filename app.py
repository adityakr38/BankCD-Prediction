from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from src.bankcdProject.pipeline.prediction import PredictionPipeline
import traceback

app = Flask(__name__)

expected_columns = [
    'default', 'balance', 'housing', 'loan', 'contact', 'day', 'duration', 
    'campaign', 'pdays', 'previous', 'poutcome', 'age_bin_Medium', 
    'age_bin_High', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 
    'job_management', 'job_retired', 'job_self-employed', 'job_services', 
    'job_student', 'job_technician', 'job_unemployed', 'job_unknown', 
    'marital_married', 'marital_single', 'education_Secondary', 
    'education_Tertiary', 'education_Unknown', 'generation_millenials', 
    'generation_older boomers', 'generation_silent generation', 
    'generation_younger boomers', 'month_aug', 'month_dec', 'month_feb', 
    'month_jan', 'month_jul', 'month_jun', 'month_mar', 'month_may', 
    'month_nov', 'month_oct', 'month_sep'
]

@app.route('/', methods=['GET'])
def homePage():
    return render_template("index.html")

@app.route('/predict', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        try:
            # Collect and preprocess the input data
            age = float(request.form['age'])
            age_bin = pd.cut([age], bins=[0, 30, 60, 100], labels=["Low", "Medium", "High"])[0]

            # Prepare data dictionary with other inputs
            data = {
                'age_bin': age_bin,
                'job': request.form['job'],
                'marital': request.form['marital'],
                'education': request.form['education'],
                'default': 1 if request.form['default'].lower() == 'yes' else 0,
                'balance': float(request.form['balance']),
                'housing': 1 if request.form['housing'].lower() == 'yes' else 0,
                'loan': 1 if request.form['loan'].lower() == 'yes' else 0,
                'contact': request.form['contact'],
                'day': int(request.form['day']),
                'month': request.form['month'],
                'duration': float(request.form['duration']),
                'campaign': int(request.form['campaign']),
                'pdays': int(request.form['pdays']),
                'previous': int(request.form['previous']),
                'poutcome': request.form['poutcome'],
                'generation': request.form['generation']
            }

            # Convert data to DataFrame and apply encoding
            input_df = pd.DataFrame([data])
            input_df = pd.get_dummies(input_df, columns=['age_bin', 'month', 'job', 'marital', 
                                                         'education', 'contact', 'poutcome', 
                                                         'generation'], drop_first=True)

            for col in expected_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[expected_columns]
            print("Final input DataFrame shape:", input_df.shape)

            data_array = input_df.values

            obj = PredictionPipeline()
            predict = obj.predict(data_array)

            return render_template('results.html', prediction=str(predict))

        except Exception as e:
            print("An exception occurred:")
            print(traceback.format_exc())
            return 'Something went wrong. Please try again.'

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
