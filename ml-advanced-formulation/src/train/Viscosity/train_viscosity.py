import numpy as np 
import pandas as pd 
import os
import json
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



################################### Model Training  ###############################################

def process(data, parameters, job_id: str):

    # Retrieve train & test subsets
    df_train = pd.read_csv(data[0]['downloaded'][0], sep=',')
    df_test = pd.read_csv(data[1]['downloaded'][0], sep=',')

    target = df_train.columns[-1]


    # Fit Gradient Boosting model
    X_train, y_train = df_train.drop(columns=[target]).values, df_train.loc[:, target].values
    X_test, y_test = df_test.drop(columns=[target]).values, df_test.loc[:, target].values

    gb = GradientBoostingRegressor(
         n_estimators=500, 
         max_depth=7, 
         random_state=42
         )
    
    gb.fit(X_train, y_train)


    # Get Predictions
    y_pred = gb.predict(X_test)


    # Initialize outputs list for model artifacts
    artifacts = []


    # Save resulting model
    path_model = os.path.abspath('Gradient_boosting_{}.sav'.format(target.lower()))
    file_id_model = "formulasRDF/processed/lot2/viscosity/model/'Gradient_boosting_{}.sav".format(target.lower())
    joblib.dump(gb, path_model)

    artifacts.append(
         {
            "file_path": path_model,
            "file_id": file_id_model,
            "bucket_name": data[0]["bucket_name"]
        }
    )


    # Save evaluation metrics json
    report_dict = {
        "metrics": {
            "MSE": {
                "value": mean_squared_error(y_test, y_pred),
            },
            
            "MAE": {
                "value": mean_absolute_error(y_test, y_pred),
            },

            "R2 Score": {
                "value": r2_score(y_test, y_pred),
            },
        },
    }

    path_evaluation = os.path.abspath('metrics.json')
    file_id_evaluation = "formulasRDF/processed/lot2/viscosity/model/metrics.json"
    with open(path_evaluation, 'w') as f:
          f.write(json.dumps(report_dict))

    artifacts.append(
         {
            "file_path": path_evaluation,
            "file_id": file_id_evaluation,
            "bucket_name": data[0]["bucket_name"]
        }
    )


    return artifacts