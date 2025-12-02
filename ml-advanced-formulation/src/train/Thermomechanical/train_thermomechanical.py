import numpy as np 
import pandas as pd 
import os
import json
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



################################### Models Training  ###############################################

def process(data, parameters, job_id: str):

    # Retrieve train & test subsets
    keys_train = [
         'L_train',
         'D7_train',
         'D20_train',
         'a_train',
         'b_train',
    ]

    keys_test = [
        'L_test',
        'D7_test',
        'D20_test',
        'a_test',
        'b_test'
    ]
    
    dic_train = {}
    dic_test = {}

    index_train = [0, 2, 4, 6, 8]
    index_test = [1, 3, 5, 7, 9]

    # Initialize outputs list for model artifacts
    artifacts = []


    for key_train, idx_train, key_test, idx_test in zip(keys_train, index_train, keys_test, index_test):
         
         # Read train & test datasets
         dic_train[key_train] = pd.read_csv(data[idx_train]['downloaded'][0], sep=',')
         dic_test[key_test] = pd.read_csv(data[idx_test]['downloaded'][0], sep=',')

         df_train = dic_train[key_train].copy()
         df_test = dic_test[key_test].copy()
         
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
         
         
         # Save model
         path_model = os.path.abspath('Gradient_boosting_{}.sav'.format(target.lower()))
         file_id_model = "formulasRDF/processed/lot1/thermomechanical/{}/model/'Gradient_boosting_{}.sav".format(target, target.lower())
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
         file_id_evaluation = "formulasRDF/processed/lot1/thermomechanical/{}/model/metrics.json".format(target)
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