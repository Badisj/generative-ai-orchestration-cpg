import numpy as np 
import pandas as pd 
import os
import json
import joblib


################################### Inference  ###############################################

def process(data, parameters, job_id: str):

    composition = [
        '%Water',
        '%Condensat',
        '%PLE',
        '%MPC 80',
        '%Milk Powder',
        '% Rennet casein',
        '%Acid casein',
        '%Lactic Acid',
        '%Anhydrci Citric Acid',
        '%PS 35',
        '%Maasdam',
        '%Baby Cheese',
        '%Cheddar Cheese',
        '(%) Gouda Cheese',
        '%Cheese Cat 2',
        '%Cheese Cat 1',
        '%Swiss Cheese',
        '%Downgraded Pressed Paste Cat 1',
        '%Downgraded Pressed Paste Cat 2',
        '(%) PP25',
        '(%)  PP50',
        '%Butter',
        '%Melting Salt P50',
        '%Melting Salt K 2285',
        '%Fine Salt Refined Dry',
        '%DIHYDRATE TRISODIUM CITRATE',
        '%CALCIUM CONCENTRATE',
        '%Starch',
        '%RMC 80 HIGH HEAT',
        '%TEXTURING AGENT FLANOGEN',
        '%SICAMELT NSQ23',
        '%SICAMELT C23',
        '%C23 NF',
        '%CCL',
        '%CLL.th',
        'Vitamine D'
    ]


    # Retrieve inference dataset from 3DX Platform
    df = pd.read_csv(data[0]['downloaded'][0])
    
    df[composition] = df[composition].replace([np.NaN, '', ' ', '  '], 0).astype(float)
    X_infer = df[composition].copy()


    # Retrieve models & predict properties 
    thermomechanical_preds = [
        'pred_L',
        'pred_a',
        'pred_b',
        'pred_D+20',
        'pred_D+7',
    ]

    for tgt, idx in zip(thermomechanical_preds, range(1, len(data))):
        model = joblib.load(data[idx]['downloaded'][0])
        df[tgt] = model.predict(X_infer)

    
    # Post processing
    df.sort_values(by=["formulaNumber"], inplace=True)

    metadata = [
        'resourceid', 'formulaName', 'fullFormulaNumber', 'formulaUri', 
        'Trial code', 'Process', 'Collection'
    ]

    thermomechanical = [
        'D+20',
        'D+7',
        'L',
        'a',
        'b'
    ]

    to_keep = metadata + composition + thermomechanical + thermomechanical_preds
    df = df[to_keep].copy()


    # Configure outputs & save dataset
    ls_infer = []
    path_infer = os.path.abspath('Data_trials_thermomechanical_inference.csv')
    file_id_infer = "formulasRDF/inference/Thermomechanical/data/Data_trials_thermomechanical_inference.csv"
    df.to_csv(path_infer, index=False)


    ls_infer.append(
         {
            "file_path": path_infer,
            "file_id": file_id_infer,
            "bucket_name": data[0]["bucket_name"]
        }
    )

    return ls_infer