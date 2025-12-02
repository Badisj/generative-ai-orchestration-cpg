import numpy as np 
import pandas as pd 
import os
import json
import joblib


def FillNa(df):
    to_complete = df.loc[df.Time.isna()==True, "resourceid"].unique()
    time_completion = np.arange(0.5, 220, 1.5)

    for elt in to_complete:
        missing = df.loc[df.resourceid==elt, :]
        for timestamp in time_completion:
            missing.loc[:,"Time"] = timestamp
            df = pd.concat((df, missing), axis=0)
            df.drop(index=df.loc[(df.resourceid==elt) & df.Time.isna()].index, inplace=True)
            df.drop_duplicates(subset=["formulaUri", "Time"], inplace=True)

    return df


################################### Inference  ###############################################

def process(data, parameters, job_id: str):

    # Retrieve data & model
    df = pd.read_csv(data[0]['downloaded'][0])
    model = joblib.load(data[1]['downloaded'][0])

    creaming = [
        'Time',
        'Viscosity'
    ]

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

    df[composition] = df[composition].replace([np.NaN, '', ' ', '  '], 0).astype(float)
    
    # Fill NaNs with dedicated function
    df = FillNa(df)
    df.sort_values(by=["formulaNumber", "Time"], inplace=True)
    df.dropna(subset=["Time"], inplace=True)
    

    X_infer = pd.concat((df[composition], df[creaming[:-1]]), axis=1)
    df['predictedViscosity'] = model.predict(X_infer)
    

    to_drop = [
        "formulaDescription",
        "ownerSection",
        "Conclusions 3",
        "Objectives",
        "trial notes",
        "Conclusions 1",
        "Conclusion 2",
        "class1"
    ]
    df.drop(columns=to_drop, inplace=True)


    ls_infer = []
    path_infer = os.path.abspath('Data_trials_creaming_inference.csv')
    file_id_infer = "formulasRDF/inference/{}/data/Data_trials_creaming_inference.csv".format(creaming[-1])
    df.to_csv(path_infer, index=False)


    ls_infer.append(
         {
            "file_path": path_infer,
            "file_id": file_id_infer,
            "bucket_name": data[0]["bucket_name"]
        }
    )

    return ls_infer