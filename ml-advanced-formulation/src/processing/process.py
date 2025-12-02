import numpy as np 
import pandas as pd 
import os
import json

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


########################## Fill NaNs with Gradient Boosting  ######################################
def fillna(df, X_cols, to_fill):
    for col in to_fill:
        df_gb = pd.concat((df[X_cols], df[col]), axis=1)
        gb = GradientBoostingRegressor().fit(df_gb.dropna()[X_cols], df_gb.dropna()[col])
        df.loc[df[col].isna(), col] = gb.predict(df.loc[df[col].isna(), X_cols])
    return df




################################### Data Processing ###############################################

def process(data, parameters, job_id: str):

    # df = pd.read_csv('export-0000 (1).csv')
    df = pd.read_csv(data[0]['downloaded'][0], sep=',')
    df_creaming = pd.read_csv(data[1]['downloaded'][0], sep=',')


    # ------------ Thermomechanical & Sensory datasets for Lot 1 ------------
    cond_lot1 = (df.formulaName.str.contains('Formula')) & (df.resourceid.isnull() == False)
    df = df.loc[cond_lot1, :]
    df.loc[cond_lot1, 'Lot'] = 1

    
    metadata = [
        'resourceid', 'formulaName', 'fullFormulaNumber', 
        'maturityState', 'owner', 'creationDate', 'modificationDate', 
        'Trial code', 'Process', 'Collection'
    ]

    ids = ['formulaName', 'Trial code']
    df.drop_duplicates(subset=ids, inplace=True)


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

    thermomechanical = [
        'D+20',
        'D+7',
        'L',
        'a',
        'b'
    ]

    sensory = [
        'Taste Rate',
        'Texture Rate',
        'Firmness Rate',
        'Appearance Rate',
        'Irritancy Rate'
    ]

    creaming = [
        'Time',
        'Viscosity'
    ]


    # Fill NaN values
    df[composition] = df[composition].replace([np.NaN, '', ' ', '  '], 0).astype(float)
    df = fillna(df, X_cols=composition, to_fill=thermomechanical)



    # -------------------------- Thermomechanical & Sensory dataset for Lot 1 --------------------------

    # Configure thermomechanical outputs
    list_dics = []
    subset = ['train', 'test']

    for fea in thermomechanical:
        df_ml = pd.concat((df[composition], df[fea]), axis=1).copy()
        df_ml_train, df_ml_test = train_test_split(df_ml, test_size=0.12)
        
        ls = {
            'train': df_ml_train, 
            'test': df_ml_test
            }
        
        for sub in subset:
            path = os.path.abspath('data_trials_{}_{}.csv'.format(fea, sub))
            file_id = "formulasRDF/processed/lot1/thermomechanical/{}/data/data_trials_{}_{}.csv".format(fea, fea, sub)
            ls[sub].to_csv(path, index=False)

            list_dics.append(
                {
                    "file_path": path,
                    "file_id": file_id,
                    "bucket_name": data[0]["bucket_name"] 
                }
        )



    # Configure sensory outputs
    for fea in sensory:
        df_ml_sensory = pd.concat((df[composition], df[fea]), axis=1).copy()
        df_ml_sensory.dropna(subset=[fea], inplace=True)
    
        df_ml_sensory_train, df_ml_sensory_test = train_test_split(df_ml_sensory, test_size=0.12)
        
        ls = {
            'train': df_ml_sensory_train, 
            'test': df_ml_sensory_test
            }
        
        for sub in subset:
            path_sensory = os.path.abspath('data_trials_{}_{}.csv'.format(fea, sub))
            file_id_sensory = "formulasRDF/processed/lot1/sensory/{}/data/data_trials_{}_{}.csv".format(fea, fea, sub)
            ls[sub].to_csv(path_sensory, index=False)

            list_dics.append(
                {
                    "file_path": path_sensory,
                    "file_id": file_id_sensory,
                    "bucket_name": data[0]["bucket_name"] 
                }
        )
        

    

    # -------------------------- Viscosity dataset for Lot 2 --------------------------
    df_creaming[composition] = df_creaming[composition].replace([np.NaN, '', ' ', '  '], 0).astype(float)
    df_creaming = pd.concat((df_creaming[composition], df_creaming[creaming].fillna(0)), axis=1)


    df_creaming_train, df_creamin_test = train_test_split(df_creaming, test_size=0.2)
    ls_creaming = {
        'train': df_creaming_train,
        'test': df_creamin_test
    }

    for mode in subset:
        path_creaming = os.path.abspath('data_trials_creaming_{}.csv'.format(mode))
        ls_creaming[mode].to_csv(path_creaming,  index=False)
        file_id_creaming = "formulasRDF/processed/lot2/viscosity/data/data_trials_creaming_{}.csv".format(mode)
        list_dics.append(
            {
                "file_path": path_creaming,
                "file_id": file_id_creaming,
                "bucket_name": data[0]["bucket_name"]
            }
        )


    
    return list_dics