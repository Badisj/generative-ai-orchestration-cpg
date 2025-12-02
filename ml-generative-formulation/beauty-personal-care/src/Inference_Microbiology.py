#!/usr/bin/env python
# coding: utf-8

# # Cosmetics Generative Formulation
# ## `Anti-ageing cream microbiology challenge test report`

# In[ ]:


import os
import pickle
import numpy as np


# In[ ]:


# WARNING: DO NOT REMOVE OR MODIFY.
runningInJupyter = True

from plp_jupyter_data_loader import loadPipelinePilotData
plp_df, plp_params, plp_globals = loadPipelinePilotData()
import pandas as pd
if plp_df.shape[0] > 0:
    print("plp_df.dtypes:\n" + plp_df.dtypes.to_string())


# In[ ]:


os.chdir('/home/data/RTV/DEMOR424/BIOVIA/BPP2022/public/cpgretail/Cosmetics Generative Formulation/')


# ## 1. Microbiological activity prediction
# ### 1.1 Build microbiology dataset

# In[ ]:


from doepy import build

plp_df['formula_id'] = list(plp_df.index)
formula_id_microbio = list(plp_df.index)
evaluation_day_microbio = np.arange(1, 28.5, 0.5)

dic_doe_microbio= {
    "formula_id": formula_id_microbio,
    "evaluation_day": evaluation_day_microbio
}

doe_microbio = build.build_full_fact(dic_doe_microbio)


# ### 1.2 Retrieve models and perfrom predictions

# In[ ]:


# Ingredient families
ids = ['FormulaID', 'formula_id']
base = ['water']
oils = ['grapessed_oil', 'dimetichone']
surfactants = ['cetearyl_alcohol', 'glyceryl_stearate', 'decyl_glucoside', 'polysorbate20']
thickeners = ['xanthan_gum', 'carbomer']
emollients = ['jojoba_oil', 'caprylic_capric_triglycerides']
preservatives = ['phenoxyethanol', 'sodium_benzoate']
actives = ['Retinol_VitaminA', 'HyaluronicAcid', 'VitaminC_AscorbicAcid']
fragrances = ['fragrance']
microbio_test = ['evaluation_day']

# Process
process = ['speed_rpm_step3', 'temperature_celsius_step3', 'time_minutes_step4']

# Microbiological targets
targets_microbio = [
    'Staphylococcus aureus cfu g',
    'Pseudomonas aeruginosa cfu g',
    'Escherichia coli cfu g',
    'Candida albicans cfu g',
    'Aspergillus niger cfu g',
    'Enterococcus faecalis cfu g',
]


# In[ ]:


# Load Models
dic_models = {}
for target in targets_microbio:
    model_filename =  './Models/model_gradient_boosting_{}.sav'.format(target)
    dic_models[target] = pickle.load(open(model_filename, 'rb'))


# In[ ]:


if len(plp_df) >= 1: 
    # Prepare data for predictions
    df = plp_df[ids+base+oils+surfactants+thickeners+emollients+preservatives+actives+fragrances+process].copy()
    doe_formulas_microbio = pd.merge(df, doe_microbio, on='formula_id', how='inner')
    X_pred = doe_formulas_microbio.drop(columns=ids).values

    # Generate predictions
    dic_pred = {}
    for tgt in targets_microbio:
        y_pred = dic_models[tgt].predict(X_pred) 
        dic_pred[tgt] = y_pred

    Y_pred = pd.DataFrame(dic_pred, columns = list(dic_pred.keys())).round(1)
    doe_formulas_microbio = pd.concat((doe_formulas_microbio.reset_index(drop=True), Y_pred.reset_index(drop=True)), axis=1)


# ### 1.3 Apply challenge test acceptance crieteria

# In[ ]:


dic_microbial_stability_limits = {
    'Staphylococcus aureus cfu g': {
        'days': 7, 
        'limit': 1000
    },
    'Pseudomonas aeruginosa cfu g': {
        'days': 7, 
        'limit': 1000
    },
    'Escherichia coli cfu g': {
        'days': 7, 
        'limit': 1000
    },
    'Candida albicans cfu g': {
        'days': 14, 
        'limit': 10000
    },
    'Aspergillus niger cfu g': {
        'days': 14,
        'limit': 10000
    },
    'Enterococcus faecalis cfu g': {
        'days': 7, 
        'limit': 1000
    }
}


# In[ ]:


if plp_globals['Microbiology'] == 'Pass':
    to_drop = []
    for tgt, values in zip (dic_microbial_stability_limits.keys(), dic_microbial_stability_limits.values()):
        doe_formulas_microbio['{} days'.format(tgt)] = values['days']
        doe_formulas_microbio['{} limit'.format(tgt)] = values['limit']
        doe_formulas_microbio.loc[doe_formulas_microbio[tgt] < 0, tgt] = 10 

        cond = (doe_formulas_microbio['evaluation_day'] >= values['days']) & (doe_formulas_microbio[tgt] > values['limit'])
        to_drop += list(doe_formulas_microbio.loc[cond, 'FormulaID'].values)

    doe_formulas_microbio.drop(
        index=list(doe_formulas_microbio.loc[doe_formulas_microbio['FormulaID'].isin(to_drop), :].index), 
        inplace=True
    )
    
else:
    for tgt, values in dic_microbial_stability_limits.keys():
        doe_formulas_microbio.loc[doe_formulas_microbio[tgt] < 0, tgt] = 10


doe_formulas_microbio.to_csv(
    './Data/generated/microbiology/generated_anti_ageing_cream_microbiology_challenge_test.csv',
    index=False
)


# In[ ]:


plp_df=plp_df.loc[plp_df['FormulaID'].isin(doe_formulas_microbio['FormulaID'].unique()),:].drop(columns=['formula_id'])

