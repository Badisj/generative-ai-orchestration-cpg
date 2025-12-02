#!/usr/bin/env python
# coding: utf-8

# # Cosmetics Generative Formulation
# ## `Anti-ageing cream stability report`

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


# ## 1. Stability assessment
# ### 1.1 Build stability dataset

# In[ ]:


os.chdir('/home/data/RTV/DEMOR424/BIOVIA/BPP2022/public/cpgretail/Cosmetics Generative Formulation/')


# In[ ]:


from doepy import build

plp_df['formula_id'] = list(plp_df.index)
formula_id = list(plp_df.index)
temperature = [4, 25, 37, 45, 50]
freeze_thaw = [0, 1]
sun_shade = [0,1]
evaluation_week = [0, 1, 4, 12]

dic_doe_stability = {
    "formula_id": formula_id,
    "temperature": temperature,
    "freeze_thaw": freeze_thaw,
    "sun_shade": sun_shade,
    "evaluation_week": evaluation_week
}

doe_stability = build.build_full_fact(dic_doe_stability)


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

# Stability targets
targets_stability = [
    'Play Time (s)',
    'Stickiness',
    'Slippery Finish',
    'Spreadability',
    'Viscosity (Pa.s)',
    'pH',
    'Color',
    'Odor',
#     'homogeneity'
]


# In[ ]:


# Load Models
dic_models = {}
for target in targets_stability:
    model_filename =  './Models/model_gradient_boosting_{}.sav'.format(target)
    dic_models[target] = pickle.load(open(model_filename, 'rb'))


# In[ ]:


if len(plp_df) >= 1: 
    # Prepare data for predictions
    df = plp_df[ids+base+oils+surfactants+thickeners+emollients+preservatives+actives+fragrances+process].copy()
    doe_formulas_stability = pd.merge(doe_stability, df, on='formula_id', how='inner')
    X_pred = doe_formulas_stability.drop(columns=ids).values

    # Generate predictions
    dic_pred = {}
    for tgt in targets_stability:
        y_pred = dic_models[tgt].predict(X_pred) 
        dic_pred[tgt] = y_pred

    Y_pred = pd.DataFrame(dic_pred, columns = list(dic_pred.keys())).round(1)
    doe_formulas_stability = pd.concat((doe_formulas_stability.reset_index(drop=True), Y_pred.reset_index(drop=True)), axis=1)


# ### 1.3 Apply stability acceptance crieteria at  `t=12 weeks`

# In[ ]:


dic_stability_limits = {
    'Play Time (s)': {
        'min': float(plp_globals['playtime'].split(';')[0]), 
        'max': float(plp_globals['playtime'].split(';')[1])
    },
    'Stickiness': {
        'min': float(plp_globals['stickiness'].split(';')[0]), 
        'max': float(plp_globals['stickiness'].split(';')[1])
    },
    'Slippery Finish': {
        'min': float(plp_globals['slipperyfinish'].split(';')[0]), 
        'max': float(plp_globals['slipperyfinish'].split(';')[1])
    },
    'Spreadability': {
        'min': float(plp_globals['spreadability'].split(';')[0]), 
        'max': float(plp_globals['spreadability'].split(';')[1])
    },
    'Viscosity (Pa.s)': {
        'min': float(plp_globals['viscosity'].split(';')[0]), 
        'max': float(plp_globals['viscosity'].split(';')[1])
    },
    'pH': {
        'min': float(plp_globals['ph'].split(';')[0]), 
        'max': float(plp_globals['ph'].split(';')[1])
    },
    'Color': {
        'min': 3, 
        'max': 5
    },
    'Odor': {
        'min': 3, 
        'max': 5
    }
}


# In[ ]:


if plp_globals['stability'] == 'Stable':
    to_drop = []
    for tgt, values in zip (dic_stability_limits.keys(), dic_stability_limits.values()):
        doe_formulas_stability['{} min'.format(tgt)] = values['min']
        doe_formulas_stability['{} max'.format(tgt)] = values['max']

        cond = (doe_formulas_stability[tgt] < values['min']) & (doe_formulas_stability[tgt] >= values['max'])
        to_drop += list(doe_formulas_stability.loc[cond, 'FormulaID'].values)

    doe_formulas_stability.drop(
        index=list(doe_formulas_stability.loc[doe_formulas_stability['FormulaID'].isin(to_drop), :].index), 
        inplace=True
    ) 


doe_formulas_stability.to_csv(
    './Data/generated/stability/generated_anti_ageing_cream_stability_test.csv',
    index=False
)


# In[ ]:


doe_formulas_microbio = pd.read_csv(
    './Data/generated/microbiology/generated_anti_ageing_cream_microbiology_challenge_test.csv'
)

doe_formulas_microbio.drop(
    index=doe_formulas_microbio.loc[doe_formulas_microbio['FormulaID'].isin(to_drop), :].index,
    inplace=True
)

doe_formulas_microbio.to_csv(
    './Data/generated/microbiology/generated_anti_ageing_cream_microbiology_challenge_test.csv',
    index=False
)


# In[ ]:


plp_df = plp_df.loc[plp_df['FormulaID'].isin(doe_formulas_stability['FormulaID'].unique()),:].drop(columns=['formula_id'])

