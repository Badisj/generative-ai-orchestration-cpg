#!/usr/bin/env python
# coding: utf-8

# # Cosmetics Generative Formulation
# ## `Anti-ageing cream performance predictions`

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


# ### 1. Organoleptic and pysiochemical properties *`t=0`*

# In[ ]:


os.chdir('/home/data/RTV/DEMOR424/BIOVIA/BPP2022/public/cpgretail/Cosmetics Generative Formulation/')


# In[ ]:


# Ingredient families
ids = ['FormulaID']
base = ['water']
oils = ['grapessed_oil', 'dimetichone']
surfactants = ['cetearyl_alcohol', 'glyceryl_stearate', 'decyl_glucoside', 'polysorbate20']
thickeners = ['xanthan_gum', 'carbomer']
emollients = ['jojoba_oil', 'caprylic_capric_triglycerides']
preservatives = ['phenoxyethanol', 'sodium_benzoate']
actives = ['Retinol_VitaminA', 'HyaluronicAcid', 'VitaminC_AscorbicAcid']
fragrances = ['fragrance']

# Process
process = ['speed_rpm_step3', 'temperature_celsius_step3', 'time_minutes_step4']

# Testing conditions (t=0)
temperature = 25*np.ones((len(plp_df), 1))
freeze_thaw = np.zeros((len(plp_df), 1))
sun_shade = np.zeros((len(plp_df), 1))
evaluation_week = np.zeros((len(plp_df), 1))


# #### 1.1 Load models and run predictions

# In[ ]:


targets = [
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

# Load Models
dic_models = {}
for target in targets:
    model_filename =  './Models/model_gradient_boosting_{}.sav'.format(target)
    dic_models[target] = pickle.load(open(model_filename, 'rb'))


# In[ ]:


if len(plp_df) >= 1: 
    # Prepare data for predictions
    df = plp_df[base +oils + surfactants + thickeners + emollients + preservatives + actives + fragrances + process].values
    X_pred = np.concatenate((temperature, freeze_thaw, sun_shade, evaluation_week, df), axis=1)

    # Generate predictions
    dic_pred = {}
    for tgt in targets:
        y_pred = dic_models[tgt].predict(X_pred) 
        dic_pred[tgt] = y_pred

    Y_pred = pd.DataFrame(dic_pred, columns = list(dic_pred.keys()))
    plp_df = pd.concat((plp_df.reset_index(drop=True), Y_pred.reset_index(drop=True)), axis=1)


# #### 1.2 Apply user constraints

# In[ ]:


# Retrieve globals from parent protocol
playtime = plp_globals['playtime']
stickiness = plp_globals['stickiness']
slipperyfinish = plp_globals['slipperyfinish']
spreadability = plp_globals['spreadability']
viscosity = plp_globals['viscosity']
ph = plp_globals['ph']
color = "3;5"
odor = "3;5"


# In[ ]:


conds = [playtime, stickiness, slipperyfinish, spreadability, viscosity, ph, color, odor]
sub_targets = [
    'Play Time (s)',
    'Stickiness',
    'Slippery Finish',
    'Spreadability',
    'Viscosity (Pa.s)',
    'pH',
    'Color',
    'Odor'
]

if len(plp_df) >= 1:
    dic_cond = {
        '{}'.format(tgt): {
            'min': cond.split(';')[0], 
            'max': cond.split(';')[1]
        } for tgt, cond in zip(sub_targets, conds)
}


# In[ ]:


for tgt, cond in zip(dic_cond.keys(), dic_cond.values()):
    if len(plp_df) >= 1:
        plp_df = plp_df.loc[(plp_df[tgt] >= float(cond['min'])) & (plp_df[tgt] <= float(cond['max'])), :]
    else:
        plp_df = pd.DataFrame(['No formulas matching the desired criteria'], index=[0])
        break

