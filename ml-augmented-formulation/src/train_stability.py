import joblib
import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import minmax_scale



to_drop = ['Stability Result']
data = plp_df.drop(columns = to_drop)
data.loc[:,'CandeillaWax'] = .0

new_columns = [
    'Parfum(Fragrance)', 'TocopherolAcetate', 'PropyleneCarbonate',
   'CI42090(Blue1Lake)', 'Tribehenin', 'HydrogenatedCastorOil',
   'StearalkoniumHectorite', 'CI15850(Red7Lake)', 'Mica',
   'AluminumStarchOctenylsuccinate', 'Ethylene/PropyleneCopolymer',
   'Ethylhexylpalmitate', 'JojobaEsters', 'Polymethylsilsesquioxane',
   'IsononylIsononanoate', 'DiisostearylMalate', 'SyntheticWax',
   'CandeillaWax', 'DicaprylylCarbonate', 'Formula', 'Temperature (°C)',
   'Freeze/thaw cycle', 'Sunlight/shade cycle', 'Evaluation Week',
   'color Fade', 'Sweating', 'Penetration (mm)', 'Breaking Point (g)','Melt Point ©'
] 

data = data[new_columns].copy()


targets = ['color Fade', 'Sweating', 'Penetration (mm)', 'Breaking Point (g)', 'Melt Point ©']
to_drop = targets + ['Formula']
classif = [True, True, False, False, False]

models_clf = [GradientBoostingClassifier()]
models_reg = [GradientBoostingRegressor()]
require_scale = [False]

X = data.drop(columns = to_drop).copy()

dic_targets = {}
for trgt, cls in zip(targets, classif):
    y = data[trgt].copy()
    if cls:
        models = models_clf
    else: 
        models = models_reg
        
    dic_models = {}
    for mdl, scale in zip(models, require_scale):
        # ---------- Train Models ----------
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 2024)
        mdl.fit(X_train, y_train)
        # ------------------------------------
        
        # ---------- Compute Scores ----------
        if cls:
            dic_models[str(mdl)] = r2_score(y_test, mdl.predict(X_test))
        else:
            dic_models[str(mdl)] = mean_squared_error(y_test, mdl.predict(X_test))
        # ------------------------------------
        
        # ---------- Save Models ----------
        
        filename = plp_globals['SharedPublicDir'] + '/cpgretail/stability_analysis/{}.sav'.format(trgt)
        #filename = './models_stability/model_{}_{}.pkl'.format(trgt, mdl)
        joblib.dump(mdl, filename)
        # ------------------------------------
        
    dic_targets[trgt] = dic_models


#plp_df = pd.concat((X, data[targets]), axis = 1)
plp_df = pd.concat((X, pd.Series(mdl.predict(X)), y.reset_index()), axis = 1)

#plp_df = pd.concat((pd.DataFrame(X), data[targets]), axis = 1)