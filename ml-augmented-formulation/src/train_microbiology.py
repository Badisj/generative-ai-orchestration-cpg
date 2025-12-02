import joblib
import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import minmax_scale


data = plp_df.copy()
data.loc[:,'CandeillaWax'] = .0

new_columns = ['Parfum(Fragrance)', 'TocopherolAcetate', 'PropyleneCarbonate',
    'CI42090(Blue1Lake)', 'Tribehenin', 'HydrogenatedCastorOil',
    'StearalkoniumHectorite', 'CI15850(Red7Lake)', 'Mica',
    'AluminumStarchOctenylsuccinate', 'Ethylene/PropyleneCopolymer',
    'Ethylhexylpalmitate', 'JojobaEsters', 'Polymethylsilsesquioxane',
    'IsononylIsononanoate', 'DiisostearylMalate', 'SyntheticWax',
    'CandeillaWax', 'DicaprylylCarbonate', 'Formula', 'Temperature (°C)', 'Evaluation Day',
    'Total Bacteria Count CFU/g', 'Candida albicans CFU/g', 'Escherichia coli  CFU/g', 
    'Pseudomonas aeruginosa CFU/g', 'Staphylococcus aureus CFU/g'
]

data = data[new_columns].copy()


targets = [
    'Total Bacteria Count CFU/g', 
    'Candida albicans CFU/g', 
    'Escherichia coli  CFU/g', 
    'Pseudomonas aeruginosa CFU/g', 
    'Staphylococcus aureus CFU/g'
]
to_drop = targets + ['Formula']


X = data.drop(columns = to_drop).copy()
Y = data[targets].copy()

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = .2, random_state = 2023)
models_reg = [GradientBoostingRegressor(n_estimators = 500, max_depth = 20)]

dic_targets = {}
for trgt in targets:
    y_train, y_test = Y_train[trgt].copy(), Y_test[trgt].copy()
    
    dic_models = {}
    for mdl in models_reg:
        # Train Models
        mdl.fit(X_train, y_train)
        
        # Error
        dic_models[str(mdl)] = mean_absolute_error(y_test, mdl.predict(X_test))
        
        # Save Models
        #filename = plp_globals['UserDir'] + '/Microbio/{}.sav'.format(trgt.replace('/', ' per '))
        filename = plp_globals['SharedPublicDir'] + '/cpgretail/microbio_analysis/{}.sav'.format(trgt.replace('/', ' per '))
        joblib.dump(mdl, filename)
    
    dic_targets[trgt] = dic_models
     
        
plp_df = pd.DataFrame(dic_targets).copy()



