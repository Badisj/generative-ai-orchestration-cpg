import joblib
import pandas as pd
import sklearn


file_path_penetration = '/home/data/RTV/DEMOR424/BIOVIA/BPP2022/public/cpgretail/stability_analysis/exported chanel folder/Penetration (mm).sav'
file_path_breaking = '/home/data/RTV/DEMOR424/BIOVIA/BPP2022/public/cpgretail/stability_analysis/exported chanel folder/Breaking Point (g).sav'
file_path_melting = '/home/data/RTV/DEMOR424/BIOVIA/BPP2022/public/cpgretail/stability_analysis/exported chanel folder/Melt Point ©.sav'
file_path_sweating = '/home/data/RTV/DEMOR424/BIOVIA/BPP2022/public/cpgretail/stability_analysis/exported chanel folder/Sweating.sav'
file_path_colorfade = '/home/data/RTV/DEMOR424/BIOVIA/BPP2022/public/cpgretail/stability_analysis/exported chanel folder/color Fade.sav'


list_models = []
attributes = ['Penetration', 'Breaking Point', 'Melting Point', 'Sweating', 'Color Fade']
files = [file_path_penetration, file_path_breaking, file_path_melting, file_path_sweating, file_path_colorfade]

plp_df.loc[:,'Predicted'] = 0
for att, file in zip(attributes, files):
    if att in plp_df.Attributes.values:
        model = joblib.load(file)
        plp_df.loc[plp_df.Attributes == att,'Predicted'] = model.predict(plp_df.loc[plp_df.Attributes == att,:].drop(columns = ['Attributes', 'Formula', 'Predicted']).to_numpy()).round(2)
        list_models.append((att, model))




#plp_df = pd.DataFrame(pickled_model.predict(plp_df.drop(columns = ['Attributes', 'Formula']).to_numpy() )) 


#plp_df = pd.DataFrame([sklearn.__version__], index = [0])

plp_globals['attributes'] = plp_df.Attributes