import joblib
import pandas as pd


#'/home/data/RTV/DEMOR424/BIOVIA/BPP2022/public/cpgretail/stability_analysis/exported chanel folder/Penetration (mm).sav'
file_path_candila = '/home/data/RTV/DEMOR424/BIOVIA/BPP2022/public/cpgretail/microbio_analysis/Candida albicans CFU per g.sav'
file_path_escherichia = '/home/data/RTV/DEMOR424/BIOVIA/BPP2022/public/cpgretail/microbio_analysis/Escherichia coli  CFU per g.sav'
file_path_pseudomonas = '/home/data/RTV/DEMOR424/BIOVIA/BPP2022/public/cpgretail/microbio_analysis/Pseudomonas aeruginosa CFU per g.sav'
file_path_staphylococcus = '/home/data/RTV/DEMOR424/BIOVIA/BPP2022/public/cpgretail/microbio_analysis/Staphylococcus aureus CFU per g.sav'
file_path_total = '/home/data/RTV/DEMOR424/BIOVIA/BPP2022/public/cpgretail/microbio_analysis/Total Bacteria Count CFU per g.sav'


list_models = []
attributes = [
    'Candida albicans CFU per g', 
    'Escherichia coli CFU per g',
    'Pseudomonas aeruginosa CFU per g', 
    'Staphylococcus aureus CFU per g', 
    'Total Bacteria Count CFU per g'
]

files = [file_path_candila, file_path_escherichia, file_path_pseudomonas, file_path_staphylococcus, file_path_total]

to_drop = ['Formula', 'recipeName2']
for att, file in zip(attributes, files):
    model = joblib.load(file)
    plp_df.loc[:,att] = model.predict(plp_df.drop(columns = to_drop).to_numpy()).round(2)
    to_drop.append(att)
    list_models.append((att, model))
        