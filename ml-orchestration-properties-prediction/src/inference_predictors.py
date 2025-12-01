import pandas as pd
import numpy as np
import argparse
import warnings
from typing import Dict, List, Optional



FEATURES = [
    'Stabilizer',
    'SKIMMED_MILK_POWDER',
    'VITAMIN_A_POWDER',
    'VITAMIN_D3_POWDER',
    'AMF',
    'CULTURE',
    'RO_Water_SFG',
    'Past_Milk_Cow_FF',
    'FROZEN_CREAM',
    'WHOLE_MILK_POWDER',
    'Red_Color',
    'Strawberry_Flavouring',
    'Sugar_Crystal_ICUMSA_45',
    'Flavour_Chocolate',
    'COMPOUND_COCOA',
    'EMULSIFIER',
    'Past_Milk_Cow_SKM',
    'Heat_Temperature_Min_C',
    'Heat_Temperature_Max_C',
    'Homogenization_Pressure_Primary_Bar',
    'Homogenization_Pressure_Secondary_Bar',
    'Homogenization_Temperature_Min_C',
    'Homogenization_Temperature_Max_C',
    'Outlet_Temperature_Min_C',
    'Outlet_Temperature_Max_C',
    'Flow_Rate_KL',
    'protein_total',
    'casein_equivalent',
    'whey_equivalent',
    'casein_whey_ratio',
    'milk_powder_total',
    'fresh_milk_total',
    'fresh_vs_powder_ratio',
    'protein_source_diversity',
    'fat_from_AMF',
    'fat_from_cream',
    'fat_from_powder',
    'fat_from_fresh_milk',
    'fat_total',
    'added_fat_ratio',
    'natural_vs_added_fat',
    'fat_source_diversity',
    'lactose_natural',
    'sugar_added',
    'sugar_total',
    'added_vs_natural_sugar',
    'sugar_loading',
    'fermentable_substrate',
    'is_strawberry',
    'is_chocolate',
    'is_plain',
    'flavor_intensity_total',
    'base_color_L',
    'base_color_a',
    'base_color_b',
    'color_L_predicted',
    'color_a_predicted',
    'color_b_predicted',
    'strawberry_harmony',
    'chocolate_harmony',
    'strawberry_sugar_interaction',
    'chocolate_sugar_interaction',
    'water_added',
    'water_from_milk',
    'water_total',
    'stabilizer_to_protein_ratio',
    'emulsifier_to_fat_ratio',
    'texture_modifier_total',
    'stabilizer_emulsifier_balance',
    'protein_water_binding',
    'stabilizer_water_binding',
    'water_binding_competition',
    'pasteurization_intensity',
    'homog_energy_density',
    'pressure_ratio',
    'process_energy_total',
    'initial_globule_size_weighted',
    'fat_globule_size_predicted',
    'protein_aggregation_potential',
    'homog_challenge_index',
    'particle_uniformity_index',
    'protein_network_strength',
    'fat_lubrication_factor',
    'texture_synergy_score',
    'culture_substrate_quality',
    'pH_buffering_capacity',
    'acidification_potential',
    'protein_bitterness_potential',
    'sugar_masking_power',
    'flavor_masking_power',
    'taste_balance_score',
    'emulsion_stability_predictor',
    'vitamin_A_total',
    'vitamin_D3_total',
    'fat_soluble_vitamin_bioavailability',
    'vitamin_fortification_intensity',
    'mineral_content_natural',
    'formulation_complexity',
    'premium_ingredient_score',
    'process_optimization_score'
]


class YogurtFeatureEngineer:
    """
    Feature Engineering Pipeline for Yogurt Manufacturing Data
    
    Transforms raw composition and process data into engineered features
    optimized for ML model training and process optimization.
    """
    
    def __init__(self):
        """Initialize with standard composition factors and process parameters"""
        
        # Protein content factors (% protein by weight)
        self.protein_factors = {
            'SKIMMED_MILK_POWDER': 0.35,
            'WHOLE_MILK_POWDER': 0.25,
            'Past_Milk_Cow_FF': 0.032,
            'Past_Milk_2_4_Fat': 0.032,
            'Past_Milk_Cow_SKM': 0.034,
            'Solid_Milk_Conc_100': 0.35
        }
        
        # Fat content factors (% fat by weight)
        self.fat_factors = {
            'AMF': 0.998,  # Anhydric Milk Fat
            'FROZEN_CREAM': 0.35,
            'WHOLE_MILK_POWDER': 0.26,
            'SKIMMED_MILK_POWDER': 0.01,
            'Past_Milk_Cow_FF': 0.035,
            'Past_Milk_2_4_Fat': 0.024,
            'Past_Milk_Cow_SKM': 0.001
        }
        
        # Lactose content factors (% lactose by weight)
        self.lactose_factors = {
            'Past_Milk_Cow_FF': 0.048,
            'Past_Milk_2_4_Fat': 0.048,
            'Past_Milk_Cow_SKM': 0.049,
            'WHOLE_MILK_POWDER': 0.38,
            'SKIMMED_MILK_POWDER': 0.52,
            'Solid_Milk_Conc_100': 0.38
        }
        
        # Process parameters (fixed values from specification)
        self.process_params = {
            'heat_treatment_temp': 93.5,  # °C (midpoint of 92-95°C)
            'heat_treatment_time': 300,   # seconds
            'Homogenization_Pressure_Primary_Bar': 200,      # bar
            'Homogenization_Pressure_Secondary_Bar': 45,       # bar
            'Homogenization_Temperature_Min_C': 65.5,           # °C (midpoint of 63-68°C)
            'outlet_temp_target': 4.0,    # °C
            'Flow_Rate_KL': 30               # KL
        }
        
        # Casein:whey ratios for different sources
        self.casein_ratios = {
            'powder_sources': 0.82,    # Milk powders
            'fresh_milk': 0.78,        # Fresh milk sources
            'concentrate': 0.82        # Milk concentrate
        }
        
    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to match expected format
        Handles variations in naming conventions
        """
        column_mapping = {
            'VITAMIN A POWDER Palmitate CWS GF': 'VITAMIN_A_POWDER',
            'VITAMIN D3 POWDER CWS Food Grade': 'VITAMIN_D3_POWDER',
            'RO Water (SFG)': 'RO_Water_SFG',
            'Past. Milk Cow FF': 'Past_Milk_Cow_FF',
            'Past. Milk at 2.4% Fat for Mixing': 'Past_Milk_2_4_Fat',
            'Past. Milk Cow SKM': 'Past_Milk_Cow_SKM',
            'Sugar Crystal ICUMSA 45': 'Sugar_Crystal_ICUMSA_45',
            'Strawberry Flavouring': 'Strawberry_Flavouring',
            'Flavour Chocolate': 'Flavour_Chocolate',
            'COMPOUND COCOA': 'COMPOUND_COCOA',
            'Solid Milk Conc 100% AUSFIN (SFG)': 'Solid_Milk_Conc_100',
            'SKIMMED MILK POWDER': 'SKIMMED_MILK_POWDER',
            'WHOLE MILK POWDER': 'WHOLE_MILK_POWDER',
            'FROZEN CREAM': 'FROZEN_CREAM',
            'RO Water': 'RO_Water',
            'Red Color': 'Red_Color'

        }
        
        # Apply column name mapping if columns exist
        df_renamed = df.rename(columns=column_mapping)
        
        # Fill missing ingredient columns with zeros (not used in this batch)
        expected_columns = list(column_mapping.values()) + ['Stabilizer', 'AMF', 'CULTURE', 'EMULSIFIER']
        for col in expected_columns:
            if col not in df_renamed.columns:
                df_renamed[col] = 0.0
                
        return df_renamed
    
    def engineer_protein_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer protein system features"""
        
        # Total protein concentration from all sources
        df['protein_total'] = (
            df['SKIMMED_MILK_POWDER'] * self.protein_factors['SKIMMED_MILK_POWDER'] +
            df['WHOLE_MILK_POWDER'] * self.protein_factors['WHOLE_MILK_POWDER'] +
            df['Past_Milk_Cow_FF'] * self.protein_factors['Past_Milk_Cow_FF'] +
            df['Past_Milk_2_4_Fat'] * self.protein_factors['Past_Milk_2_4_Fat'] +
            df['Past_Milk_Cow_SKM'] * self.protein_factors['Past_Milk_Cow_SKM'] +
            df['Solid_Milk_Conc_100'] * self.protein_factors['Solid_Milk_Conc_100']
        )
        
        # Casein equivalent calculation
        df['casein_equivalent'] = (
            # From powder sources (higher casein ratio)
            (df['SKIMMED_MILK_POWDER'] * self.protein_factors['SKIMMED_MILK_POWDER'] * 
             self.casein_ratios['powder_sources']) +
            (df['WHOLE_MILK_POWDER'] * self.protein_factors['WHOLE_MILK_POWDER'] * 
             self.casein_ratios['powder_sources']) +
            (df['Solid_Milk_Conc_100'] * self.protein_factors['Solid_Milk_Conc_100'] * 
             self.casein_ratios['concentrate']) +
            # From fresh milk sources  
            (df['Past_Milk_Cow_FF'] * self.protein_factors['Past_Milk_Cow_FF'] * 
             self.casein_ratios['fresh_milk']) +
            (df['Past_Milk_2_4_Fat'] * self.protein_factors['Past_Milk_2_4_Fat'] * 
             self.casein_ratios['fresh_milk']) +
            (df['Past_Milk_Cow_SKM'] * self.protein_factors['Past_Milk_Cow_SKM'] * 
             self.casein_ratios['fresh_milk'])
        )
        
        # Whey protein equivalent
        df['whey_equivalent'] = df['protein_total'] - df['casein_equivalent']
        
        # Ratios and diversity metrics
        df['casein_whey_ratio'] = df['casein_equivalent'] / (df['whey_equivalent'] + 0.001)
        
        # Protein source quantities
        df['milk_powder_total'] = df['SKIMMED_MILK_POWDER'] + df['WHOLE_MILK_POWDER']
        df['fresh_milk_total'] = (df['Past_Milk_Cow_FF'] + df['Past_Milk_2_4_Fat'] + 
                                 df['Past_Milk_Cow_SKM'])
        
        # Source ratios
        df['fresh_vs_powder_ratio'] = df['fresh_milk_total'] / (df['milk_powder_total'] + 0.001)
        df['concentrate_ratio'] = df['Solid_Milk_Conc_100'] / (df['protein_total'] + 0.001)
        
        # Protein source diversity index
        protein_sources_used = (
            (df['SKIMMED_MILK_POWDER'] > 0).astype(int) +
            (df['WHOLE_MILK_POWDER'] > 0).astype(int) +
            (df['Past_Milk_Cow_FF'] > 0).astype(int) +
            (df['Past_Milk_2_4_Fat'] > 0).astype(int) +
            (df['Past_Milk_Cow_SKM'] > 0).astype(int) +
            (df['Solid_Milk_Conc_100'] > 0).astype(int)
        )
        df['protein_source_diversity'] = protein_sources_used / 6
        
        return df
    
    def engineer_fat_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer fat system features"""
        
        # Fat from different sources
        df['fat_from_AMF'] = df['AMF'] * self.fat_factors['AMF']
        df['fat_from_cream'] = df['FROZEN_CREAM'] * self.fat_factors['FROZEN_CREAM']
        
        df['fat_from_powder'] = (
            df['WHOLE_MILK_POWDER'] * self.fat_factors['WHOLE_MILK_POWDER'] +
            df['SKIMMED_MILK_POWDER'] * self.fat_factors['SKIMMED_MILK_POWDER']
        )
        
        df['fat_from_fresh_milk'] = (
            df['Past_Milk_Cow_FF'] * self.fat_factors['Past_Milk_Cow_FF'] +
            df['Past_Milk_2_4_Fat'] * self.fat_factors['Past_Milk_2_4_Fat'] +
            df['Past_Milk_Cow_SKM'] * self.fat_factors['Past_Milk_Cow_SKM']
        )
        
        # Total fat content
        df['fat_total'] = (df['fat_from_AMF'] + df['fat_from_cream'] + 
                          df['fat_from_powder'] + df['fat_from_fresh_milk'])
        
        # Fat source ratios
        df['added_fat_ratio'] = ((df['fat_from_AMF'] + df['fat_from_cream']) / 
                                (df['fat_total'] + 0.001))
        df['natural_vs_added_fat'] = (df['fat_from_fresh_milk'] + df['fat_from_powder']) / (
                                    df['fat_from_AMF'] + df['fat_from_cream'] + 0.001)
        
        # Fat source diversity
        fat_sources_used = (
            (df['AMF'] > 0).astype(int) +
            (df['FROZEN_CREAM'] > 0).astype(int) +
            (df['fat_from_fresh_milk'] > 0).astype(int) +
            (df['fat_from_powder'] > 0).astype(int)
        )
        df['fat_source_diversity'] = fat_sources_used / 4
        
        return df
    
    def engineer_carbohydrate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer sugar and carbohydrate features"""
        
        # Natural lactose from milk sources
        df['lactose_natural'] = (
            df['Past_Milk_Cow_FF'] * self.lactose_factors['Past_Milk_Cow_FF'] +
            df['Past_Milk_2_4_Fat'] * self.lactose_factors['Past_Milk_2_4_Fat'] +
            df['Past_Milk_Cow_SKM'] * self.lactose_factors['Past_Milk_Cow_SKM'] +
            df['WHOLE_MILK_POWDER'] * self.lactose_factors['WHOLE_MILK_POWDER'] +
            df['SKIMMED_MILK_POWDER'] * self.lactose_factors['SKIMMED_MILK_POWDER'] +
            df['Solid_Milk_Conc_100'] * self.lactose_factors['Solid_Milk_Conc_100']
        )
        
        # Added sugar (sucrose)
        df['sugar_added'] = df['Sugar_Crystal_ICUMSA_45']
        
        # Total sugars
        df['sugar_total'] = df['sugar_added'] + df['lactose_natural']
        
        # Sugar ratios
        df['added_vs_natural_sugar'] = df['sugar_added'] / (df['lactose_natural'] + 0.001)
        df['sugar_loading'] = df['sugar_total'] / 100  # Assume 100g base for percentage
        
        # Fermentation substrate availability
        df['fermentable_substrate'] = df['lactose_natural'] * 1.0 + df['sugar_added'] * 0.8
        
        return df
    
    def engineer_flavor_color_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer flavor and color system features"""
        
        # Product variant identification
        df['is_strawberry'] = (df['Strawberry_Flavouring'] > 0).astype(int)
        df['is_chocolate'] = ((df['Flavour_Chocolate'] > 0) | 
                             (df['COMPOUND_COCOA'] > 0)).astype(int)
        df['is_plain'] = ((df['is_strawberry'] == 0) & (df['is_chocolate'] == 0)).astype(int)
        
        # Flavor intensity
        df['flavor_intensity_total'] = (
            df['Strawberry_Flavouring'] * 1.0 +
            df['Flavour_Chocolate'] * 1.0 +
            df['COMPOUND_COCOA'] * 0.6  # Lower flavor intensity than pure flavor
        )
        
        # Color prediction (L*a*b* estimation)
        df['base_color_L'] = 85 + df['fat_total'] * 2  # Fat increases lightness
        df['base_color_a'] = -2 + df['protein_total'] * 0.5  # Slight green to red shift
        df['base_color_b'] = 8 + df['fat_from_powder'] * 0.3  # Natural yellowness
        
        # Color modifications from additives
        df['color_L_predicted'] = (df['base_color_L'] - df['Red_Color'] * 2 - 
                                  df['COMPOUND_COCOA'] * 20)
        df['color_a_predicted'] = (df['base_color_a'] + df['Red_Color'] * 15 + 
                                  df['COMPOUND_COCOA'] * 2)
        df['color_b_predicted'] = (df['base_color_b'] + df['COMPOUND_COCOA'] * 8)
        
        # Flavor-color harmony scores
        df['strawberry_harmony'] = (
            df['is_strawberry'] * (df['Strawberry_Flavouring'] > 0).astype(int) * 
            (df['Red_Color'] > 0).astype(int) * 
            np.minimum(df['Strawberry_Flavouring'] / 10, df['Red_Color'] / 5)
        )
        
        df['chocolate_harmony'] = (
            df['is_chocolate'] * 
            ((df['Flavour_Chocolate'] + df['COMPOUND_COCOA']) > 0).astype(int) *
            (df['color_L_predicted'] < 75).astype(int)  # Sufficient darkening
        )
        
        # Flavor-sugar interactions
        df['strawberry_sugar_interaction'] = df['Strawberry_Flavouring'] * df['sugar_added'] * 0.001
        df['chocolate_sugar_interaction'] = (df['Flavour_Chocolate'] + df['COMPOUND_COCOA']) * df['sugar_added'] * 0.001
        
        return df
    
    def engineer_functional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer functional ingredient features"""
        
        # Water content calculation
        df['water_added'] = df['RO_Water_SFG'] + df['RO_Water']
        df['water_from_milk'] = (
            df['Past_Milk_Cow_FF'] * 0.87 +
            df['Past_Milk_2_4_Fat'] * 0.87 +
            df['Past_Milk_Cow_SKM'] * 0.91
        )
        df['water_total'] = df['water_added'] + df['water_from_milk']
        
        # Functional ingredient ratios
        df['stabilizer_to_protein_ratio'] = df['Stabilizer'] / (df['protein_total'] + 0.001)
        df['emulsifier_to_fat_ratio'] = df['EMULSIFIER'] / (df['fat_total'] + 0.001)
        
        # Combined functionality
        df['texture_modifier_total'] = df['Stabilizer'] + df['EMULSIFIER']
        df['stabilizer_emulsifier_balance'] = df['Stabilizer'] / (df['EMULSIFIER'] + 0.001)
        
        # Water binding competition
        df['protein_water_binding'] = df['protein_total'] * 4  # ~4g water per g protein
        df['stabilizer_water_binding'] = df['Stabilizer'] * 50  # ~50g water per g hydrocolloid
        
        df['water_binding_competition'] = (
            df['protein_water_binding'] + df['stabilizer_water_binding'] + 
            df['sugar_added'] * 0.3  # Sugar affects water activity
        ) / (df['water_total'] + 0.001)
        
        return df
    
    def engineer_process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer process-related features"""
        
        # Heat treatment features
        df['pasteurization_intensity'] = (
            ((df['Heat_Temperature_Min_C'] + df['Heat_Temperature_Max_C'])*0.5 - 72) * 
            df['Heat_Time_s'] / 15
        )
        
        # Homogenization features
        df['homog_energy_density'] = (
            (df['Homogenization_Pressure_Primary_Bar'] + df['Homogenization_Pressure_Secondary_Bar']) * 
            df['Flow_Rate_KL']
        )
        
        df['pressure_ratio'] = (df['Homogenization_Pressure_Primary_Bar'] / 
                               df['Homogenization_Pressure_Secondary_Bar'])
        
        # Process efficiency indicators
        df['process_energy_total'] = (
            df['pasteurization_intensity'] * 0.5 +
            df['homog_energy_density'] * 0.0003 +
            abs((df['Outlet_Temperature_Min_C'] + df['Outlet_Temperature_Max_C'])*0.5 - (df['Homogenization_Temperature_Max_C'] + df['Homogenization_Temperature_Min_C']*0.5 ) * 2
        ))
        
        return df
    
    def engineer_particle_size_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer particle size related features"""
        
        # Initial fat globule size (weighted by fat source)
        df['initial_globule_size_weighted'] = (
            (df['fat_from_AMF'] * 0.8 +     # AMF has small initial globules
             df['fat_from_cream'] * 4.5 +   # Cream has larger globules
             df['fat_from_fresh_milk'] * 3.5 + # Natural milk fat
             df['fat_from_powder'] * 2.0)   # Reconstituted powder fat
        ) / (df['fat_total'] + 0.001)
        
        # Homogenization efficiency
        pressure_effect = (df['Homogenization_Pressure_Primary_Bar'] ** 0.6) * 1.15
        df['fat_globule_size_predicted'] = (
            df['initial_globule_size_weighted'] / pressure_effect *
            (1 + abs(df['Homogenization_Temperature_Min_C'] - 65) * 0.02)
        )
        
        # Protein particle size factors
        df['protein_aggregation_potential'] = (
            df['pasteurization_intensity'] / 400 *  # Heat effect
            df['protein_total'] * 0.02 *            # Concentration effect
            (1 + df['casein_whey_ratio'] * 0.1)     # Casein aggregates more
        )
        
        # Homogenization challenge (harder to homogenize = larger particles)
        df['homog_challenge_index'] = (
            df['FROZEN_CREAM'] * 2.0 +      # Cream is challenging
            df['AMF'] * 0.5 +               # AMF homogenizes easily
            df['Stabilizer'] * 1.5 +        # Can interfere
            df['protein_total'] * 0.8       # Higher protein = more complex
        )
        
        df['particle_uniformity_index'] = (
            df['homog_energy_density'] / (df['homog_challenge_index'] + 100)
        )
        
        return df
    
    def engineer_synergistic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer advanced synergistic interaction features"""
        
        # Protein network strength
        df['protein_network_strength'] = (
            df['casein_equivalent'] * 1.2 +        # Casein stronger networks
            df['whey_equivalent'] * 0.8 +          # Whey contributes
            df['Solid_Milk_Conc_100'] * 1.3       # Concentrated proteins
        )
        
        # Fat lubrication factor
        df['fat_lubrication_factor'] = (
            df['fat_from_AMF'] * 0.9 +             # Pure milk fat, high lubrication
            df['fat_from_cream'] * 1.2 +           # Cream fat with phospholipids
            df['fat_from_fresh_milk'] * 1.0 +      # Natural baseline
            df['fat_from_powder'] * 0.7            # Powder fat less effective
        )
        
        # Texture synergy score
        df['texture_synergy_score'] = (
            df['protein_network_strength'] * 
            df['fat_lubrication_factor'] * 
            (df['Stabilizer'] / 100) * 
            (df['homog_energy_density'] / 10000)
        ) ** 0.25
        
        # Culture-substrate interaction
        df['culture_substrate_quality'] = (
            df['lactose_natural'] * 1.0 +          # Primary substrate
            df['sugar_added'] * 0.3 +              # Secondary substrate
            df['protein_total'] * 0.1              # Nitrogen source
        )
        
        # pH buffering capacity
        df['pH_buffering_capacity'] = (
            df['protein_total'] * 0.3 +            # Protein buffering
            df['lactose_natural'] * 0.05           # Lactose buffering
        )
        
        # Acidification potential
        df['acidification_potential'] = (
            df['CULTURE'] * df['culture_substrate_quality'] / 
            (df['pH_buffering_capacity'] + 0.001)
        )
        
        # Nutritional taste balance
        df['protein_bitterness_potential'] = df['protein_total'] * 0.1
        df['sugar_masking_power'] = df['sugar_added'] * 1.0 + df['lactose_natural'] * 0.3
        df['flavor_masking_power'] = (
            df['Strawberry_Flavouring'] * 1.5 +
            df['Flavour_Chocolate'] * 1.2 +
            df['COMPOUND_COCOA'] * 0.8
        )
        
        df['taste_balance_score'] = (
            (df['sugar_masking_power'] + df['flavor_masking_power']) / 
            (df['protein_bitterness_potential'] + 0.001)
        )
        
        # Emulsion stability
        df['emulsion_stability_predictor'] = (
            df['fat_from_AMF'] * 0.1 +             # AMF creates stable emulsions
            df['fat_from_cream'] * 0.3 +           # Cream has natural emulsifiers
            df['EMULSIFIER'] * 100 +               # Added emulsifier
            df['homog_energy_density'] * 0.0001   # Mechanical stabilization
        )
        
        return df
    
    def engineer_vitamin_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer vitamin and micronutrient features"""
        
        # Vitamin fortification (assuming standard potency factors)
        df['vitamin_A_total'] = df['VITAMIN_A_POWDER'] * 500000  # IU per gram (example)
        df['vitamin_D3_total'] = df['VITAMIN_D3_POWDER'] * 40000  # IU per gram (example)
        
        # Bioavailability enhancement through fat
        df['fat_soluble_vitamin_bioavailability'] = (
            (df['vitamin_A_total'] + df['vitamin_D3_total']) * 
            df['fat_total'] * 0.001  # Fat enhances absorption
        )
        
        # Fortification intensity
        df['vitamin_fortification_intensity'] = (
            (df['vitamin_A_total'] + df['vitamin_D3_total']) / 1000000  # Per million IU
        )
        
        # Natural mineral content (from milk sources)
        df['mineral_content_natural'] = (
            df['protein_total'] * 0.15 +           # Protein sources contain minerals
            df['Solid_Milk_Conc_100'] * 0.08      # Concentrated minerals
        )
        
        return df
    
    def add_quality_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add quality indicator features"""
        
        # Batch complexity score
        total_ingredients = (
            (df['SKIMMED_MILK_POWDER'] > 0).astype(int) +
            (df['WHOLE_MILK_POWDER'] > 0).astype(int) +
            (df['AMF'] > 0).astype(int) +
            (df['FROZEN_CREAM'] > 0).astype(int) +
            (df['Stabilizer'] > 0).astype(int) +
            (df['EMULSIFIER'] > 0).astype(int) +
            (df['Sugar_Crystal_ICUMSA_45'] > 0).astype(int) +
            (df['Strawberry_Flavouring'] > 0).astype(int) +
            (df['Flavour_Chocolate'] > 0).astype(int) +
            (df['COMPOUND_COCOA'] > 0).astype(int)
        )
        df['formulation_complexity'] = total_ingredients / 10
        
        # Premium ingredient indicator
        df['premium_ingredient_score'] = (
            (df['AMF'] > 0).astype(int) * 0.3 +
            (df['FROZEN_CREAM'] > 0).astype(int) * 0.2 +
            (df['vitamin_A_total'] > 0).astype(int) * 0.2 +
            (df['vitamin_D3_total'] > 0).astype(int) * 0.2 +
            (df['Solid_Milk_Conc_100'] > 0).astype(int) * 0.1
        )
        
        # Process optimization score
        df['process_optimization_score'] = (
            (abs(df['pressure_ratio'] - 4.4) < 0.5).astype(int) * 0.3 +  # Optimal pressure ratio
            (df['homog_energy_density'] > 7000).astype(int) * 0.4 +       # Sufficient homogenization
            (df['pasteurization_intensity'] > 400).astype(int) * 0.3      # Adequate heat treatment
        )
        
        return df
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to engineer all features
        
        Args:
            df: Raw dataframe with composition and process data
            
        Returns:
            DataFrame with all engineered features
        """
        
        # Standardize column names
        df_processed = self.standardize_column_names(df.copy())
        
        # Apply all feature engineering steps
        df_processed = self.engineer_protein_features(df_processed)
        df_processed = self.engineer_fat_features(df_processed)
        df_processed = self.engineer_carbohydrate_features(df_processed)
        df_processed = self.engineer_flavor_color_features(df_processed)
        df_processed = self.engineer_functional_features(df_processed)
        df_processed = self.engineer_process_features(df_processed)
        df_processed = self.engineer_particle_size_features(df_processed)
        df_processed = self.engineer_synergistic_features(df_processed)
        df_processed = self.engineer_vitamin_features(df_processed)
        df_processed = self.add_quality_indicators(df_processed)
        
        return df_processed
      
      
# -------------------------------------
# Main entry point for script execution
# -------------------------------------
if __name__ == "__main__":
    # Load input data
    plp_df = pd.read_csv(args.input_csv)
    # Initialize the feature engineer
    feature_engineer = YogurtFeatureEngineer()

    # Engineer all features
    df = plp_df.copy()
    plp_df = feature_engineer.engineer_all_features(df)[FEATURES]