import pandas as pd
import numpy as np

def engineer_basic_chocolate_features(df):
    """
    Compute basic engineered features for chocolate formulations dataset
    These are composition-based, nutritional, and categorical features that can be calculated immediately.
    
    Parameters:
    df (pandas.DataFrame): Original formulations dataset
    
    Returns:
    pandas.DataFrame: Enhanced dataset with basic engineered features
    """
    
    # Create a copy to avoid modifying original data
    enhanced_df = df.copy()
    
    # Helper function for safe calculations (handle NaN/None values)
    def safe_divide(numerator, denominator, default=0):
        """Safe division that handles zero denominators and NaN values"""
        try:
            result = numerator / denominator if denominator != 0 else default
            return result if not np.isnan(result) else default
        except:
            return default
    
    def safe_value(value, default=0):
        """Safe value extraction that handles NaN/None"""
        # try:
        #     return value if not pd.isna(value) else default
        # except:
        #     return default
        return value
    
    print("Computing basic engineered features...")
    
    # =================================================================
    # A. INGREDIENT-BASED ENGINEERED FEATURES
    # =================================================================
    
    print("1. Computing ingredient-based features...")
    
    # Total cocoa content (cocoa beans + cocoa butter)
    enhanced_df['total_cocoa_pct'] = (
        safe_value(enhanced_df['cocoa_beans'], 0) + 
        safe_value(enhanced_df['cocoa_butter'], 0)
    )
    
    # Milk to sugar ratio (key texture predictor)
    enhanced_df['milk_sugar_ratio'] = enhanced_df.apply(
        lambda row: safe_divide(
            safe_value(row['whole_milk_powder'], 0), 
            safe_value(row['beetroot_sugar'], 1)
        ), axis=1
    )
    
    # Total flavor intensity (vanilla + natural flavors)
    enhanced_df['flavor_intensity'] = (
        safe_value(enhanced_df['artificial_vanilla_aroma'], 0) +
        safe_value(enhanced_df['natural_vanilla_aroma'], 0) +
        safe_value(enhanced_df['natural_flavor'], 0)
    )
    
    # =================================================================
    # B. NUTRITIONAL-BASED ENGINEERED FEATURES  
    # =================================================================
    
    print("2. Computing nutritional-based features...")
    
    # Unsaturated fat content
    enhanced_df['unsaturated_fat_pct'] = (
        safe_value(enhanced_df['fat'], 0) - 
        safe_value(enhanced_df['SFA'], 0)
    )
    
    # Fat quality index (ratio of saturated to total fat)
    enhanced_df['fat_quality_index'] = enhanced_df.apply(
        lambda row: safe_divide(
            safe_value(row['SFA'], 0), 
            safe_value(row['fat'], 1)
        ), axis=1
    )
    
    # Sweetness ratio (sugar relative to sugar + cocoa)
    enhanced_df['sweetness_ratio'] = enhanced_df.apply(
        lambda row: safe_divide(
            safe_value(row['beetroot_sugar'], 0), 
            safe_value(row['beetroot_sugar'], 0) + safe_value(row['total_cocoa_pct'], 1)
        ), axis=1
    )
    
    # =================================================================
    # C. CATEGORICAL/CLASSIFICATION FEATURES
    # =================================================================
    
    print("3. Computing categorical features...")
    
    # Classic milk identification
    enhanced_df['is_classic_milk'] = enhanced_df['Flavor'].apply(
        lambda x: 1 if str(x).strip() == 'Classic Milk' else 0
    )
    
    # Fruit flavor identification
    fruit_flavors = ['Mango', 'Cherry', 'Blackberry', 'Strawberry', 'Orange', 'Passion Fruit']
    enhanced_df['is_fruit_flavor'] = enhanced_df['Flavor'].apply(
        lambda x: 1 if str(x).strip() in fruit_flavors else 0
    )
    
    # Flavor complexity (has natural flavors)
    enhanced_df['flavor_complexity'] = enhanced_df['natural_flavor'].apply(
        lambda x: 1 if safe_value(x, 0) > 0 else 0
    )
    
    # Formulation type
    enhanced_df['formulation_type'] = enhanced_df['Flavor'].apply(
        lambda x: 'Classic' if str(x).strip() == 'Classic Milk' else 'Flavored'
    )
    
    # =================================================================
    # D. ADVANCED DERIVED FEATURES
    # =================================================================
    
    print("4. Computing advanced derived features...")
    
    # Cocoa intensity (normalized)
    enhanced_df['cocoa_intensity'] = enhanced_df['total_cocoa_pct'] / 100.0
    
    # Estimated dairy content (milk powder + protein equivalent)
    enhanced_df['dairy_content'] = (
        safe_value(enhanced_df['whole_milk_powder'], 0) + 
        safe_value(enhanced_df['protein'], 0) * 2  # Protein factor approximation
    )
    
    # Premium score (cocoa quality + natural ingredients)
    enhanced_df['premium_score'] = enhanced_df.apply(
        lambda row: (
            safe_value(row['total_cocoa_pct'], 0) + 
            safe_value(row['natural_flavor'], 0) * 100 + 
            safe_value(row['natural_vanilla_aroma'], 0) * 100
        ) / 50.0, axis=1
    )
    
    # =================================================================
    # E. PROCESSING/TEXTURE PREDICTION HELPERS
    # =================================================================
    
    print("5. Computing processing prediction features...")
    
    # Expected hardness index (based on fat and sugar content)
    enhanced_df['expected_hardness_index'] = enhanced_df.apply(
        lambda row: (
            safe_value(row['beetroot_sugar'], 0) * 0.4 + 
            safe_value(row['SFA'], 0) * 0.6
        ) / 100.0, axis=1
    )
    
    # Processing difficulty score
    enhanced_df['processing_difficulty'] = enhanced_df.apply(
        lambda row: safe_divide(
            safe_value(row['protein'], 0) + safe_value(row['whole_milk_powder'], 0) * 0.2,
            safe_value(row['fat'], 1)
        ), axis=1
    )
    
    # Melting prediction index (based on fat composition)
    enhanced_df['melting_prediction_index'] = enhanced_df.apply(
        lambda row: (
            safe_value(row['cocoa_butter'], 0) * 0.7 + 
            safe_value(row['SFA'], 0) * 0.3
        ) / 100.0, axis=1
    )
    
    # =================================================================
    # F. INTERACTION FEATURES
    # =================================================================
    
    print("6. Computing interaction features...")
    
    # Sweet-creamy balance
    enhanced_df['sweet_creamy_balance'] = enhanced_df.apply(
        lambda row: safe_value(row['milk_sugar_ratio'], 0) * safe_value(row['sweetness_ratio'], 0), 
        axis=1
    )
    
    # Cocoa-dairy interaction
    enhanced_df['cocoa_dairy_interaction'] = enhanced_df.apply(
        lambda row: safe_value(row['total_cocoa_pct'], 0) * safe_value(row['dairy_content'], 0) / 1000.0, 
        axis=1
    )
    
    # Flavor-quality interaction
    enhanced_df['flavor_quality_interaction'] = enhanced_df.apply(
        lambda row: safe_value(row['flavor_intensity'], 0) * safe_value(row['premium_score'], 0) * 10, 
        axis=1
    )
    
    print("Basic feature engineering completed!")
    print(f"New basic features added: {enhanced_df.shape[1] - df.shape[1]}")
    
    return enhanced_df

def engineer_shelf_life_features(df):
    """
    Add Phase 1 engineered features for shelf life prediction to the chocolate formulations dataset
    These are stability-focused features that predict degradation and shelf life.
    
    Parameters:
    df (pandas.DataFrame): Dataset with basic engineered features already computed
    
    Returns:
    pandas.DataFrame: Enhanced dataset with Phase 1 shelf life features
    """
    
    # Create a copy to avoid modifying original data
    enhanced_df = df.copy()
    
    # Helper function for safe calculations
    def safe_divide(numerator, denominator, default=0):
        try:
            result = numerator / denominator if denominator != 0 else default
            return result if not np.isnan(result) else default
        except:
            return default
    
    def safe_value(value, default=0):
        try:
            return value if not pd.isna(value) else default
        except:
            return default
    
    print("Adding Phase 1 Shelf Life Engineered Features...")
    
    # ================================================================
    # PHASE 1 FEATURE 1: OXIDATION SUSCEPTIBILITY INDICES (CRITICAL)
    # ================================================================
    
    print("1. Computing oxidation susceptibility features...")
    
    # Processing intensity factor (normalized) - using defaults if processing params not available
    enhanced_df['processing_intensity_factor'] = enhanced_df.apply(
        lambda row: (
            (safe_value(row.get('conching_temperature_c', 75), 75) - 60) * 
            safe_value(row.get('conching_time_min', 180), 180)
        ) / 10000, axis=1  # Normalized to 0-15 range
    )
    
    # Fat oxidation vulnerability - strongest shelf life predictor
    enhanced_df['oxidation_risk_index'] = enhanced_df.apply(
        lambda row: (
            safe_value(row['unsaturated_fat_pct'], 0) / 
            (safe_value(row['fat_quality_index'], 0.6) + 0.01)
        ) * (1 + safe_value(row['processing_intensity_factor'], 0.5)), axis=1
    )
    
    # Fat stability score (higher = more stable)
    enhanced_df['fat_stability_score'] = enhanced_df.apply(
        lambda row: safe_divide(
            safe_value(row['SFA'], 0), 
            safe_value(row['unsaturated_fat_pct'], 1) + 0.01
        ), axis=1
    )
    
    # Thermal stress index during processing
    enhanced_df['thermal_stress_index'] = enhanced_df.apply(
        lambda row: (
            safe_value(row.get('conching_temperature_c', 75), 75) * 
            safe_value(row.get('conching_time_min', 180), 180)
        ) / (safe_value(row['fat_quality_index'], 0.6) * 10000), axis=1  # Normalized
    )
    
    # ================================================================
    # PHASE 1 FEATURE 2: MOISTURE ACTIVITY PREDICTORS (HIGH IMPACT)
    # ================================================================
    
    print("2. Computing moisture activity predictors...")
    
    # Water activity estimation from composition
    enhanced_df['moisture_risk_index'] = enhanced_df.apply(
        lambda row: (
            safe_value(row['dairy_content'], 0) * 0.4 + 
            safe_value(row['Total_sugars'], 0) * 0.3 + 
            safe_value(row['protein'], 0) * 0.3
        ) / 10, axis=1  # Normalized to 0-20 range
    )
    
    # Hygroscopic ingredient score (attracts moisture)
    enhanced_df['hygroscopic_ingredient_score'] = enhanced_df.apply(
        lambda row: (
            safe_value(row['beetroot_sugar'], 0) + 
            safe_value(row['whole_milk_powder'], 0) + 
            safe_value(row['soy_lecithin'], 0) * 10  # Lecithin is highly hygroscopic
        ), axis=1
    )
    
    # Water binding capacity
    enhanced_df['water_binding_capacity'] = enhanced_df.apply(
        lambda row: (
            safe_value(row['protein'], 0) * 4 + 
            safe_value(row['Total_sugars'], 0) * 0.8
        ), axis=1  # Protein binds 4x more water than sugar
    )
    
    # ================================================================
    # PHASE 1 FEATURE 3: CRYSTAL STRUCTURE STABILITY (MEDIUM-HIGH IMPACT)
    # ================================================================
    
    print("3. Computing crystal structure stability features...")
    
    # Fat bloom susceptibility
    enhanced_df['bloom_susceptibility'] = enhanced_df.apply(
        lambda row: (
            safe_divide(safe_value(row['cocoa_butter'], 0), safe_value(row['fat'], 1) + 0.01)
        ) * (1 - safe_value(row['fat_quality_index'], 0.6)) * 10, axis=1  # Higher = more susceptible
    )
    
    # Fat migration risk (incompatible fats)
    enhanced_df['fat_migration_risk'] = enhanced_df.apply(
        lambda row: safe_divide(
            safe_value(row.get('palm_oil', 0), 0), 
            safe_value(row['cocoa_butter'], 1) + 0.01
        ) * 5, axis=1  # Normalized
    )
    
    # Crystal stability index (higher = more stable)
    tempering_quality_proxy = 1.0  # Assume standard tempering for all samples
    enhanced_df['crystal_stability_index'] = enhanced_df.apply(
        lambda row: (
            tempering_quality_proxy * safe_value(row.get('cooling_rate_c_per_min', 2.0), 2.0)
        ) / (1 + safe_value(row['fat_migration_risk'], 0)), axis=1
    )
    
    # ================================================================
    # PHASE 1 FEATURE 4: CHEMICAL REACTIVITY INDICES (MEDIUM IMPACT)
    # ================================================================
    
    print("4. Computing chemical reactivity indices...")
    
    # Maillard reaction potential (browning, off-flavors)
    enhanced_df['maillard_risk'] = enhanced_df.apply(
        lambda row: (
            safe_value(row['protein'], 0) * safe_value(row['Total_sugars'], 0)
        ) / 100, axis=1  # Normalized
    )
    
    # Amino-sugar interaction potential
    enhanced_df['amino_sugar_interaction'] = enhanced_df.apply(
        lambda row: (
            safe_value(row['whole_milk_powder'], 0) * safe_value(row['beetroot_sugar'], 0)
        ) / 100, axis=1
    )
    
    # Browning potential index
    enhanced_df['browning_potential'] = enhanced_df.apply(
        lambda row: (
            safe_value(row['protein'], 0) * 0.6 + 
            safe_value(row['Total_sugars'], 0) * 0.4
        ) / 10, axis=1  # Normalized
    )
    
    # ================================================================
    # PHASE 1 FEATURE 5: PROCESSING STRESS ACCUMULATION (MEDIUM IMPACT)
    # ================================================================
    
    print("5. Computing processing stress features...")
    
    # Thermal damage index
    enhanced_df['thermal_damage_index'] = enhanced_df.apply(
        lambda row: (
            safe_value(row.get('conching_temperature_c', 75), 75) * 
            safe_value(row.get('conching_time_min', 180), 180) * 
            safe_value(row.get('grinding_fineness_micron', 20), 20)
        ) / 1000000, axis=1  # Normalized
    )
    
    # Mechanical stress accumulation
    enhanced_df['mechanical_stress'] = enhanced_df.apply(
        lambda row: (
            safe_value(row.get('mixing_speed_rpm', 30), 30) * 
            safe_value(row.get('refining_passes', 3), 3) * 
            safe_value(row.get('conching_time_min', 180), 180)
        ) / 10000, axis=1  # Normalized
    )
    
    # Process severity score
    enhanced_df['process_severity_score'] = enhanced_df.apply(
        lambda row: (
            safe_value(row['thermal_damage_index'], 0) + safe_value(row['mechanical_stress'], 0)
        ) / 2, axis=1
    )
    
    # ================================================================
    # PHASE 1 FEATURE 6: COMPOSITE INDICES (HIGHEST IMPACT)
    # ================================================================
    
    print("6. Computing composite stability indices...")
    
    # Composite stability index - weighted combination of top risk factors
    enhanced_df['composite_stability_index'] = enhanced_df.apply(
        lambda row: (
            safe_value(row['oxidation_risk_index'], 0) * 0.4 +          # Highest weight
            safe_value(row['moisture_risk_index'], 0) * 0.3 +           # Second weight
            safe_value(row['bloom_susceptibility'], 0) * 0.2 +          # Third weight  
            safe_value(row['process_severity_score'], 0) * 0.1          # Fourth weight
        ), axis=1
    )
    
    # Overall stability score (inverted - higher = more stable)
    enhanced_df['overall_stability_score'] = enhanced_df.apply(
        lambda row: 100 / (safe_value(row['composite_stability_index'], 1) + 1), axis=1
    )
    
    # ================================================================
    # PHASE 1 FEATURE 7: PROTECTIVE FACTORS (STABILITY ENHANCERS)
    # ================================================================
    
    print("7. Computing protective factors...")
    
    # Natural antioxidant content (protective)
    enhanced_df['antioxidant_protection'] = enhanced_df.apply(
        lambda row: (
            safe_value(row['cocoa_beans'], 0) * 0.8 +           # Cocoa has natural antioxidants
            safe_value(row['natural_flavor'], 0) * 20 +         # Natural flavors often have antioxidants
            safe_value(row['natural_vanilla_aroma'], 0) * 20    # Vanilla has antioxidant properties
        ), axis=1
    )
    
    # Fat crystal protection (pure cocoa butter is most stable)
    enhanced_df['fat_crystal_protection'] = enhanced_df.apply(
        lambda row: (
            safe_value(row['cocoa_butter'], 0) / 
            (safe_value(row['cocoa_butter'], 0) + safe_value(row.get('palm_oil', 0), 0) + 0.01)
        ) * 10, axis=1
    )
    
    # Emulsifier stabilization effect
    enhanced_df['emulsifier_stabilization'] = enhanced_df.apply(
        lambda row: safe_value(row['soy_lecithin'], 0) * 20, axis=1  # Lecithin protects fat-water interfaces
    )
    
    # Combined protective effect
    enhanced_df['total_protective_effect'] = enhanced_df.apply(
        lambda row: (
            safe_value(row['antioxidant_protection'], 0) + 
            safe_value(row['fat_crystal_protection'], 0) + 
            safe_value(row['emulsifier_stabilization'], 0)
        ) / 3, axis=1
    )
    
    # ================================================================
    # PHASE 1 FEATURE 8: FAILURE MODE PREDICTORS (TARGETED)
    # ================================================================
    
    print("8. Computing failure mode predictors...")
    
    # Fat bloom predictor (specific to visual defects)
    enhanced_df['fat_bloom_predictor'] = enhanced_df.apply(
        lambda row: (
            safe_value(row['bloom_susceptibility'], 0) * 
            (1 + safe_value(row['thermal_stress_index'], 0)) *
            (1 - safe_value(row['fat_crystal_protection'], 0) / 10)
        ), axis=1
    )
    
    # Color degradation predictor (browning/fading)
    enhanced_df['color_degradation_predictor'] = enhanced_df.apply(
        lambda row: (
            safe_value(row['maillard_risk'], 0) * 
            (1 + safe_value(row['processing_intensity_factor'], 0))
        ), axis=1
    )
    
    # Texture failure predictor (hardening/softening)
    enhanced_df['texture_failure_predictor'] = enhanced_df.apply(
        lambda row: (
            safe_value(row['moisture_risk_index'], 0) * 
            safe_value(row['hygroscopic_ingredient_score'], 0) / 10
        ), axis=1
    )
    
    # Flavor degradation predictor (rancidity/off-flavors)
    enhanced_df['flavor_degradation_predictor'] = enhanced_df.apply(
        lambda row: (
            safe_value(row['oxidation_risk_index'], 0) * 
            (1 - safe_value(row['antioxidant_protection'], 0) / 50)
        ), axis=1
    )
    
    # ================================================================
    # PHASE 1 FEATURE 9: INTERACTION TERMS (SYNERGISTIC EFFECTS)
    # ================================================================
    
    print("9. Computing interaction terms...")
    
    # Moisture-fat interaction (amplifies degradation)
    enhanced_df['moisture_fat_interaction'] = enhanced_df.apply(
        lambda row: (
            safe_value(row['moisture_risk_index'], 0) * safe_value(row['oxidation_risk_index'], 0)
        ) / 10, axis=1
    )
    
    # Temperature-composition interaction
    enhanced_df['temperature_composition_interaction'] = enhanced_df.apply(
        lambda row: (
            safe_value(row['process_severity_score'], 0) * 
            (safe_value(row['fat_migration_risk'], 0) + safe_value(row['maillard_risk'], 0))
        ), axis=1
    )
    
    # Protein-sugar browning interaction
    enhanced_df['protein_sugar_browning'] = enhanced_df.apply(
        lambda row: (
            safe_value(row['protein'], 0) * safe_value(row['Total_sugars'], 0) * 
            safe_value(row['thermal_stress_index'], 0)
        ) / 100, axis=1
    )
    
    # ================================================================
    # SUMMARY STATISTICS AND VALIDATION
    # ================================================================
    
    print("10. Computing summary statistics...")
    
    # Count of new features added
    new_features = [col for col in enhanced_df.columns if col not in df.columns]
    
    print(f"Phase 1 Shelf Life Feature Engineering Complete!")
    print(f"Original features: {len(df.columns)}")
    print(f"New shelf life features added: {len(new_features)}")
    print(f"Total features: {len(enhanced_df.columns)}")
    
    # Display feature ranges for validation
    print(f"Key Feature Ranges:")
    key_features = [
        'oxidation_risk_index', 'moisture_risk_index', 'bloom_susceptibility', 
        'composite_stability_index', 'overall_stability_score'
    ]
    
    for feature in key_features:
        min_val = enhanced_df[feature].min()
        max_val = enhanced_df[feature].max()
        mean_val = enhanced_df[feature].mean()
        print(f"{feature}: {min_val:.3f} - {max_val:.3f} (mean: {mean_val:.3f})")
    
    return enhanced_df

# =================================================================
# MAIN PROCESSING FUNCTION
# =================================================================

def process_complete_feature_engineering(df):
    """
    Complete feature engineering pipeline combining basic and shelf life features
    
    Parameters:
    df (pandas.DataFrame): Original formulations dataset
    
    Returns:
    pandas.DataFrame: Fully enhanced dataset with all engineered features
    """
    
    print("COMPLETE CHOCOLATE FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    
    # Step 1: Basic features
    enhanced_df = engineer_basic_chocolate_features(df)
    
    print("\n" + "=" * 40)
    
    # Step 2: Shelf life features
    enhanced_df = engineer_shelf_life_features(enhanced_df)
    
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE!")
    print(f"Original features: {df.shape[1]}")
    print(f"Total features after engineering: {enhanced_df.shape[1]}")
    print(f"New features added: {enhanced_df.shape[1] - df.shape[1]}")
    
    return enhanced_df