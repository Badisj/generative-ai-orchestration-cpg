import pandas as pd
import numpy as np
import random

def estimate_chocolate_targets(df):
    """
    Estimate target values for chocolate formulations based on scientific correlations
    and established food science principles.
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Create copy to avoid modifying original
    enhanced_df = df.copy()
    
    print("Estimating target values based on scientific correlations...")
    
    # ================================================================
    # PHYSICAL PROPERTIES ESTIMATION
    # ================================================================
    
    print("1. Estimating physical properties...")
    
    # Hardness (Newtons) - Strong correlation with cocoa content and fat structure
    # Literature: 15-45N range, higher cocoa = harder
    def estimate_hardness(row):
        base_hardness = 18.0
        cocoa_effect = row['total_cocoa_pct'] * 0.35  # Strong correlation R²~0.85
        sugar_hardening = row['beetroot_sugar'] * 0.15  # Sugar crystallization
        fat_effect = row['SFA'] * 0.12  # Saturated fat increases hardness
        processing_effect = row['crystal_stability_index'] * 2.0  # Crystal structure
        
        hardness = base_hardness + cocoa_effect + sugar_hardening + fat_effect + processing_effect
        # Add realistic measurement noise (±8%)
        noise = np.random.normal(0, hardness * 0.08)
        return max(12.0, min(48.0, hardness + noise))
    
    enhanced_df['hardness_newtons'] = enhanced_df.apply(estimate_hardness, axis=1)
    
    # Snap Force (Newtons) - Related to crystal structure and fat quality
    def estimate_snap_force(row):
        base_snap = 10.0
        crystal_effect = row['crystal_stability_index'] * 4.0
        cocoa_butter_effect = row['cocoa_butter'] * 0.25
        fat_quality_effect = row['fat_quality_index'] * 8.0
        
        snap = base_snap + crystal_effect + cocoa_butter_effect + fat_quality_effect
        noise = np.random.normal(0, snap * 0.1)
        return max(6.0, min(28.0, snap + noise))
    
    enhanced_df['snap_force_newtons'] = enhanced_df.apply(estimate_snap_force, axis=1)
    
    # Viscosity (Pa·s) - Processing and fat content dependent
    def estimate_viscosity(row):
        base_viscosity = 3.0
        fat_effect = row['fat'] * 0.08
        processing_effect = row['conching_time_min'] * -0.008  # Longer conching reduces viscosity
        particle_effect = row['grinding_fineness_micron'] * 0.1
        lecithin_effect = row['soy_lecithin'] * -2.0  # Lecithin reduces viscosity
        
        viscosity = base_viscosity + fat_effect + processing_effect + particle_effect + lecithin_effect
        try:
            noise = np.random.normal(0, viscosity * 0.12)
        except:
            noise = np.random.normal(0, -viscosity * 0.12)
        return max(1.5, min(9.0, viscosity + noise))
    
    enhanced_df['viscosity_pas'] = enhanced_df.apply(estimate_viscosity, axis=1)
    
    # Particle Size D50 (microns) - Processing dependent
    def estimate_particle_size(row):
        base_size = row['grinding_fineness_micron']  # Starting point
        refining_effect = row['refining_passes'] * -1.2  # More passes = smaller particles
        conching_effect = row['conching_time_min'] * -0.002  # Slight reduction with time
        
        particle_size = base_size + refining_effect + conching_effect
        noise = np.random.normal(0, 1.5)  # Measurement variation
        return max(10.0, min(30.0, particle_size + noise))
    
    enhanced_df['particle_size_d50_micron'] = enhanced_df.apply(estimate_particle_size, axis=1)
    
    # ================================================================
    # COLOR PROPERTIES ESTIMATION  
    # ================================================================
    
    print("2. Estimating color properties...")
    
    # Color L* (Lightness) - Strong inverse correlation with cocoa content
    def estimate_color_l(row):
        max_lightness = 58.0  # White chocolate range
        cocoa_darkening = row['total_cocoa_pct'] * 0.28  # Strong correlation R²~0.90
        maillard_darkening = row['maillard_risk'] * 0.8
        
        l_value = max_lightness - cocoa_darkening - maillard_darkening
        noise = np.random.normal(0, 1.2)
        return max(25.0, min(60.0, l_value + noise))
    
    enhanced_df['color_l_value'] = enhanced_df.apply(estimate_color_l, axis=1)
    
    # Color a* (Red-Green) - Maillard reactions and cocoa content
    def estimate_color_a(row):
        base_a = 8.0
        maillard_effect = row['maillard_risk'] * 0.15
        cocoa_effect = row['total_cocoa_pct'] * 0.05
        dairy_effect = row['dairy_content'] * 0.02
        
        a_value = base_a + maillard_effect + cocoa_effect + dairy_effect
        noise = np.random.normal(0, 0.8)
        return max(5.0, min(18.0, a_value + noise))
    
    enhanced_df['color_a_value'] = enhanced_df.apply(estimate_color_a, axis=1)
    
    # Color b* (Blue-Yellow) - Dairy content and sugar browning
    def estimate_color_b(row):
        base_b = 15.0
        dairy_effect = row['dairy_content'] * 0.1
        sugar_effect = row['beetroot_sugar'] * 0.08
        browning_effect = row['browning_potential'] * 0.4
        
        b_value = base_b + dairy_effect + sugar_effect + browning_effect
        noise = np.random.normal(0, 1.0)
        return max(10.0, min(30.0, b_value + noise))
    
    enhanced_df['color_b_value'] = enhanced_df.apply(estimate_color_b, axis=1)
    
    # Gloss (Units) - Crystal structure and surface quality
    def estimate_gloss(row):
        base_gloss = 70.0
        crystal_effect = row['crystal_stability_index'] * 8.0
        cocoa_butter_effect = row['cocoa_butter'] * 0.3
        fat_quality_effect = row['fat_quality_index'] * 15.0
        processing_effect = row['tempering_temperature_c'] * 0.8
        
        gloss = base_gloss + crystal_effect + cocoa_butter_effect + fat_quality_effect + (processing_effect - 25)
        noise = np.random.normal(0, gloss * 0.06)
        return max(55.0, min(95.0, gloss + noise))
    
    enhanced_df['gloss_units'] = enhanced_df.apply(estimate_gloss, axis=1)
    
    # ================================================================
    # SENSORY PROPERTIES ESTIMATION
    # ================================================================
    
    print("3. Estimating sensory properties...")
    
    # Sweetness Score (1-10) - Strong correlation with sugar ratios
    def estimate_sweetness(row):
        base_sweetness = 2.0
        sweetness_ratio_effect = row['sweetness_ratio'] * 7.0  # Strong correlation R²~0.88
        milk_sugar_balance = min(2.0, row['milk_sugar_ratio'] * -0.3)  # High milk reduces perceived sweetness
        
        sweetness = base_sweetness + sweetness_ratio_effect + milk_sugar_balance
        noise = np.random.normal(0, 0.4)
        return max(1.0, min(10.0, sweetness + noise))
    
    enhanced_df['sweetness_score'] = enhanced_df.apply(estimate_sweetness, axis=1)
    
    # Bitterness Score (1-10) - Strong correlation with cocoa content
    def estimate_bitterness(row):
        base_bitterness = 1.5
        cocoa_effect = row['cocoa_intensity'] * 7.5  # Strong correlation R²~0.80
        cocoa_beans_effect = row['cocoa_beans'] * 0.12
        sugar_masking = row['sweetness_ratio'] * -2.0  # Sugar masks bitterness
        
        bitterness = base_bitterness + cocoa_effect + cocoa_beans_effect + sugar_masking
        noise = np.random.normal(0, 0.35)
        return max(1.0, min(10.0, bitterness + noise))
    
    enhanced_df['bitterness_score'] = enhanced_df.apply(estimate_bitterness, axis=1)
    
    # Creaminess Score (1-10) - Strong correlation with dairy content
    def estimate_creaminess(row):
        base_creaminess = 2.0
        dairy_effect = row['dairy_content'] * 0.08  # Strong correlation R²~0.80
        milk_sugar_effect = row['milk_sugar_ratio'] * 1.2
        fat_effect = row['fat'] * 0.06
        
        creaminess = base_creaminess + dairy_effect + milk_sugar_effect + fat_effect
        noise = np.random.normal(0, 0.4)
        return max(1.0, min(10.0, creaminess + noise))
    
    enhanced_df['creaminess_score'] = enhanced_df.apply(estimate_creaminess, axis=1)
    
    # Overall Flavor Balance (1-10) - Complex interaction
    def estimate_flavor_balance(row):
        base_balance = 5.0
        flavor_complexity_bonus = row['flavor_complexity'] * 1.5
        premium_effect = (row['premium_score'] - 1.0) * 1.0
        balance_penalty = abs(row['sweetness_score'] - row['bitterness_score']) * -0.2  # Extreme imbalance reduces score
        
        balance = base_balance + flavor_complexity_bonus + premium_effect + balance_penalty
        noise = np.random.normal(0, 0.5)
        return max(1.0, min(10.0, balance + noise))
    
    enhanced_df['overall_flavor_balance'] = enhanced_df.apply(estimate_flavor_balance, axis=1)
    
    # Texture Liking (1-10) - Processing quality dependent
    def estimate_texture_liking(row):
        optimal_hardness = 28.0  # Optimal hardness for liking
        hardness_penalty = abs(row['hardness_newtons'] - optimal_hardness) * -0.1
        particle_effect = (25.0 - row['particle_size_d50_micron']) * 0.15  # Smaller = better
        processing_quality = row['process_severity_score'] * 0.02
        
        texture_liking = 6.0 + hardness_penalty + particle_effect + processing_quality
        noise = np.random.normal(0, 0.6)
        return max(1.0, min(10.0, texture_liking + noise))
    
    enhanced_df['texture_liking'] = enhanced_df.apply(estimate_texture_liking, axis=1)
    
    # Overall Preference (1-10) - Composite of all sensory aspects
    def estimate_overall_preference(row):
        sweetness_contribution = row['sweetness_score'] * 0.15
        bitterness_contribution = (10 - row['bitterness_score']) * 0.1  # Lower bitterness preferred for mass market
        creaminess_contribution = row['creaminess_score'] * 0.12
        balance_contribution = row['overall_flavor_balance'] * 0.2
        texture_contribution = row['texture_liking'] * 0.15
        
        preference = sweetness_contribution + bitterness_contribution + creaminess_contribution + balance_contribution + texture_contribution
        noise = np.random.normal(0, 0.5)
        return max(1.0, min(10.0, preference + noise))
    
    enhanced_df['overall_preference'] = enhanced_df.apply(estimate_overall_preference, axis=1)
    
    # ================================================================
    # STABILITY/SHELF LIFE PROPERTIES
    # ================================================================
    
    print("4. Estimating stability properties...")
    
    # Fat Bloom Severity (0-5 ordinal scale)
    def estimate_fat_bloom(row):
        bloom_risk = row['fat_bloom_predictor'] / 15.0  # Scale to 0-5 range
        noise = np.random.normal(0, 0.3)
        bloom_severity = bloom_risk + noise
        return max(0, min(5, round(bloom_severity)))
    
    enhanced_df['fat_bloom_severity'] = enhanced_df.apply(estimate_fat_bloom, axis=1)
    
    # Sugar Bloom Severity (0-5 ordinal scale) 
    def estimate_sugar_bloom(row):
        moisture_risk = row['moisture_risk_index'] - 2.0  # Adjust baseline
        hygroscopic_effect = row['hygroscopic_ingredient_score'] / 20.0
        sugar_bloom = (moisture_risk + hygroscopic_effect) / 2.0
        noise = np.random.normal(0, 0.4)
        return max(0, min(5, round(sugar_bloom + noise)))
    
    enhanced_df['sugar_bloom_severity'] = enhanced_df.apply(estimate_sugar_bloom, axis=1)
    
    # Color Change (ΔE units)
    def estimate_color_delta_e(row):
        base_change = 2.0
        degradation_effect = row['color_degradation_predictor'] * 0.15
        thermal_effect = row['thermal_stress_index'] * 8.0
        time_factor = 1.0  # Assuming 6 months storage
        
        delta_e = (base_change + degradation_effect + thermal_effect) * time_factor
        noise = np.random.normal(0, delta_e * 0.15)
        return max(0.5, min(15.0, delta_e + noise))
    
    enhanced_df['color_delta_e'] = enhanced_df.apply(estimate_color_delta_e, axis=1)
    
    # Hardness Change Percentage
    def estimate_hardness_change(row):
        texture_failure_effect = row['texture_failure_predictor'] * 0.8
        moisture_effect = row['moisture_fat_interaction'] * 0.3
        
        hardness_change = texture_failure_effect + moisture_effect
        noise = np.random.normal(0, 1.5)
        return max(-10.0, min(25.0, hardness_change + noise))
    
    enhanced_df['hardness_change_pct'] = enhanced_df.apply(estimate_hardness_change, axis=1)
    
    # Peroxide Value (meq O2/kg fat) - Strong correlation with oxidation risk
    def estimate_peroxide_value(row):
        oxidation_effect = row['oxidation_risk_index'] * 0.18  # Strong correlation R²~0.80
        antioxidant_protection_effect = row['antioxidant_protection'] * -0.05
        
        peroxide = oxidation_effect + antioxidant_protection_effect
        try: 
            noise = np.random.normal(0, peroxide * 0.2)
        except: 
            noise = np.random.normal(0, -peroxide * 0.2)
        return max(0.1, min(25.0, peroxide + noise))
    
    enhanced_df['peroxide_value'] = enhanced_df.apply(estimate_peroxide_value, axis=1)
    
    # Free Fatty Acids (% oleic acid)
    def estimate_free_fatty_acids(row):
        base_ffa = 0.3
        oxidation_effect = row['oxidation_risk_index'] * 0.08
        thermal_damage_effect = row['thermal_damage_index'] * 0.02
        
        ffa = base_ffa + oxidation_effect + thermal_damage_effect
        noise = np.random.normal(0, ffa * 0.25)
        return max(0.05, min(4.0, ffa + noise))
    
    enhanced_df['free_fatty_acids'] = enhanced_df.apply(estimate_free_fatty_acids, axis=1)
    
    # Water Activity (aw)
    def estimate_water_activity(row):
        base_aw = 0.45
        moisture_effect = row['moisture_risk_index'] * 0.02
        hygroscopic_effect = row['water_binding_capacity'] * 0.001
        
        water_activity = base_aw + moisture_effect + hygroscopic_effect
        noise = np.random.normal(0, 0.03)
        return max(0.25, min(0.75, water_activity + noise))
    
    enhanced_df['water_activity'] = enhanced_df.apply(estimate_water_activity, axis=1)
    
    # Moisture Content (%)
    def estimate_moisture_content(row):
        base_moisture = 1.2
        dairy_effect = row['dairy_content'] * 0.015
        processing_humidity_effect = (row['processing_humidity_pct'] - 55) * 0.01
        
        moisture = base_moisture + dairy_effect + processing_humidity_effect
        noise = np.random.normal(0, 0.2)
        return max(0.3, min(3.5, moisture + noise))
    
    enhanced_df['moisture_content_pct'] = enhanced_df.apply(estimate_moisture_content, axis=1)
    
    # Overall Acceptability (1-10)
    def estimate_overall_acceptability(row):
        stability_factor = row['overall_stability_score'] / 10.0
        quality_factor = row['overall_preference'] * 0.8
        degradation_penalty = row['composite_stability_index'] * -0.1
        
        acceptability = stability_factor + quality_factor + degradation_penalty + 2.0
        noise = np.random.normal(0, 0.4)
        return max(1.0, min(10.0, acceptability + noise))
    
    enhanced_df['overall_acceptability'] = enhanced_df.apply(estimate_overall_acceptability, axis=1)
    
    # Shelf Life Exceeded (Binary: 0=acceptable, 1=exceeded)
    def estimate_shelf_life_exceeded(row):
        # Based on multiple failure criteria
        fat_bloom_failure = 1 if row['fat_bloom_severity'] >= 4 else 0
        peroxide_failure = 1 if row['peroxide_value'] > 10.0 else 0
        acceptability_failure = 1 if row['overall_acceptability'] < 4.0 else 0
        
        # Any critical failure results in shelf life exceeded
        return max(fat_bloom_failure, peroxide_failure, acceptability_failure)
    
    enhanced_df['shelf_life_exceeded'] = enhanced_df.apply(estimate_shelf_life_exceeded, axis=1)
    
    # ================================================================
    # ROUND VALUES FOR REALISM
    # ================================================================
    
    print("5. Rounding values for measurement realism...")
    
    # Round to appropriate decimal places based on measurement precision
    enhanced_df['hardness_newtons'] = enhanced_df['hardness_newtons'].round(1)
    enhanced_df['snap_force_newtons'] = enhanced_df['snap_force_newtons'].round(1)
    enhanced_df['viscosity_pas'] = enhanced_df['viscosity_pas'].round(2)
    enhanced_df['particle_size_d50_micron'] = enhanced_df['particle_size_d50_micron'].round(1)
    enhanced_df['color_l_value'] = enhanced_df['color_l_value'].round(1)
    enhanced_df['color_a_value'] = enhanced_df['color_a_value'].round(1)
    enhanced_df['color_b_value'] = enhanced_df['color_b_value'].round(1)
    enhanced_df['gloss_units'] = enhanced_df['gloss_units'].round(1)
    enhanced_df['sweetness_score'] = enhanced_df['sweetness_score'].round(1)
    enhanced_df['bitterness_score'] = enhanced_df['bitterness_score'].round(1)
    enhanced_df['creaminess_score'] = enhanced_df['creaminess_score'].round(1)
    enhanced_df['overall_flavor_balance'] = enhanced_df['overall_flavor_balance'].round(1)
    enhanced_df['texture_liking'] = enhanced_df['texture_liking'].round(1)
    enhanced_df['overall_preference'] = enhanced_df['overall_preference'].round(1)
    enhanced_df['color_delta_e'] = enhanced_df['color_delta_e'].round(2)
    enhanced_df['hardness_change_pct'] = enhanced_df['hardness_change_pct'].round(1)
    enhanced_df['peroxide_value'] = enhanced_df['peroxide_value'].round(2)
    enhanced_df['free_fatty_acids'] = enhanced_df['free_fatty_acids'].round(3)
    enhanced_df['water_activity'] = enhanced_df['water_activity'].round(3)
    enhanced_df['moisture_content_pct'] = enhanced_df['moisture_content_pct'].round(2)
    enhanced_df['overall_acceptability'] = enhanced_df['overall_acceptability'].round(1)
    
    # ================================================================
    # SUMMARY STATISTICS
    # ================================================================
    
    print("6. Generating summary statistics...")
    
    target_columns = [
        'hardness_newtons', 'snap_force_newtons', 'viscosity_pas', 'particle_size_d50_micron',
        'color_l_value', 'color_a_value', 'color_b_value', 'gloss_units',
        'sweetness_score', 'bitterness_score', 'creaminess_score', 'overall_flavor_balance',
        'texture_liking', 'overall_preference', 'fat_bloom_severity', 'sugar_bloom_severity',
        'color_delta_e', 'hardness_change_pct', 'peroxide_value', 'free_fatty_acids',
        'water_activity', 'moisture_content_pct', 'overall_acceptability', 'shelf_life_exceeded'
    ]
    
    print("\nTarget Value Ranges (Estimated):")
    print("-" * 50)
    for col in target_columns:  # Show first 12 for brevity
        min_val = enhanced_df[col].min()
        max_val = enhanced_df[col].max()
        mean_val = enhanced_df[col].mean()
        print(f"{col:25} | {min_val:6.2f} - {max_val:6.2f} | Mean: {mean_val:6.2f}")
    
    print(f"\nTotal formulations processed: {len(enhanced_df)}")
    print(f"Target columns estimated: {len(target_columns)}")
    print(f"Shelf life failures: {enhanced_df['shelf_life_exceeded'].sum()}/{len(enhanced_df)} formulations")
    
    # Correlation validation examples
    print(f"\nCorrelation Validation Examples:")
    print(f"total_cocoa_pct vs hardness_newtons: r = {enhanced_df['total_cocoa_pct'].corr(enhanced_df['hardness_newtons']):.3f}")
    print(f"sweetness_ratio vs sweetness_score: r = {enhanced_df['sweetness_ratio'].corr(enhanced_df['sweetness_score']):.3f}")
    print(f"oxidation_risk_index vs peroxide_value: r = {enhanced_df['oxidation_risk_index'].corr(enhanced_df['peroxide_value']):.3f}")
    
    return enhanced_df

# Example usage:
# Load your dataset
# df = pd.read_csv('complete_100_formulations.csv')
# enhanced_df = estimate_chocolate_targets(df)
# enhanced_df.to_csv('complete_100_formulations_with_targets.csv', index=False)
# print("Enhanced dataset with estimated targets saved!")
