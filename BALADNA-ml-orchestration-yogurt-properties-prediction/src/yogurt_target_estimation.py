import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
import random
import argparse

@dataclass
class TargetCorrelation:
    """Data structure for target correlation parameters"""
    name: str
    r_squared_min: float
    r_squared_max: float
    relationship_type: str
    noise_factor: float
    bounds: Tuple[float, float]
    units: str

class RealisticYogurtTargetEstimator:
    """
    Realistic Target Estimator for Yogurt Quality Parameters
    
    Based on industry expertise and published research correlations.
    Uses engineered features to estimate targets with realistic noise and variability.
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize with target correlation parameters"""
        
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Initialize target correlation parameters based on industry knowledge
        self.target_correlations = self._initialize_target_correlations()
        
        # Process constants (from your specifications)
        self.process_constants = {
            'HEAT_TEMP': 93.5,
            'HEAT_TIME': 300,
            'HOMOG_P1': 200,
            'HOMOG_P2': 45,
            'FLOW_RATE': 30,
            'TARGET_TEMP': 4.0
        }
        
        # Industry variability factors
        self.variability_factors = {
            'culture_batch_variation': 0.15,      # 15% variation between culture batches
            'milk_composition_variation': 0.08,   # 8% variation in milk composition
            'process_control_variation': 0.05,    # 5% process parameter variation
            'measurement_error': 0.03,            # 3% measurement error
            'environmental_variation': 0.10       # 10% environmental effects
        }
    
    def _initialize_target_correlations(self) -> Dict[str, TargetCorrelation]:
        """Initialize target correlation parameters based on realistic industry data"""
        
        return {
            # TIER 1 TARGETS - HIGH CONFIDENCE
            'pH_evolution': TargetCorrelation(
                'pH Evolution During Fermentation', 0.72, 0.85, 'exponential_decay', 0.12,
                (4.0, 6.8), 'pH units'
            ),
            
            'viscosity': TargetCorrelation(
                'Viscosity', 0.64, 0.77, 'power_law', 0.18,
                (100, 10000), 'cP'
            ),
            
            'fermentation_endpoint': TargetCorrelation(
                'Fermentation Endpoint', 0.61, 0.74, 'sigmoid', 0.15,
                (180, 480), 'minutes'
            ),
            
            'fat_globule_size': TargetCorrelation(
                'Fat Globule Size', 0.67, 0.81, 'power_law', 0.14,
                (0.3, 3.0), 'μm'
            ),
            
            # TIER 2 TARGETS - MODERATE CONFIDENCE
            'firmness': TargetCorrelation(
                'Firmness/Gel Strength', 0.56, 0.72, 'polynomial', 0.20,
                (0.1, 5.0), 'N'
            ),
            
            'syneresis': TargetCorrelation(
                'Syneresis', 0.49, 0.67, 'exponential', 0.25,
                (0, 20), 'ml/100g'
            ),
            
            'color_L': TargetCorrelation(
                'Color L* (Lightness)', 0.72, 0.86, 'linear', 0.10,
                (50, 95), 'L* units'
            ),
            
            'color_a': TargetCorrelation(
                'Color a* (Red-Green)', 0.65, 0.80, 'linear', 0.15,
                (-10, 25), 'a* units'
            ),
            
            'color_b': TargetCorrelation(
                'Color b* (Yellow-Blue)', 0.68, 0.82, 'linear', 0.12,
                (5, 35), 'b* units'
            ),
            
            'graininess_perception': TargetCorrelation(
                'Graininess Perception', 0.42, 0.61, 'threshold_sigmoid', 0.30,
                (1, 9), 'scale 1-9'
            ),
            
            'lactic_acid_rate': TargetCorrelation(
                'Lactic Acid Production Rate', 0.56, 0.72, 'michaelis_menten', 0.18,
                (5, 50), 'mg/100g/hour'
            ),
            
            # TIER 3 TARGETS - COMPLEX RELATIONSHIPS
            'overall_liking': TargetCorrelation(
                'Overall Liking Score', 0.34, 0.52, 'multi_factor_interaction', 0.35,
                (1, 9), 'hedonic scale'
            ),
            
            'purchase_intent': TargetCorrelation(
                'Purchase Intent', 0.27, 0.46, 'probabilistic', 0.40,
                (1, 5), '5-point scale'
            ),
            
            'yeast_mold_growth': TargetCorrelation(
                'Yeast and Mold Growth', 0.61, 0.77, 'microbial_kinetics', 0.20,
                (0, 6), 'log CFU/g'
            ),
            
            'ph_drift_storage': TargetCorrelation(
                'pH Drift During Storage', 0.56, 0.74, 'first_order_kinetics', 0.15,
                (-0.3, 0.3), 'pH units change'
            ),
            
            'yield_optimization': TargetCorrelation(
                'Yield Optimization', 0.46, 0.64, 'efficiency_curve', 0.12,
                (0.85, 0.95), 'fraction'
            ),
            
            'acetaldehyde_formation': TargetCorrelation(
                'Acetaldehyde Formation', 0.36, 0.56, 'biochemical_pathway', 0.28,
                (0, 50), 'mg/kg'
            ),
            
            'probiotic_viability': TargetCorrelation(
                'Probiotic Viability', 0.46, 0.67, 'survival_kinetics', 0.22,
                (6, 10), 'log CFU/g'
            )
        }
    
    def estimate_ph_evolution(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate pH evolution during fermentation
        Relationship: Exponential decay with acidification potential
        """
        
        # Primary predictor: acidification potential
        acidification = df.get('acidification_potential', 0)
        buffering = df.get('pH_buffering_capacity', 1) + 0.001
        substrate = df.get('culture_substrate_quality', 1)
        
        # Exponential decay model: pH = pH_initial * exp(-k * acidification_rate)
        initial_ph = 6.7 + df.get('lactose_natural', 0) * 0.002  # Slight variation based on lactose
        
        # Rate constant based on culture activity and buffering
        k = 0.8 * (acidification / buffering) * np.log(substrate + 1)
        
        # Base pH prediction
        ph_predicted = initial_ph * np.exp(-k * 0.3) + 4.0  # Asymptote at ~4.0
        
        # Add realistic variation sources
        culture_variation = np.random.normal(0, 0.1, len(df))  # Culture batch variation
        milk_variation = np.random.normal(0, 0.05, len(df))   # Milk composition variation
        process_variation = np.random.normal(0, 0.03, len(df)) # Process control variation
        
        ph_final = ph_predicted + culture_variation + milk_variation + process_variation
        
        # Apply bounds and add correlation-based noise
        target_r2 = np.random.uniform(0.72, 0.85)
        noise_level = np.sqrt(1 - target_r2) * 0.15
        noise = np.random.normal(0, noise_level, len(df))
        
        result = ph_final + noise
        return np.clip(result, 4.0, 6.8)
    
    def estimate_viscosity(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate viscosity using power law relationship
        Relationship: Power law with texture synergy score
        """
        
        # Primary predictors
        texture_synergy = df.get('texture_synergy_score', 0.1) + 0.01
        protein_network = df.get('protein_network_strength', 0) + 0.01
        fat_total = df.get('fat_total', 0) + 0.01
        homog_energy = df.get('homog_energy_density', 7000) / 1000
        
        # Power law model: viscosity = A * (texture_synergy)^B * fat^C * protein^D
        A = 200  # Base viscosity factor
        B = 1.5  # Texture synergy exponent
        C = 0.8  # Fat contribution
        D = 0.6  # Protein contribution
        
        # Base prediction
        viscosity_base = A * (texture_synergy ** B) * (fat_total ** C) * (protein_network ** D)
        
        # Homogenization effect (reduces viscosity through better dispersion)
        homog_factor = 1.0 + 0.1 * np.log(homog_energy / 7.5)  # Optimal at 7500
        viscosity_predicted = viscosity_base * homog_factor
        
        # Temperature effect (your 4°C storage)
        temp_factor = 1.2  # Higher viscosity at cold temperature
        viscosity_predicted *= temp_factor
        
        # Add realistic variations
        stabilizer_effect = df.get('Stabilizer', 0) * 50  # Direct stabilizer contribution
        viscosity_predicted += stabilizer_effect
        
        # Process variations
        batch_variation = np.random.lognormal(0, 0.12, len(df))  # Log-normal for viscosity
        measurement_error = np.random.normal(1, 0.05, len(df))   # 5% measurement error
        
        viscosity_final = viscosity_predicted * batch_variation * measurement_error
        
        # Apply correlation-realistic noise
        target_r2 = np.random.uniform(0.64, 0.77)
        noise_factor = np.sqrt(1 - target_r2) * 0.20
        noise = np.random.lognormal(0, noise_factor, len(df))
        
        result = viscosity_final * noise
        return np.clip(result, 100, 10000)
    
    def estimate_fermentation_endpoint(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate fermentation endpoint using sigmoid relationship
        Relationship: Sigmoid curve based on acidification rate
        """
        
        # Primary predictors
        acidification = df.get('acidification_potential', 0) + 0.01
        substrate_quality = df.get('culture_substrate_quality', 1)
        buffering = df.get('pH_buffering_capacity', 1) + 0.01
        
        # Sigmoid model: time = L / (1 + exp(-k*(acidification - x0))) + baseline
        L = 200  # Maximum additional time
        k = 2.0  # Steepness factor
        x0 = np.mean(acidification) if len(acidification) > 0 else 1.0  # Inflection point
        baseline = 180  # Minimum fermentation time (3 hours)
        
        # Base time prediction
        fermentation_time = L / (1 + np.exp(-k * (acidification - x0))) + baseline
        
        # Substrate effect (better substrate = faster fermentation)
        substrate_factor = 1.0 - 0.2 * np.log(substrate_quality + 0.1)
        fermentation_time *= substrate_factor
        
        # Temperature effect (your 93.5°C heat treatment affects culture activity)
        temp_effect = 1.0 + 0.05  # Slight increase due to heat stress on culture
        fermentation_time *= temp_effect
        
        # Culture concentration effect
        culture_conc = df.get('CULTURE', 1) + 0.01
        culture_factor = 1.0 / np.sqrt(culture_conc + 0.1)  # More culture = faster fermentation
        fermentation_time *= culture_factor
        
        # Add realistic variations
        culture_batch_var = np.random.normal(1, 0.15, len(df))  # 15% culture batch variation
        process_control_var = np.random.normal(1, 0.08, len(df))  # 8% process variation
        operator_variation = np.random.normal(1, 0.05, len(df))   # 5% operator effect
        
        fermentation_final = fermentation_time * culture_batch_var * process_control_var * operator_variation
        
        # Apply correlation-realistic noise
        target_r2 = np.random.uniform(0.61, 0.74)
        noise_factor = np.sqrt(1 - target_r2) * 0.18
        noise = np.random.normal(1, noise_factor, len(df))
        
        result = fermentation_final * noise
        return np.clip(result, 180, 480)
    
    def estimate_fat_globule_size(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate fat globule size using power law (mechanical relationship)
        Relationship: Inverse power law with homogenization energy
        """
        
        # Primary predictors
        initial_size = df.get('initial_globule_size_weighted', 3.5)
        homog_energy = df.get('homog_energy_density', 7000) + 100
        fat_complexity = df.get('fat_source_diversity', 1) + 0.1
        
        # Power law: final_size = initial_size * (energy/energy_ref)^(-0.6)
        energy_ref = 7350  # Your system: (200+45)*30
        size_reduction_factor = (homog_energy / energy_ref) ** (-0.6)
        
        base_size = initial_size * size_reduction_factor
        
        # Fat source complexity effect (multiple sources harder to homogenize uniformly)
        complexity_penalty = 1.0 + 0.15 * (fat_complexity - 1.0)
        predicted_size = base_size * complexity_penalty
        
        # Pressure ratio effect (your 200:45 = 4.4:1 ratio)
        pressure_ratio = self.process_constants['HOMOG_P1'] / self.process_constants['HOMOG_P2']
        optimal_ratio = 4.5
        ratio_efficiency = 1.0 / (1 + 0.1 * abs(pressure_ratio - optimal_ratio))
        predicted_size /= ratio_efficiency
        
        # Temperature effect (your 65.5°C homogenization temperature)
        homog_temp = 65.5
        temp_efficiency = 1.0 + 0.02 * (homog_temp - 60) - 0.001 * (homog_temp - 60)**2
        predicted_size /= temp_efficiency
        
        # Add realistic variations
        equipment_variation = np.random.lognormal(0, 0.08, len(df))  # Equipment wear/variation
        flow_variation = np.random.normal(1, 0.05, len(df))          # Flow rate variations
        product_variation = np.random.lognormal(0, 0.10, len(df))    # Product-to-product variation
        
        size_final = predicted_size * equipment_variation * flow_variation * product_variation
        
        # Apply correlation-realistic noise
        target_r2 = np.random.uniform(0.67, 0.81)
        noise_factor = np.sqrt(1 - target_r2) * 0.16
        noise = np.random.lognormal(0, noise_factor, len(df))
        
        result = size_final * noise
        return np.clip(result, 0.3, 3.0)
    
    def estimate_color_parameters(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Estimate L*a*b* color parameters using linear relationships
        Strong correlations with ingredient composition
        """
        
        # L* (Lightness) prediction
        fat_total = df.get('fat_total', 3.5)
        protein_total = df.get('protein_total', 3.5)
        red_color = df.get('Red_Color', 0)
        cocoa = df.get('COMPOUND_COCOA', 0)
        
        # Linear model for L*
        L_base = 85 + fat_total * 1.8 + protein_total * 0.5
        L_predicted = L_base - red_color * 1.5 - cocoa * 18
        
        # a* (Red-Green) prediction
        a_base = -2.5 + protein_total * 0.3
        a_predicted = a_base + red_color * 12 + cocoa * 1.8
        
        # b* (Yellow-Blue) prediction
        powder_fat = df.get('fat_from_powder', 0)
        b_base = 8 + powder_fat * 0.4 + fat_total * 0.2
        b_predicted = b_base + cocoa * 7
        
        # Maillard browning effect (your 93.5°C × 300s heat treatment)
        maillard_effect = df.get('pasteurization_intensity', 430) / 430
        sugar_protein_interaction = (df.get('sugar_total', 5) * protein_total) ** 0.5
        maillard_browning = maillard_effect * sugar_protein_interaction * 0.002
        
        L_predicted -= maillard_browning * 2  # Darkening
        a_predicted += maillard_browning * 0.5  # Slight red shift
        b_predicted += maillard_browning * 1.5  # Yellowing
        
        # Add realistic variations
        lighting_variation = np.random.normal(0, 1.5, len(df))   # Measurement lighting
        instrument_variation = np.random.normal(0, 0.8, len(df)) # Colorimeter precision
        sample_variation = np.random.normal(0, 1.0, len(df))     # Sample preparation
        
        # Apply variations
        L_final = L_predicted + lighting_variation + instrument_variation
        a_final = a_predicted + lighting_variation * 0.5 + sample_variation
        b_final = b_predicted + instrument_variation + sample_variation * 0.8
        
        # Apply correlation-realistic noise for each parameter
        target_r2_L = np.random.uniform(0.72, 0.86)
        target_r2_a = np.random.uniform(0.65, 0.80)
        target_r2_b = np.random.uniform(0.68, 0.82)
        
        noise_L = np.random.normal(0, np.sqrt(1 - target_r2_L) * 2.5, len(df))
        noise_a = np.random.normal(0, np.sqrt(1 - target_r2_a) * 2.0, len(df))
        noise_b = np.random.normal(0, np.sqrt(1 - target_r2_b) * 2.2, len(df))
        
        L_result = np.clip(L_final + noise_L, 50, 95)
        a_result = np.clip(a_final + noise_a, -10, 25)
        b_result = np.clip(b_final + noise_b, 5, 35)
        
        return L_result, a_result, b_result
    
    def estimate_graininess_perception(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate graininess perception using threshold sigmoid
        High individual variability, threshold effects
        """
        
        # Primary predictors
        particle_perception = df.get('consumer_graininess_perception', 2.0)
        fat_globule_size = df.get('fat_globule_size_predicted', 1.0)
        protein_aggregation = df.get('protein_aggregation_risk', 0.5)
        
        # Threshold sigmoid model: perception increases sharply above thresholds
        fat_threshold = 2.0  # μm threshold for graininess detection
        protein_threshold = 1.0
        
        # Base graininess score
        fat_contribution = 1 + 6 / (1 + np.exp(-3 * (fat_globule_size - fat_threshold)))
        protein_contribution = 1 + 4 / (1 + np.exp(-2 * (protein_aggregation - protein_threshold)))
        perception_base = particle_perception * 0.5 + 1
        
        graininess_predicted = (fat_contribution + protein_contribution + perception_base) / 3
        
        # Flavor masking effects
        is_strawberry = df.get('is_strawberry', 0)
        is_chocolate = df.get('is_chocolate', 0)
        
        flavor_masking = (
            is_strawberry * 0.8 +   # Strawberry masks graininess
            is_chocolate * 1.2 +    # Chocolate emphasizes texture issues
            (1 - is_strawberry - is_chocolate) * 1.0  # Plain baseline
        )
        
        graininess_adjusted = graininess_predicted * flavor_masking
        
        # High individual consumer variability
        consumer_sensitivity = np.random.normal(1, 0.25, len(df))  # 25% individual variation
        cultural_background = np.random.normal(1, 0.15, len(df))   # Cultural preferences
        expectation_bias = np.random.normal(1, 0.20, len(df))      # Expectation effects
        
        graininess_final = graininess_adjusted * consumer_sensitivity * cultural_background * expectation_bias
        
        # Apply correlation-realistic noise (high noise due to subjectivity)
        target_r2 = np.random.uniform(0.42, 0.61)
        noise_factor = np.sqrt(1 - target_r2) * 0.35
        noise = np.random.normal(0, noise_factor, len(df))
        
        result = graininess_final + noise
        return np.clip(result, 1, 9)
    
    def estimate_overall_liking(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate overall liking using multi-factor interaction model
        Complex interactions, high individual variability
        """
        
        # Primary predictors with weighted importance
        flavor_harmony = df.get('flavor_color_harmony', 0.5)
        texture_quality = df.get('texture_synergy_score', 0.1) + 0.01
        graininess = df.get('graininess_perception', 5.0)  # From estimated or actual
        sweetness_balance = df.get('sugar_masking_power', 5) / 10
        
        # Multi-factor interaction model
        # Base liking from texture (fundamental)
        texture_contribution = 2 + 4 * (texture_quality / (texture_quality + 0.2))
        
        # Flavor harmony contribution (appearance-expectation matching)
        flavor_contribution = 1 + 5 * flavor_harmony
        
        # Graininess penalty (threshold effect)
        graininess_penalty = np.where(graininess < 4, 0, 
                                    np.where(graininess < 6, 1, 
                                           np.where(graininess < 7, 3, 5)))
        
        # Sweetness balance (optimal around 0.5)
        sweetness_contribution = 1 + 2 * (1 - abs(sweetness_balance - 0.5) * 2)
        
        # Base liking score
        liking_base = (texture_contribution + flavor_contribution + sweetness_contribution) / 3
        liking_adjusted = liking_base - graininess_penalty * 0.3
        
        # Consumer segment effects
        age_effect = np.random.normal(0, 0.8, len(df))        # Age preferences
        gender_effect = np.random.normal(0, 0.5, len(df))     # Gender differences
        culture_effect = np.random.normal(0, 0.6, len(df))    # Cultural background
        brand_loyalty = np.random.normal(0, 0.4, len(df))     # Brand perception
        mood_context = np.random.normal(0, 0.7, len(df))      # Testing context/mood
        
        liking_final = (liking_adjusted + age_effect + gender_effect + 
                       culture_effect + brand_loyalty + mood_context)
        
        # Apply correlation-realistic noise (very high due to subjectivity)
        target_r2 = np.random.uniform(0.34, 0.52)
        noise_factor = np.sqrt(1 - target_r2) * 0.40
        noise = np.random.normal(0, noise_factor, len(df))
        
        result = liking_final + noise
        return np.clip(result, 1, 9)
    
    def estimate_yeast_mold_growth(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate yeast/mold growth using microbial kinetics
        Predictive microbiology models
        """
        
        # Environmental predictors
        water_activity = 0.98 - df.get('sugar_added', 0) * 0.003  # Sugar reduces aw
        pH_final = df.get('pH_evolution', 4.5)  # Use estimated or actual pH
        storage_temp = 4.0  # Your target temperature
        
        # Microbial kinetics model (modified Gompertz)
        # Growth rate depends on aw, pH, temperature
        
        # Water activity effect
        aw_factor = np.maximum(0, (water_activity - 0.85) / 0.13)  # No growth below 0.85
        
        # pH effect (optimal around 5.5-6.5 for yeasts/molds)
        pH_factor = np.exp(-0.5 * ((pH_final - 6.0) / 1.5)**2)  # Gaussian curve
        
        # Temperature effect (Q10 model)
        temp_ref = 25  # Reference temperature
        Q10 = 2.0
        temp_factor = Q10**((storage_temp - temp_ref) / 10)
        
        # Base growth potential (log CFU/g after 7 days)
        growth_potential = 6 * aw_factor * pH_factor * temp_factor
        
        # Preservation effects
        lactic_acid_inhibition = np.maximum(0, 1 - (6.5 - pH_final) * 0.3)  # Organic acid effect
        competitive_inhibition = 0.8  # LAB competition effect
        
        growth_adjusted = growth_potential * lactic_acid_inhibition * competitive_inhibition
        
        # Processing effects (your heat treatment)
        heat_treatment_reduction = 2.0  # 2 log reduction from 93.5°C × 300s
        initial_load = np.random.uniform(1, 3, len(df))  # Initial contamination
        
        final_growth = np.maximum(0, growth_adjusted - heat_treatment_reduction + initial_load)
        
        # Add realistic variations
        facility_hygiene = np.random.normal(0, 0.3, len(df))    # Facility variation
        packaging_integrity = np.random.normal(0, 0.2, len(df)) # Package defects
        storage_conditions = np.random.normal(0, 0.25, len(df)) # Storage variability
        
        growth_final = final_growth + facility_hygiene + packaging_integrity + storage_conditions
        
        # Apply correlation-realistic noise
        target_r2 = np.random.uniform(0.61, 0.77)
        noise_factor = np.sqrt(1 - target_r2) * 0.22
        noise = np.random.normal(0, noise_factor, len(df))
        
        result = growth_final + noise
        return np.clip(result, 0, 6)
    
    def estimate_all_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate all targets using engineered features
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            DataFrame with estimated target columns
        """
        
        print("Estimating yogurt quality targets using industry-realistic correlations...")
        
        df_results = df.copy()
        
        # Tier 1 targets (high confidence)
        print("Computing Tier 1 targets...")
        df_results['pH_evolution'] = self.estimate_ph_evolution(df)
        df_results['viscosity'] = self.estimate_viscosity(df)
        df_results['fermentation_endpoint_minutes'] = self.estimate_fermentation_endpoint(df)
        df_results['fat_globule_size_um'] = self.estimate_fat_globule_size(df)
        
        # Color parameters
        L_star, a_star, b_star = self.estimate_color_parameters(df)
        df_results['color_L_star'] = L_star
        df_results['color_a_star'] = a_star
        df_results['color_b_star'] = b_star
        
        # Tier 2 targets (moderate confidence)
        print("Computing Tier 2 targets...")
        df_results['graininess_perception'] = self.estimate_graininess_perception(df)
        df_results['yeast_mold_growth_log_CFU'] = self.estimate_yeast_mold_growth(df)
        
        # Use graininess perception for overall liking (dependency)
        df['graininess_perception'] = df_results['graininess_perception']  # Pass estimated value
        df_results['overall_liking_score'] = self.estimate_overall_liking(df)
        
        # Additional targets using simplified relationships
        print("Computing additional targets...")
        
        # Firmness (power relationship with casein-whey ratio)
        casein_whey = df.get('casein_whey_ratio', 3.0) + 0.1
        protein_network = df.get('protein_network_strength', 1.0) + 0.1
        firmness_base = 0.5 * (casein_whey ** 0.6) * (protein_network ** 0.4)
        firmness_noise = np.random.normal(1, np.sqrt(1-0.64) * 0.3, len(df))
        df_results['firmness_N'] = np.clip(firmness_base * firmness_noise, 0.1, 5.0)
        
        # Syneresis (exponential relationship with stabilizer efficiency)
        stabilizer_eff = df.get('stabilizer_efficiency_ratio', 0.1) + 0.01
        protein_stability = protein_network / (protein_network + 1)
        syneresis_base = 15 * np.exp(-2 * stabilizer_eff) * (1 - protein_stability)
        syneresis_noise = np.random.normal(1, np.sqrt(1-0.58) * 0.4, len(df))
        df_results['syneresis_ml_per_100g'] = np.clip(syneresis_base * syneresis_noise, 0, 20)
        
        # Yield (efficiency curve)
        process_efficiency = df.get('process_efficiency_score', 0.8)
        ingredient_losses = np.random.uniform(0.02, 0.08, len(df))  # 2-8% losses
        yield_base = 0.95 * process_efficiency - ingredient_losses
        yield_noise = np.random.normal(0, np.sqrt(1-0.55) * 0.03, len(df))
        df_results['yield_fraction'] = np.clip(yield_base + yield_noise, 0.85, 0.95)
        
        # pH drift during storage (first-order kinetics)
        buffering_capacity = df.get('pH_buffering_capacity', 1) + 0.1
        residual_activity = np.random.uniform(0.1, 0.3, len(df))  # Residual culture activity
        drift_rate = 0.2 / buffering_capacity * residual_activity
        storage_time_days = 14  # 14-day storage
        ph_drift_base = -drift_rate * np.log(storage_time_days + 1)
        ph_drift_noise = np.random.normal(0, np.sqrt(1-0.65) * 0.08, len(df))
        df_results['ph_drift_14days'] = np.clip(ph_drift_base + ph_drift_noise, -0.3, 0.1)
        
        print(f"Successfully estimated {len([col for col in df_results.columns if col not in df.columns])} target variables")
        
        return df_results
    
    def validate_correlations(self, df_with_targets: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Validate that estimated targets achieve realistic correlations
        
        Returns:
            Dictionary of correlation validation results
        """
        
        validation_results = {}
        
        # Define feature-target pairs for validation
        validation_pairs = {
            'pH_evolution': ['acidification_potential', 'substrate_quality', 'pH_buffering_capacity'],
            'viscosity': ['texture_synergy_score', 'protein_network_strength', 'fat_total'],
            'fermentation_endpoint_minutes': ['acidification_potential', 'culture_substrate_quality'],
            'fat_globule_size_um': ['homog_energy_density', 'initial_globule_size_weighted'],
            'color_L_star': ['fat_total', 'COMPOUND_COCOA'],
            'graininess_perception': ['consumer_graininess_perception', 'fat_globule_size_um'],
            'overall_liking_score': ['flavor_color_harmony', 'texture_synergy_score', 'graininess_perception']
        }
        
        for target, features in validation_pairs.items():
            if target in df_with_targets.columns:
                correlations = {}
                for feature in features:
                    if feature in df_with_targets.columns:
                        corr = df_with_targets[feature].corr(df_with_targets[target])
                        if not np.isnan(corr):
                            correlations[feature] = abs(corr)
                
                # Calculate combined R² (approximate)
                if correlations:
                    combined_r2 = 1 - np.prod([1 - corr**2 for corr in correlations.values()])
                    correlations['combined_R2'] = combined_r2
                
                validation_results[target] = correlations
        
        return validation_results
    
    def generate_summary_report(self, df_with_targets: pd.DataFrame) -> str:
        """Generate summary report of estimated targets"""
        
        target_columns = [col for col in df_with_targets.columns 
                         if any(target in col.lower() for target in 
                               ['ph_', 'viscosity', 'fermentation_', 'color_', 'graininess', 
                                'liking', 'yeast', 'firmness', 'syneresis', 'yield', 'drift'])]
        
        report = "YOGURT TARGET ESTIMATION SUMMARY REPORT\n"
        report += "=" * 50 + "\n\n"
        
        report += f"Dataset size: {len(df_with_targets)} samples\n"
        report += f"Total features: {len(df_with_targets.columns) - len(target_columns)}\n"
        report += f"Estimated targets: {len(target_columns)}\n\n"
        
        report += "TARGET STATISTICS:\n"
        report += "-" * 30 + "\n"
        
        for target in sorted(target_columns):
            values = df_with_targets[target].dropna()
            if len(values) > 0:
                report += f"{target:25s}: mean={values.mean():.3f}, std={values.std():.3f}, "
                report += f"range=[{values.min():.3f}, {values.max():.3f}]\n"
        
        # Validation summary
        validation = self.validate_correlations(df_with_targets)
        if validation:
            report += f"\nCORRELATION VALIDATION:\n"
            report += "-" * 30 + "\n"
            for target, corrs in validation.items():
                if 'combined_R2' in corrs:
                    report += f"{target:25s}: R² = {corrs['combined_R2']:.3f}\n"
        
        return report

def main():
    """Example usage of RealisticYogurtTargetEstimator"""
    
    print("Yogurt Realistic Target Estimator")
    print("=" * 40)
    
    # Initialize estimator
    estimator = RealisticYogurtTargetEstimator(random_seed=42)
    
    parser = argparse.ArgumentParser(description='Yogurt Feature Engineering')
    parser.add_argument('--input_csv', type=str, required=True, default='', help='Path to input XL file with yogurt data')
    parser.add_argument('--output_csv', type=str, required=False, default='./data/processed/Recipe_yogurt_data_generated_with_all_features_and_targets.csv', help='Path to output CSV file with estimated targets')
    args = parser.parse_args()
    
    # Estimate targets (requires engineered features dataframe)
    df_features = pd.read_csv(args.input_csv)
    df_with_targets = estimator.estimate_all_targets(df_features)

    # validation_results = estimator.validate_correlations(df_with_targets)
    report = estimator.generate_summary_report(df_with_targets)
    print(report)
    
    print("\nTarget correlation parameters loaded:")
    for name, params in estimator.target_correlations.items():
        print(f"  {name:25s}: R² = {params.r_squared_min:.2f}-{params.r_squared_max:.2f}")
    
    print(f"\nEstimator ready for {len(estimator.target_correlations)} targets")
    print("Use: estimator.estimate_all_targets(your_engineered_features_df)")

    # Compute and display correlation matrix
    correlation_matrix = df_with_targets.select_dtypes(include=[np.number]).corr()
    correlation_matrix = correlation_matrix.dropna(how='all').dropna(axis=1, how='all')

    # Identify columns with nan correlations
    cols_nan = correlation_matrix.isna().sum()
    print(f"\nDropping {len(cols_nan)} columns with NaN correlations: {cols_nan}")

    # Keep columns with valid correlations
    df_with_targets = df_with_targets[correlation_matrix.columns]
    df_with_targets.to_csv(args.output_csv, index=False)

    print("\nSaving Correlation matrix of estimated targets:")
    sns.heatmap(correlation_matrix, annot=False, fmt=".2f", cmap='Blues', cbar=True)
    plt.title("Correlation Matrix of Estimated Yogurt Targets")
    plt.savefig('./data/processed/yogurt_targets_correlation_matrix.png', dpi=1200, bbox_inches='tight')

if __name__ == "__main__":
    main()