import pandas as pd
import numpy as np
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class YogurtTargetPredictor:
    """
    Scientific Target Prediction for Yogurt Manufacturing
    
    Implements mechanistic models based on dairy science literature
    for predicting quality, sensory, and process targets from formulation
    and process parameters.
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize with random seed for reproducibility"""
        np.random.seed(random_seed)
        
        # Target ranges and expected correlations
        self.target_specs = {
            'pH_evolution': {'range': (4.0, 6.8), 'correlation': (0.72, 0.85), 'noise': 0.12},
            'viscosity': {'range': (100, 10000), 'correlation': (0.64, 0.77), 'noise': 0.18},
            'fermentation_endpoint': {'range': (180, 480), 'correlation': (0.61, 0.74), 'noise': 0.15},
            'fat_globule_size': {'range': (0.3, 3.0), 'correlation': (0.67, 0.81), 'noise': 0.14},
            'firmness': {'range': (0.1, 5.0), 'correlation': (0.56, 0.72), 'noise': 0.20},
            'syneresis': {'range': (0, 20), 'correlation': (0.49, 0.67), 'noise': 0.25},
            'color_L': {'range': (50, 95), 'correlation': (0.72, 0.86), 'noise': 0.10},
            'color_a': {'range': (-10, 25), 'correlation': (0.65, 0.80), 'noise': 0.15},
            'color_b': {'range': (5, 35), 'correlation': (0.68, 0.82), 'noise': 0.12},
            'graininess_perception': {'range': (1, 9), 'correlation': (0.42, 0.61), 'noise': 0.30},
            'lactic_acid_rate': {'range': (5, 50), 'correlation': (0.56, 0.72), 'noise': 0.18},
            'overall_liking': {'range': (1, 9), 'correlation': (0.34, 0.52), 'noise': 0.35},
            'purchase_intent': {'range': (1, 5), 'correlation': (0.27, 0.46), 'noise': 0.40},
            'yeast_mold_growth': {'range': (0, 6), 'correlation': (0.61, 0.77), 'noise': 0.20},
            'ph_drift_storage': {'range': (-0.3, 0.3), 'correlation': (0.56, 0.74), 'noise': 0.15},
            'yield_optimization': {'range': (0.85, 0.95), 'correlation': (0.46, 0.64), 'noise': 0.12},
            'acetaldehyde_formation': {'range': (0, 50), 'correlation': (0.36, 0.56), 'noise': 0.28},
            'probiotic_viability': {'range': (6, 10), 'correlation': (0.46, 0.67), 'noise': 0.22}
        }
    
    def _add_noise(self, values: np.ndarray, noise_level: float) -> np.ndarray:
        """Add realistic noise to predictions"""
        noise = np.random.normal(0, noise_level, size=len(values))
        return values * (1 + noise)
    
    def _clip_to_range(self, values: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """Clip values to realistic range"""
        return np.clip(values, min_val, max_val)
    
    # ========== TIER 1 TARGETS - HIGH CONFIDENCE ==========
    
    def predict_pH_evolution(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict pH during fermentation (exponential decay model)
        
        Mechanism: Lactic acid production by cultures
        pH = pH_initial - ΔpH × (1 - exp(-k × acidification_potential))
        
        Key drivers:
        - acidification_potential (primary)
        - culture_substrate_quality
        - pH_buffering_capacity (resists change)
        """
        # Initial pH (fresh milk)
        pH_initial = 6.6
        
        # Acidification rate constant
        k = 0.15
        
        # Maximum pH drop (to isoelectric point)
        max_pH_drop = 2.2
        
        # Main acidification driver (exponential decay)
        acid_factor = df['acidification_potential'] / (df['acidification_potential'].mean() + 0.001)
        
        # Buffering resistance factor
        buffer_factor = 1.0 + (df['pH_buffering_capacity'] / 
                               (df['pH_buffering_capacity'].mean() + 0.001)) * 0.3
        
        # Substrate quality effect
        substrate_factor = df['culture_substrate_quality'] / (df['culture_substrate_quality'].mean() + 0.001)
        
        # pH evolution model
        pH_drop = max_pH_drop * (1 - np.exp(-k * acid_factor * substrate_factor)) / buffer_factor
        pH = pH_initial - pH_drop
        
        # Add realistic noise and clip
        pH = self._add_noise(pH, self.target_specs['pH_evolution']['noise'])
        pH = self._clip_to_range(pH, 4.0, 6.8)
        
        return pH
    
    def predict_viscosity(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict viscosity (power law model)
        
        Mechanism: Protein gel network with stabilizer reinforcement
        η = K × [protein_network]^2.0 × [stabilizer]^1.5 × heat^0.5 / [fat]^0.3
        
        Key drivers:
        - texture_synergy_score (master predictor)
        - protein_network_strength
        - water_binding_competition
        - Stabilizer
        """
        # Target mean around 3,000 cP (mid-range for stirred yogurt)
        base_viscosity = 3000
        
        # Protein effect (scaled to reasonable range)
        protein_norm = df['protein_total'] / df['protein_total'].mean()
        protein_effect = protein_norm ** 1.5  # Reduced from 2.0
        
        # Stabilizer effect (strong but bounded)
        stabilizer_norm = df['Stabilizer'] / (df['Stabilizer'].mean() + 0.01)
        stabilizer_effect = 1.0 + np.tanh(stabilizer_norm) * 1.5  # Bounded by tanh
        
        # Heat effect (moderate)
        heat_norm = df['pasteurization_intensity'] / df['pasteurization_intensity'].mean()
        heat_effect = 1.0 + (heat_norm - 1.0) * 0.4
        
        # Fat reduction (moderate)
        fat_norm = df['fat_total'] / (df['fat_total'].mean() + 0.01)
        fat_reduction = 1.0 / (1.0 + fat_norm * 0.3)
        
        # Water binding (moderate)
        water_norm = df['water_binding_competition'] / df['water_binding_competition'].mean()
        water_effect = 1.0 + (water_norm - 1.0) * 0.3
        
        # Texture synergy (bounded multiplier)
        synergy_norm = df['texture_synergy_score'] / df['texture_synergy_score'].mean()
        synergy_effect = 1.0 + np.tanh(synergy_norm - 1.0) * 0.5
        
        # Combined model (multiplicative effects bounded)
        viscosity = (base_viscosity * protein_effect * stabilizer_effect * 
                    heat_effect * fat_reduction * water_effect * synergy_effect)
        
        # Multiplicative noise (stays within reasonable bounds)
        noise_factor = np.exp(np.random.normal(0, 0.15, len(df)))
        viscosity *= noise_factor
        
        return np.clip(viscosity, 100, 10000)
    
    def predict_fermentation_endpoint(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict time to reach pH 4.6 (sigmoid model)
        
        Mechanism: Monod growth kinetics
        Time ∝ ln(ΔpH) / (μ × [Culture])
        
        Key drivers:
        - acidification_potential (inverse)
        - CULTURE amount
        - pH_buffering_capacity (extends time)
        """
        # Base fermentation time (minutes)
        base_time = 300
        
        # Acidification rate effect (inverse)
        acid_rate = df['acidification_potential'] / (df['acidification_potential'].mean() + 0.001)
        time_factor = 1.0 / (acid_rate + 0.3)
        
        # Culture amount effect (inverse)
        culture_factor = df['CULTURE'].mean() / (df['CULTURE'] + 1)
        
        # Buffering extends time (linear)
        buffer_extension = 1.0 + (df['pH_buffering_capacity'] / 
                                 (df['pH_buffering_capacity'].mean() + 0.001)) * 0.4
        
        # Substrate quality effect (inverse)
        substrate_factor = df['culture_substrate_quality'].mean() / (df['culture_substrate_quality'] + 0.001)
        substrate_factor = np.clip(substrate_factor, 0.5, 2.0)
        
        # Sigmoid behavior (lag + exponential + plateau)
        fermentation_time = base_time * time_factor * culture_factor * buffer_extension * substrate_factor
        
        # Add noise and clip
        fermentation_time = self._add_noise(fermentation_time, 
                                           self.target_specs['fermentation_endpoint']['noise'])
        fermentation_time = self._clip_to_range(fermentation_time, 180, 480)
        
        return fermentation_time
    
    def predict_fat_globule_size(self, df: pd.DataFrame) -> np.ndarray:
        """
        Estimate fat globule size using validated power law
        """
        # Base parameters
        initial_size = df.get('initial_globule_size_weighted', 3.5)
        homog_energy = df.get('homog_energy_density', 7000) + 100
        
        # Power law: d ∝ d0 * (P)^(-0.6)
        energy_ref = 7350  # (200+45)*30
        size_reduction = (homog_energy / energy_ref) ** (-0.6)
        base_size = initial_size * size_reduction
        
        # Homogenization challenge (difficulty of the specific formulation)
        challenge = df.get('homog_challenge_index', 100)
        challenge_effect = (challenge / 100) ** 0.3  # Sublinear effect
        
        # Pressure ratio efficiency (sharp penalty for deviation)
        pressure_ratio = df['Homogenization_Pressure_Primary_Bar'] / df['Homogenization_Pressure_Secondary_Bar']
        optimal_ratio = 4.5
        ratio_deviation = abs(pressure_ratio - optimal_ratio)
        ratio_efficiency = np.exp(-0.3 * ratio_deviation)  # Exponential penalty
        
        # Temperature effect on viscosity (affects breakup efficiency)
        temp = df.get('Homogenization_Temperature_Min_C', 65)
        temp_optimal = 65
        # Viscosity changes ~3% per °C, affects breakup efficiency
        temp_efficiency = np.exp(-0.025 * abs(temp - temp_optimal))
        
        # Emulsifier stabilization (prevents re-coalescence)
        emulsifier_effect = 1.0 / (1.0 + df.get('EMULSIFIER', 0) * 40)
        
        # Combined mechanistic model
        predicted_size = (base_size * challenge_effect * emulsifier_effect / 
                        (ratio_efficiency * temp_efficiency))
        
        # Single noise layer (realistic process variation)
        # Based on expected R² ~ 0.74 (sqrt(1-0.74²) ≈ 0.16)
        total_variation = np.random.lognormal(0, 0.14, len(df))
        
        result = predicted_size * total_variation
        return np.clip(result, 0.3, 3.0)
    
    # ========== TIER 2 TARGETS - MODERATE CONFIDENCE ==========
    
    def predict_firmness(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict gel firmness (polynomial model)
        
        Mechanism: Casein network formation at pH 4.6
        Firmness ≈ [Casein]^2.5 × [Denatured Whey]^1.2 × Heat^0.8 / [Fat]^0.4
        
        Key drivers:
        - protein_network_strength
        - casein_equivalent
        - pasteurization_intensity
        """
        # Target mean around 1.5 N
        base_firmness = 1.5
        
        # Protein network (scaled)
        protein_norm = df['protein_network_strength'] / df['protein_network_strength'].mean()
        protein_effect = 0.5 + protein_norm * 0.8  # Range: 0.5-1.8x
        
        # Casein (primary gel former)
        casein_norm = df['casein_equivalent'] / df['casein_equivalent'].mean()
        casein_effect = 0.6 + casein_norm ** 1.2 * 0.7  # Range: 0.6-1.5x
        
        # Heat treatment (moderate effect)
        heat_norm = df['pasteurization_intensity'] / df['pasteurization_intensity'].mean()
        heat_effect = 0.8 + heat_norm * 0.5  # Range: 0.8-1.5x
        
        # Stabilizer (strong but saturating)
        stabilizer_boost = np.tanh(df['Stabilizer'] * 3) * 0.8 + 1.0  # Range: 1.0-1.8x
        
        # Fat (moderate reduction)
        fat_norm = df['fat_total'] / (df['fat_total'].mean() + 0.01)
        fat_effect = 1.0 / (1.0 + fat_norm * 0.25)  # Max 20% reduction
        
        # Water binding
        water_norm = df['water_binding_competition'] / df['water_binding_competition'].mean()
        water_effect = 0.9 + water_norm * 0.3
        
        # Combined
        firmness = (base_firmness * protein_effect * casein_effect * heat_effect * 
                stabilizer_boost * fat_effect * water_effect)
        
        # Multiplicative noise
        noise_factor = np.exp(np.random.normal(0, 0.16, len(df)))
        firmness *= noise_factor
        
        return np.clip(firmness, 0.1, 5.0)
    
    def predict_syneresis(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict whey separation (exponential model)
        
        Mechanism: Network contraction expels water
        Syneresis = A × exp(-B × [Stabilizer]) × [Aggregation]
        
        Key drivers:
        - water_binding_competition (inverse)
        - Stabilizer (strong inverse)
        - protein_aggregation_potential
        """
        # Maximum syneresis without stabilizer
        max_syneresis = 18.0
        
        # Exponential stabilizer effect (most important)
        stabilizer_reduction = np.exp(-5.0 * df['Stabilizer'])
        
        # Water binding reduces syneresis
        water_binding_factor = 1.0 / (df['water_binding_competition'] + 0.3)
        
        # Protein aggregation increases syneresis
        aggregation_factor = 1.0 + df['protein_aggregation_potential'] * 0.5
        
        # Protein network strength reduces syneresis
        network_factor = df['protein_network_strength'].mean() / (df['protein_network_strength'] + 0.001)
        network_factor = np.clip(network_factor, 0.5, 2.0)
        
        # Fat helps trap water
        fat_reduction = 1.0 / (1.0 + df['fat_total'] * 0.1)
        
        # Syneresis calculation
        syneresis = (max_syneresis * stabilizer_reduction * water_binding_factor * 
                    aggregation_factor * network_factor * fat_reduction)
        
        # Add noise and clip
        syneresis = self._add_noise(syneresis, self.target_specs['syneresis']['noise'])
        syneresis = self._clip_to_range(syneresis, 0, 20)
        
        return syneresis
    
    def predict_color_parameters(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
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
    
    def predict_graininess_perception(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict graininess on 1-9 scale (threshold sigmoid)
        
        Mechanism: Particle size detection threshold
        Graininess = 1 / (1 + exp(-k × (Particle Size - threshold)))
        
        Key drivers:
        - protein_aggregation_potential
        - pasteurization_intensity (excessive causes graininess)
        - particle_uniformity_index (inverse)
        """
        # Sigmoid threshold (particle size ~25 μm detection)
        k = 0.15
        threshold = 5.0
        
        # Aggregation drives graininess
        aggregation_score = df['protein_aggregation_potential'] * 2
        
        # Excessive heat causes grainy texture
        heat_score = np.maximum(0, (df['pasteurization_intensity'] - 400) / 100)
        
        # Poor uniformity increases graininess
        uniformity_score = (df['particle_uniformity_index'].mean() / 
                           (df['particle_uniformity_index'] + 0.001) - 1) * 3
        
        # Fat globule size contribution
        globule_score = (df['fat_globule_size_predicted'] - 0.5) * 2
        
        # Homogenization quality (inverse)
        homog_score = (df['homog_energy_density'].mean() / 
                      (df['homog_energy_density'] + 100)) * 2
        
        # Combined graininess score
        grain_factor = aggregation_score + heat_score + uniformity_score + globule_score + homog_score
        
        # Sigmoid transformation to 1-9 scale
        graininess_raw = 1.0 / (1.0 + np.exp(-k * (grain_factor - threshold)))
        graininess = 1 + graininess_raw * 8  # Scale to 1-9
        
        # Add noise and clip
        graininess = self._add_noise(graininess, self.target_specs['graininess_perception']['noise'])
        graininess = self._clip_to_range(graininess, 1, 9)
        
        return graininess
    
    def predict_lactic_acid_rate(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict lactic acid production rate (Michaelis-Menten)
        
        Mechanism: Enzymatic lactose fermentation
        v = Vmax × [S] / (Km + [S])
        
        Key drivers:
        - acidification_potential
        - CULTURE (enzyme concentration)
        - lactose_natural (substrate)
        """
        # Vmax scaled to realistic range
        Vmax = 35.0  # Reduced from 40
        
        # Michaelis constant
        Km = 20.0  # Increased from 15 (less sensitive)
        
        # Culture effect (bounded)
        culture_norm = df['CULTURE'] / (df['CULTURE'].mean() + 0.01)
        culture_effect = 0.5 + np.tanh(culture_norm - 1.0) * 0.7  # Range: 0.5-1.5x
        
        # Substrate (Michaelis-Menten)
        substrate = df['lactose_natural']
        MM_factor = substrate / (Km + substrate)
        
        # Quality factor (bounded)
        quality_norm = df['culture_substrate_quality'] / df['culture_substrate_quality'].mean()
        quality_effect = 0.7 + quality_norm * 0.5  # Range: 0.7-1.5x
        
        # Acidification (bounded)
        acid_norm = df['acidification_potential'] / df['acidification_potential'].mean()
        acid_effect = 0.6 + np.tanh(acid_norm - 1.0) * 0.6  # Range: 0.6-1.5x
        
        # Combined
        rate = Vmax * culture_effect * MM_factor * quality_effect * acid_effect
        
        # Multiplicative noise
        noise_factor = np.exp(np.random.normal(0, 0.14, len(df)))
        rate *= noise_factor
        
        return np.clip(rate, 5, 50)
    
    # ========== TIER 3 TARGETS - COMPLEX RELATIONSHIPS ==========
    
    def predict_overall_liking(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict hedonic liking on 1-9 scale (multi-factor interaction)
        
        Mechanism: Multi-sensory integration
        Liking = α×Taste + β×Texture + γ×Appearance - δ×Defects
        
        Key drivers:
        - taste_balance_score
        - texture_synergy_score
        - harmony features
        - defect features (negative)
        """
        # Base liking (neutral)
        base = 5.5
        
        # All effects bounded to ±1.5 range
        taste_norm = df['taste_balance_score'] / df['taste_balance_score'].mean()
        taste_effect = np.tanh((taste_norm - 1.0) * 2) * 1.2  # ±1.2
        
        texture_norm = df['texture_synergy_score'] / df['texture_synergy_score'].mean()
        texture_effect = np.tanh((texture_norm - 1.0) * 2) * 1.0  # ±1.0
        
        harmony = (df['strawberry_harmony'] + df['chocolate_harmony'])
        harmony_norm = harmony / (harmony.mean() + 0.01)
        harmony_effect = (harmony_norm - 1.0) * 0.4  # ±0.4
        harmony_effect = np.clip(harmony_effect, -0.5, 0.5)
        
        fat_norm = df['fat_total'] / (df['fat_total'].mean() + 0.01)
        fat_effect = (fat_norm - 1.0) * 0.5  # ±0.5
        fat_effect = np.clip(fat_effect, -0.6, 0.6)
        
        bitter_norm = df['protein_bitterness_potential'] / df['protein_bitterness_potential'].mean()
        bitter_penalty = -(bitter_norm - 1.0) * 0.4  # ±0.4
        bitter_penalty = np.clip(bitter_penalty, -0.5, 0.5)
        
        grain_norm = df['protein_aggregation_potential'] / df['protein_aggregation_potential'].mean()
        grain_penalty = -grain_norm * 0.3  # Max -0.3
        grain_penalty = np.clip(grain_penalty, -0.5, 0)
        
        # Combined (all bounded, sum can't exceed ±4 from base)
        liking = (base + taste_effect + texture_effect + harmony_effect + 
                fat_effect + bitter_penalty + grain_penalty)
        
        # Moderate noise
        noise = np.random.normal(0, 0.5, len(df))
        liking += noise
        
        return np.clip(liking, 1, 9)
    
    def predict_purchase_intent(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict purchase intent on 1-5 scale (probabilistic)
        
        Mechanism: Consumer decision making
        P(Purchase) = 1 / (1 + exp(-[β0 + β1×Liking + β2×Health + β3×Premium]))
        
        Key drivers:
        - overall_liking (if calculated)
        - premium_ingredient_score
        - protein_total (health halo)
        """
        # Base intent
        base_intent = 3.0
        
        # Liking is primary driver (use taste balance as proxy if liking not available)
        liking_proxy = df['taste_balance_score'] / (df['taste_balance_score'].mean() + 0.001)
        liking_effect = (liking_proxy - 1) * 2.0
        
        # Premium positioning
        premium_effect = (df['premium_ingredient_score'] / 
                         (df['premium_ingredient_score'].mean() + 0.001) - 1) * 0.5
        
        # Health halo (high protein)
        health_effect = (df['protein_total'] / (df['protein_total'].mean() + 0.001) - 1) * 0.3
        
        # Low sugar appeal (negative of sugar)
        sugar_effect = -(df['sugar_total'] / (df['sugar_total'].mean() + 0.001) - 1) * 0.2
        
        # Harmony (trust factor)
        harmony_trust = ((df['strawberry_harmony'] + df['chocolate_harmony']) / 
                        (df['strawberry_harmony'].mean() + df['chocolate_harmony'].mean() + 0.001)) * 0.3
        
        # Purchase intent calculation
        purchase_intent = (base_intent + liking_effect + premium_effect + 
                          health_effect + sugar_effect + harmony_trust)
        
        # Add substantial noise (many external factors)
        purchase_intent = self._add_noise(purchase_intent, 
                                         self.target_specs['purchase_intent']['noise'])
        purchase_intent = self._clip_to_range(purchase_intent, 1, 5)
        
        return purchase_intent
    
    def predict_yeast_mold_growth(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict microbial growth in log CFU/g (microbial kinetics)
        
        Mechanism: Logistic growth with pH inhibition
        Growth inhibited by: Low pH, high pasteurization
        
        Key drivers:
        - pH_evolution (if available, otherwise use acid potential)
        - pasteurization_intensity (kills organisms)
        - sugar_total (substrate for growth)
        """
        # Maximum growth (log CFU/g)
        max_growth = 6.0
        
        # pH effect (strong inhibition below pH 4.5)
        # Use acidification potential as inverse proxy for pH
        pH_proxy = 6.0 - (df['acidification_potential'] / 
                         (df['acidification_potential'].mean() + 0.001)) * 1.5
        pH_inhibition = np.where(pH_proxy < 4.5, 
                                np.exp(-(4.5 - pH_proxy) * 2), 
                                1.0)
        
        # Pasteurization kills organisms (exponential reduction)
        pasteurization_kill = np.exp(-(df['pasteurization_intensity'] - 200) / 150)
        pasteurization_kill = np.clip(pasteurization_kill, 0.05, 1.0)
        
        # Sugar provides growth substrate
        sugar_effect = (df['sugar_total'] / (df['sugar_total'].mean() + 0.001)) * 0.5 + 0.5
        
        # Water activity (more water = more growth)
        water_effect = (df['water_total'] / (df['water_total'].mean() + 0.001)) * 0.3 + 0.7
        
        # Growth calculation
        yeast_mold = max_growth * pH_inhibition * pasteurization_kill * sugar_effect * water_effect
        
        # Add noise and clip
        yeast_mold = self._add_noise(yeast_mold, self.target_specs['yeast_mold_growth']['noise'])
        yeast_mold = self._clip_to_range(yeast_mold, 0, 6)
        
        return yeast_mold
    
    def predict_ph_drift_storage(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict pH change during storage (first-order kinetics)
        
        Mechanism: Continued fermentation by residual culture
        dpH/dt = -k × [Residual Substrate] × [Active Culture]
        
        Key drivers:
        - fermentable_substrate (residual)
        - CULTURE (active organisms)
        - pH_buffering_capacity (resists change)
        """
        # Maximum pH drift (negative = decrease)
        max_drift = -0.25
        
        # Residual substrate effect
        substrate_effect = (df['fermentable_substrate'] / 
                           (df['fermentable_substrate'].mean() + 0.001))
        
        # Culture activity (continued acidification)
        culture_effect = (df['CULTURE'] / (df['CULTURE'].mean() + 0.001))
        
        # Buffering resistance
        buffer_resistance = df['pH_buffering_capacity'].mean() / (df['pH_buffering_capacity'] + 0.001)
        buffer_resistance = np.clip(buffer_resistance, 0.5, 2.0)
        
        # Acidification potential proxy
        acid_effect = (df['acidification_potential'] / 
                      (df['acidification_potential'].mean() + 0.001))
        
        # First-order kinetics (exponential approach to asymptote)
        drift_factor = 1 - np.exp(-0.3 * substrate_effect * culture_effect * acid_effect)
        
        # pH drift calculation
        ph_drift = max_drift * drift_factor * buffer_resistance
        
        # Add noise and clip
        ph_drift = self._add_noise(ph_drift, self.target_specs['ph_drift_storage']['noise'])
        ph_drift = self._clip_to_range(ph_drift, -0.3, 0.3)
        
        return ph_drift
    
    def predict_yield_optimization(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict manufacturing yield (efficiency curve)
        
        Mechanism: Mass balance with losses from syneresis and processing
        Yield = 1 - (Syneresis Loss + Processing Loss)
        
        Key drivers:
        - syneresis (if calculated, major loss)
        - water_binding_competition
        - Stabilizer
        - process_optimization_score
        """
        # Target mean 0.91
        base_yield = 0.91
        
        # All effects scaled for narrow range (total ±0.04)
        
        # Water binding (reduces syneresis loss)
        water_norm = df['water_binding_competition'] / df['water_binding_competition'].mean()
        water_effect = (water_norm - 1.0) * 0.015  # ±0.015
        water_effect = np.clip(water_effect, -0.02, 0.02)
        
        # Stabilizer (major effect but saturates)
        stabilizer_effect = np.tanh(df['Stabilizer'] * 4) * 0.025  # Max +0.025
        
        # Protein network
        protein_norm = df['protein_network_strength'] / df['protein_network_strength'].mean()
        protein_effect = (protein_norm - 1.0) * 0.01  # ±0.01
        protein_effect = np.clip(protein_effect, -0.015, 0.015)
        
        # Process optimization
        process_norm = df['process_optimization_score'] / df['process_optimization_score'].mean()
        process_effect = (process_norm - 1.0) * 0.012  # ±0.012
        process_effect = np.clip(process_effect, -0.015, 0.015)
        
        # Combined (max deviation ±0.06 from base)
        yield_opt = base_yield + water_effect + stabilizer_effect + protein_effect + process_effect
        
        # Small noise for narrow range
        noise = np.random.normal(0, 0.008, len(df))
        yield_opt += noise
        
        return np.clip(yield_opt, 0.85, 0.95)
    
    def predict_acetaldehyde_formation(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict acetaldehyde concentration (biochemical pathway)
        
        Mechanism: Threonine aldolase pathway in S. thermophilus
        [Acetaldehyde] ∝ [Culture] × [Protein] × [Substrate]
        
        Key drivers:
        - CULTURE (S. thermophilus produces more)
        - protein_total (threonine source)
        - culture_substrate_quality
        """
        # Base acetaldehyde production
        base_acetaldehyde = 20.0  # mg/kg
        
        # Culture amount (linear with S. thermophilus activity)
        culture_effect = (df['CULTURE'] / (df['CULTURE'].mean() + 0.001))
        
        # Protein provides threonine substrate
        protein_effect = (df['protein_total'] / (df['protein_total'].mean() + 0.001))
        
        # Substrate quality supports growth and metabolism
        substrate_effect = (df['culture_substrate_quality'] / 
                           (df['culture_substrate_quality'].mean() + 0.001))
        
        # Acidification indicates active fermentation
        acid_effect = (df['acidification_potential'] / 
                      (df['acidification_potential'].mean() + 0.001)) * 0.5 + 0.5
        
        # Heat treatment creates precursors
        heat_effect = (df['pasteurization_intensity'] / 
                      (df['pasteurization_intensity'].mean() + 0.001)) * 0.3 + 0.7
        
        # Acetaldehyde formation
        acetaldehyde = (base_acetaldehyde * culture_effect * protein_effect * 
                       substrate_effect * acid_effect * heat_effect)
        
        # Add substantial noise (strain-specific variability)
        acetaldehyde = self._add_noise(acetaldehyde, 
                                      self.target_specs['acetaldehyde_formation']['noise'])
        acetaldehyde = self._clip_to_range(acetaldehyde, 0, 50)
        
        return acetaldehyde
    
    def predict_probiotic_viability(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict probiotic survival in log CFU/g (survival kinetics)
        
        Mechanism: First-order death kinetics
        N(t) = N0 × exp(-kd × t)
        Death rate (kd) increases with low pH and high temperature
        
        Key drivers:
        - pH_evolution (if available - low pH kills probiotics)
        - protein_total (buffering protects)
        - pH_buffering_capacity
        """
        # Initial count (realistic for commercial products)
        initial = 8.5  # log CFU/g
        
        # pH stress (major factor, bounded)
        pH_proxy = 6.0 - (df['acidification_potential'] / 
                        df['acidification_potential'].mean()) * 1.5
        pH_stress = np.where(pH_proxy < 4.5,
                            np.exp(-(4.5 - pH_proxy) * 1.0),  # Reduced from 1.5
                            np.exp(-(pH_proxy - 4.5) * 0.2))
        pH_survival = 0.7 + pH_stress * 0.3  # Range: 0.7-1.0 (max 3-log loss)
        
        # Buffering protection (moderate)
        buffer_norm = df['pH_buffering_capacity'] / df['pH_buffering_capacity'].mean()
        buffer_protect = 0.9 + buffer_norm * 0.15  # Range: 0.9-1.2
        buffer_protect = np.clip(buffer_protect, 0.85, 1.15)
        
        # Protein protection (moderate)
        protein_norm = df['protein_total'] / df['protein_total'].mean()
        protein_protect = 0.95 + protein_norm * 0.1  # Range: 0.95-1.15
        protein_protect = np.clip(protein_protect, 0.9, 1.1)
        
        # Fat protection (small)
        fat_norm = df['fat_total'] / (df['fat_total'].mean() + 0.01)
        fat_protect = 0.95 + fat_norm * 0.08  # Range: 0.95-1.08
        fat_protect = np.clip(fat_protect, 0.93, 1.05)
        
        # Sugar (small)
        sugar_norm = df['sugar_total'] / df['sugar_total'].mean()
        sugar_protect = 0.97 + sugar_norm * 0.05
        sugar_protect = np.clip(sugar_protect, 0.95, 1.03)
        
        # Combined (max ~1.5 log reduction)
        viability = initial * pH_survival * buffer_protect * protein_protect * fat_protect * sugar_protect
        
        # Moderate noise
        noise = np.random.normal(0, 0.3, len(df))
        viability += noise
        
        return np.clip(viability, 6, 10)
    
    def predict_all_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict all targets and return updated dataframe
        
        Args:
            df: DataFrame with all engineered features
            
        Returns:
            DataFrame with all target predictions added
        """
        df_with_targets = df.copy()
        
        print("Predicting yogurt quality targets...")
        print("=" * 60)
        
        # Tier 1 - High Confidence
        print("\nTIER 1 TARGETS (High Confidence):")
        df_with_targets['pH_evolution'] = self.predict_pH_evolution(df)
        print("  ✓ pH_evolution predicted")
        
        df_with_targets['viscosity'] = self.predict_viscosity(df)
        print("  ✓ viscosity predicted")
        
        df_with_targets['fermentation_endpoint'] = self.predict_fermentation_endpoint(df)
        print("  ✓ fermentation_endpoint predicted")
        
        df_with_targets['fat_globule_size'] = self.predict_fat_globule_size(df)
        print("  ✓ fat_globule_size predicted")
        
        # Tier 2 - Moderate Confidence
        print("\nTIER 2 TARGETS (Moderate Confidence):")
        df_with_targets['firmness'] = self.predict_firmness(df)
        print("  ✓ firmness predicted")
        
        df_with_targets['syneresis'] = self.predict_syneresis(df)
        print("  ✓ syneresis predicted")
        
        L_star, a_star, b_star = self.predict_color_parameters(df)
        df_with_targets['color_L'] = L_star
        df_with_targets['color_a'] = a_star
        df_with_targets['color_b'] = b_star
        print("  ✓ color parameters predicted")

        df_with_targets['graininess_perception'] = self.predict_graininess_perception(df)
        print("  ✓ graininess_perception predicted")
        
        df_with_targets['lactic_acid_rate'] = self.predict_lactic_acid_rate(df)
        print("  ✓ lactic_acid_rate predicted")
        
        # Tier 3 - Complex Relationships
        print("\nTIER 3 TARGETS (Complex Relationships):")
        df_with_targets['overall_liking'] = self.predict_overall_liking(df)
        print("  ✓ overall_liking predicted")
        
        df_with_targets['purchase_intent'] = self.predict_purchase_intent(df)
        print("  ✓ purchase_intent predicted")
        
        df_with_targets['yeast_mold_growth'] = self.predict_yeast_mold_growth(df)
        print("  ✓ yeast_mold_growth predicted")
        
        df_with_targets['ph_drift_storage'] = self.predict_ph_drift_storage(df)
        print("  ✓ ph_drift_storage predicted")
        
        df_with_targets['yield_optimization'] = self.predict_yield_optimization(df)
        print("  ✓ yield_optimization predicted")
        
        df_with_targets['acetaldehyde_formation'] = self.predict_acetaldehyde_formation(df)
        print("  ✓ acetaldehyde_formation predicted")
        
        df_with_targets['probiotic_viability'] = self.predict_probiotic_viability(df)
        print("  ✓ probiotic_viability predicted")
        
        print("\n" + "=" * 60)
        print(f"All {len(self.target_specs)} targets predicted successfully!")
        
        return df_with_targets
    
    def validate_correlations(self, df: pd.DataFrame, feature_cols: list) -> Dict[str, float]:
        """
        Validate that predicted targets have expected correlation ranges
        
        Args:
            df: DataFrame with features and predicted targets
            feature_cols: List of feature column names
            
        Returns:
            Dictionary with correlation statistics
        """
        from scipy.stats import pearsonr
        
        print("\n" + "=" * 80)
        print("TARGET VALIDATION - Checking Correlation Ranges")
        print("=" * 80)
        
        results = {}
        
        # Get target columns (exclude features and Recipe Name)
        target_cols = [col for col in df.columns if col not in feature_cols and col != 'Recipe Name']
        
        for target in target_cols:
            if target not in self.target_specs:
                continue
                
            expected_min, expected_max = self.target_specs[target]['correlation']
            target_range = self.target_specs[target]['range']
            
            # Calculate correlation with key features
            # Use top 5 features for each target (simplified)
            feature_correlations = []
            
            for feature in feature_cols:
                if feature in df.columns and df[feature].std() > 0:
                    try:
                        corr, p_value = pearsonr(df[feature], df[target])
                        if not np.isnan(corr):
                            feature_correlations.append(abs(corr))
                    except:
                        pass
            
            if feature_correlations:
                # Use max correlation with any feature as proxy
                max_corr = max(feature_correlations)
                mean_corr = np.mean(sorted(feature_correlations, reverse=True)[:5])
                
                # Check if in expected range
                in_range = expected_min <= mean_corr <= expected_max
                
                # Check if values in expected range
                actual_min, actual_max = df[target].min(), df[target].max()
                range_check = (target_range[0] <= actual_min and actual_max <= target_range[1])
                
                status = "✓ PASS" if (in_range and range_check) else "⚠ CHECK"
                
                print(f"\n{target}:")
                print(f"  Expected Correlation: {expected_min:.2f} - {expected_max:.2f}")
                print(f"  Achieved Correlation: {mean_corr:.2f} (max: {max_corr:.2f})")
                print(f"  Expected Range: {target_range[0]:.2f} - {target_range[1]:.2f}")
                print(f"  Actual Range: {actual_min:.2f} - {actual_max:.2f}")
                print(f"  Status: {status}")
                
                results[target] = {
                    'mean_correlation': mean_corr,
                    'max_correlation': max_corr,
                    'expected_range': expected_min,
                    'in_range': in_range,
                    'value_range_ok': range_check
                }
        
        # Summary
        passed = sum(1 for r in results.values() if r['in_range'] and r['value_range_ok'])
        total = len(results)
        
        print("\n" + "=" * 80)
        print(f"VALIDATION SUMMARY: {passed}/{total} targets passed validation")
        print("=" * 80)
        
        return results


def main():
    """
    Example usage of the YogurtTargetPredictor
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict yogurt quality targets from formulation data')
    parser.add_argument('--input_csv', type=str, required=True, 
                       help='Path to input CSV file with engineered features')
    parser.add_argument('--output_csv', type=str, 
                       default='yogurt_data_with_targets.csv',
                       help='Path to output CSV file with predicted targets')
    parser.add_argument('--validate', action='store_true',
                       help='Run correlation validation after prediction')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} samples with {len(df.columns)} features")
    
    # Define feature columns (all except Recipe Name)
    feature_cols = [col for col in df.columns if col != 'Recipe Name']
    
    # Initialize predictor
    predictor = YogurtTargetPredictor(random_seed=42)
    
    # Predict all targets
    df_with_targets = predictor.predict_all_targets(df)
    
    # Save results
    df_with_targets.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")
    print(f"Dataset now contains {len(df_with_targets.columns)} columns")
    
    # Validation if requested
    if args.validate:
        validation_results = predictor.validate_correlations(df_with_targets, feature_cols)
    
    # Display sample predictions
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS (First 3 Recipes):")
    print("=" * 80)
    
    target_cols = [col for col in df_with_targets.columns if col not in feature_cols and col != 'Recipe Name']
    
    for idx in range(min(3, len(df_with_targets))):
        print(f"\nRecipe: {df_with_targets.iloc[idx]['Recipe Name']}")
        for target in target_cols:
            value = df_with_targets.iloc[idx][target]
            print(f"  {target}: {value:.2f}")
    
    print("\n" + "=" * 80)
    print("Target prediction complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()