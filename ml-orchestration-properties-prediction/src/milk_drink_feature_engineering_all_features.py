import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
from dataclasses import dataclass

@dataclass
class MilkDrinkConstants:
    """Constants for milk drink composition and process parameters"""
    
    # Composition factors (% by weight)
    MILK_FF_FAT: float = 0.035          # 3.5% fat in full-fat milk
    MILK_24_FAT: float = 0.024          # 2.4% fat in mixing milk
    MILK_SKM_FAT: float = 0.001         # 0.1% residual fat in skim milk
    FROZEN_CREAM_FAT: float = 0.35      # 35% fat in frozen cream
    MILK_CONC_FAT: float = 0.26         # 26% fat in milk concentrate
    
    MILK_FF_PROTEIN: float = 0.032      # 3.2% protein in full-fat milk
    MILK_24_PROTEIN: float = 0.032      # 3.2% protein in mixing milk  
    MILK_SKM_PROTEIN: float = 0.034     # 3.4% protein in skim milk
    MILK_CONC_PROTEIN: float = 0.35     # 35% protein in concentrate
    
    MILK_FF_WATER: float = 0.87         # 87% water in full-fat milk
    MILK_24_WATER: float = 0.87         # 87% water in mixing milk
    MILK_SKM_WATER: float = 0.91        # 91% water in skim milk
    
    # Process optimization constants
    OPTIMAL_VISCOSITY_RANGE: Tuple[float, float] = (20, 30)  # cP
    OPTIMAL_EMULSIFIER_RATIO: float = 0.005  # 0.5% of fat content
    COCOA_DENSITY: float = 1.3          # g/cm³ for suspension calculations

class MilkDrinkPredictorEngine:
    """
    Feature Engineering Engine for Milk-Based Drinks
    
    Computes predictors for physical performance, stability, and sensory analysis
    optimized for chocolate and strawberry milk drink formulations.
    """
    
    def __init__(self):
        """Initialize with milk drink constants and processing parameters"""
        self.constants = MilkDrinkConstants()
        self.required_columns = self._get_required_columns()
        
    def _get_required_columns(self) -> List[str]:
        """Return list of required columns for processing"""
        return [
            # Ingredient columns
            'Stabilizer', 'SKIMMED MILK POWDER', 'VITAMIN A POWDER Palmitate CWS GF',
            'VITAMIN D3 POWDER CWS Food Grade', 'AMF', 'CULTURE', 'RO Water (SFG)',
            'Past. Milk Cow FF', 'FROZEN CREAM', 'WHOLE MILK POWDER', 'RO Water',
            'Red Color', 'Strawberry Flavouring', 'Sugar Crystal ICUMSA 45',
            'Flavour Chocolate', 'COMPOUND COCOA', 'EMULSIFIER',
            'Past. Milk at 2.4% Fat for Mixing', 'Past. Milk Cow SKM',
            'Solid Milk Conc 100% AUSFIN (SFG)',
            
            # Process columns
            'Heat_Time_s', 'Heat_Temperature_Min_C', 'Heat_Temperature_Max_C',
            'Homogenization_Pressure_Primary_Bar', 'Homogenization_Pressure_Secondary_Bar',
            'Homogenization_Temperature_Min_C', 'Homogenization_Temperature_Max_C',
            'Outlet_Temperature_Min_C', 'Outlet_Temperature_Max_C', 'Flow_Rate_KL'
        ]
    
    def load_and_validate_data(self, file_path: str) -> pd.DataFrame:
        """
        Load dataset and validate required columns
        
        Args:
            file_path: Path to Excel or CSV file
            
        Returns:
            Validated DataFrame
        """
        
        # Load data
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("File must be Excel (.xlsx) or CSV (.csv)")
        
        # Check for required columns
        missing_columns = []
        for col in self.required_columns:
            if col not in df.columns:
                missing_columns.append(col)
        
        if missing_columns:
            print(f"Warning: Missing columns will be filled with zeros: {missing_columns}")
            for col in missing_columns:
                df[col] = 0.0
        
        # Fill NaN values with zeros
        df[self.required_columns] = df[self.required_columns].fillna(0)
        
        # Identify milk drink samples (no CULTURE)
        milk_drink_mask = (df['CULTURE'] == 0) | (df['CULTURE'].isna())
        df_milk_drinks = df[milk_drink_mask].copy()
        
        if len(df_milk_drinks) == 0:
            warnings.warn("No milk drink samples found (samples without CULTURE). Processing all samples.")
            df_milk_drinks = df.copy()
        
        print(f"Loaded {len(df_milk_drinks)} milk drink formulations for analysis")
        
        return df_milk_drinks
    
    def compute_fat_system_predictors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute fat-related predictors critical for emulsion stability"""
        
        # Fat content from different sources
        df['fat_from_milk_FF'] = df['Past. Milk Cow FF'] * self.constants.MILK_FF_FAT
        df['fat_from_milk_24'] = df['Past. Milk at 2.4% Fat for Mixing'] * self.constants.MILK_24_FAT
        df['fat_from_milk_SKM'] = df['Past. Milk Cow SKM'] * self.constants.MILK_SKM_FAT
        df['fat_from_cream'] = df['FROZEN CREAM'] * self.constants.FROZEN_CREAM_FAT
        df['fat_from_concentrate'] = df['Solid Milk Conc 100% AUSFIN (SFG)'] * self.constants.MILK_CONC_FAT
        
        # Total fat content (critical predictor)
        df['total_fat_content'] = (
            df['fat_from_milk_FF'] + df['fat_from_milk_24'] + df['fat_from_milk_SKM'] +
            df['fat_from_cream'] + df['fat_from_concentrate']
        )
        
        # Fat source complexity (affects emulsion behavior)
        fat_sources_used = (
            (df['fat_from_milk_FF'] > 0).astype(int) +
            (df['fat_from_milk_24'] > 0).astype(int) +
            (df['fat_from_cream'] > 0).astype(int) +
            (df['fat_from_concentrate'] > 0).astype(int)
        )
        df['fat_source_diversity'] = fat_sources_used / 4
        
        # Added vs natural fat ratio
        added_fat = df['fat_from_cream']  # FROZEN CREAM is added fat
        natural_fat = (df['fat_from_milk_FF'] + df['fat_from_milk_24'] + 
                      df['fat_from_milk_SKM'] + df['fat_from_concentrate'])
        df['added_vs_natural_fat_ratio'] = added_fat / (natural_fat + 0.001)
        
        # Fat distribution uniformity predictor
        fat_sources = [df['fat_from_milk_FF'], df['fat_from_milk_24'], 
                      df['fat_from_cream'], df['fat_from_concentrate']]
        df['fat_distribution_uniformity'] = 1 - np.std(fat_sources, axis=0) / (df['total_fat_content'] + 0.001)
        
        return df
    
    def compute_protein_system_predictors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute protein-related predictors for body and mouthfeel"""
        
        # Protein content from different sources
        df['protein_from_milk_FF'] = df['Past. Milk Cow FF'] * self.constants.MILK_FF_PROTEIN
        df['protein_from_milk_24'] = df['Past. Milk at 2.4% Fat for Mixing'] * self.constants.MILK_24_PROTEIN
        df['protein_from_milk_SKM'] = df['Past. Milk Cow SKM'] * self.constants.MILK_SKM_PROTEIN
        df['protein_from_concentrate'] = df['Solid Milk Conc 100% AUSFIN (SFG)'] * self.constants.MILK_CONC_PROTEIN
        
        # Total protein content
        df['total_protein_content'] = (
            df['protein_from_milk_FF'] + df['protein_from_milk_24'] + 
            df['protein_from_milk_SKM'] + df['protein_from_concentrate']
        )
        
        # Protein concentration (g/L, assuming density ~1.03 kg/L)
        total_liquid = (
            df['Past. Milk Cow FF'] + df['Past. Milk at 2.4% Fat for Mixing'] +
            df['Past. Milk Cow SKM'] + df['RO Water (SFG)'] + df['RO Water']
        )
        df['protein_concentration_gL'] = (df['total_protein_content'] * 1000) / (total_liquid + 0.001)
        
        # Protein network potential (for body/mouthfeel)
        df['protein_network_potential'] = (
            df['total_protein_content'] * 
            (1 + df['protein_concentration_gL'] * 0.01)  # Concentration enhancement
        )
        
        return df
    
    def compute_emulsion_system_predictors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute emulsion-related predictors (critical for milk drink stability)"""
        
        # Emulsifier to fat ratio (most critical predictor for stability)
        df['emulsifier_to_fat_ratio'] = df['EMULSIFIER'] / (df['total_fat_content'] + 0.001)
        
        # Emulsifier efficiency (optimal around 0.5% of fat)
        optimal_ratio = self.constants.OPTIMAL_EMULSIFIER_RATIO
        df['emulsifier_efficiency'] = 1 / (1 + abs(df['emulsifier_to_fat_ratio'] - optimal_ratio) * 100)
        
        # Emulsion complexity index
        df['emulsion_complexity_index'] = (df['total_fat_content'] * df['EMULSIFIER']) ** 0.5
        
        # Over-emulsification risk (critical for stability)
        df['over_emulsification_risk'] = np.where(
            df['emulsifier_to_fat_ratio'] > 0.015,  # 1.5% threshold
            (df['emulsifier_to_fat_ratio'] - 0.015) * 100,
            0
        )
        
        # Natural emulsifier content (from cream)
        df['natural_emulsifier_equivalent'] = df['fat_from_cream'] * 0.02  # Phospholipids in cream
        df['total_emulsifier_equivalent'] = df['EMULSIFIER'] + df['natural_emulsifier_equivalent']
        
        return df
    
    def compute_particle_system_predictors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute particle-related predictors (chocolate-specific)"""
        
        # Total batch weight estimation
        df['total_batch_weight'] = (
            df['Stabilizer'] + df['SKIMMED MILK POWDER'] + df['AMF'] + 
            df['Past. Milk Cow FF'] + df['FROZEN CREAM'] + df['WHOLE MILK POWDER'] +
            df['Sugar Crystal ICUMSA 45'] + df['Flavour Chocolate'] + df['COMPOUND COCOA'] +
            df['EMULSIFIER'] + df['Past. Milk at 2.4% Fat for Mixing'] + 
            df['Past. Milk Cow SKM'] + df['Solid Milk Conc 100% AUSFIN (SFG)'] +
            df['RO Water (SFG)'] + df['RO Water'] + df['Red Color'] + 
            df['Strawberry Flavouring'] + df['VITAMIN A POWDER Palmitate CWS GF'] + 
            df['VITAMIN D3 POWDER CWS Food Grade']
        )
        
        # Cocoa particle loading (chocolate drinks)
        df['cocoa_particle_loading'] = df['COMPOUND COCOA'] / (df['total_batch_weight'] + 0.001)
        
        # Particle to stabilizer ratio
        df['particle_to_stabilizer_ratio'] = df['COMPOUND COCOA'] / (df['Stabilizer'] + 0.001)
        
        # Particle suspension challenge index
        df['particle_suspension_challenge'] = (
            df['cocoa_particle_loading'] * 100 *  # Higher concentration = harder to suspend
            (1 + df['total_fat_content'] * 0.1)   # Fat affects suspension behavior
        )
        
        # Color particle loading (strawberry drinks)
        df['color_particle_loading'] = df['Red Color'] / (df['total_batch_weight'] + 0.001)
        
        return df
    
    def compute_flavor_system_predictors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute flavor-related predictors for taste and aroma"""
        
        # Product type identification
        df['is_chocolate'] = (
            (df['Flavour Chocolate'] > 0) | (df['COMPOUND COCOA'] > 0)
        ).astype(int)
        df['is_strawberry'] = (df['Strawberry Flavouring'] > 0).astype(int)
        df['is_plain'] = (1 - df['is_chocolate'] - df['is_strawberry']).clip(0, 1)
        
        # Flavor intensity predictors
        df['chocolate_flavor_intensity'] = (
            df['Flavour Chocolate'] + df['COMPOUND COCOA'] * 0.6  # Cocoa has lower flavor intensity
        ) / (df['total_batch_weight'] + 0.001)
        
        df['strawberry_flavor_intensity'] = (
            df['Strawberry Flavouring']
        ) / (df['total_batch_weight'] + 0.001)
        
        # Sugar concentration
        df['sugar_concentration'] = df['Sugar Crystal ICUMSA 45'] / (df['total_batch_weight'] + 0.001)
        
        # Flavor to sugar balance
        df['chocolate_sugar_balance'] = (
            df['chocolate_flavor_intensity'] / (df['sugar_concentration'] + 0.001)
        )
        df['strawberry_sugar_balance'] = (
            df['strawberry_flavor_intensity'] / (df['sugar_concentration'] + 0.001)
        )
        
        # Color-flavor harmony (strawberry)
        df['color_flavor_harmony'] = (
            df['Red Color'] * df['Strawberry Flavouring'] * 0.0001
        ).clip(0, 1)
        
        # Cocoa fat interaction (chocolate mouthfeel)
        df['cocoa_fat_interaction'] = (
            df['COMPOUND COCOA'] * df['total_fat_content'] * 0.001
        )
        
        return df
    
    def compute_process_predictors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute process-related predictors from your equipment parameters"""
        
        # Heat treatment intensity
        df['heat_temperature_avg'] = (df['Heat_Temperature_Min_C'] + df['Heat_Temperature_Max_C']) / 2
        df['pasteurization_intensity'] = (
            (df['heat_temperature_avg'] - 72) * df['Heat_Time_s'] / 15  # Reference: LTLT
        )
        
        # Homogenization energy
        df['homogenization_energy_density'] = (
            (df['Homogenization_Pressure_Primary_Bar'] + 
             df['Homogenization_Pressure_Secondary_Bar']) * df['Flow_Rate_KL']
        )
        
        # Homogenization pressure ratio
        df['homogenization_pressure_ratio'] = (
            df['Homogenization_Pressure_Primary_Bar'] / 
            (df['Homogenization_Pressure_Secondary_Bar'] + 0.001)
        )
        
        # Homogenization temperature
        df['homog_temperature_avg'] = (
            df['Homogenization_Temperature_Min_C'] + df['Homogenization_Temperature_Max_C']
        ) / 2
        
        # Fat globule size prediction
        # Simplified model: size inversely proportional to pressure^0.6
        df['fat_globule_size_predicted'] = (
            3.5 / ((df['Homogenization_Pressure_Primary_Bar'] / 100) ** 0.6) *
            (1 + abs(df['homog_temperature_avg'] - 65) * 0.02)  # Temperature effect
        )
        
        # Cooling rate estimation
        df['outlet_temperature_avg'] = (df['Outlet_Temperature_Min_C'] + df['Outlet_Temperature_Max_C']) / 2
        cooling_rate_estimate = (df['homog_temperature_avg'] - df['outlet_temperature_avg']) / 10  # Assume 10 min cooling
        df['cooling_rate_C_per_min'] = cooling_rate_estimate
        
        # Process efficiency score
        optimal_pressure_ratio = 3.5  # Typical optimal ratio
        df['process_efficiency_score'] = (
            # Pressure ratio optimization (0-0.3)
            (1 - abs(df['homogenization_pressure_ratio'] - optimal_pressure_ratio) / optimal_pressure_ratio).clip(0, 1) * 0.3 +
            # Adequate homogenization energy (0-0.4)
            (df['homogenization_energy_density'] / 7000).clip(0, 1) * 0.4 +
            # Heat treatment adequacy (0-0.3)
            (df['pasteurization_intensity'] / 400).clip(0, 1) * 0.3
        )
        
        return df
    
    def compute_stability_predictors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute stability-related predictors for shelf-life and quality"""
        
        # Density gradient calculation (cream separation risk)
        total_liquid = (
            df['Past. Milk Cow FF'] + df['Past. Milk at 2.4% Fat for Mixing'] +
            df['Past. Milk Cow SKM'] + df['RO Water (SFG)'] + df['RO Water'] +
            df['Solid Milk Conc 100% AUSFIN (SFG)']
        ) + 0.001
        
        cream_fraction = df['FROZEN CREAM'] / total_liquid
        skim_fraction = df['Past. Milk Cow SKM'] / total_liquid
        df['density_gradient_factor'] = abs(cream_fraction - skim_fraction) * 0.02
        
        # Stabilizer network strength
        df['stabilizer_to_liquid_ratio'] = df['Stabilizer'] / total_liquid
        df['stabilizer_network_strength'] = (
            df['stabilizer_to_liquid_ratio'] * 1000 *  # Convert to g/L
            (1 + df['protein_concentration_gL'] * 0.02)  # Protein synergy
        )
        
        # Emulsion stability index (primary stability predictor)
        df['emulsion_stability_index'] = (
            df['emulsifier_efficiency'] * 40 +           # Emulsifier optimization (0-40)
            (df['homogenization_energy_density'] / 200) +  # Mechanical stability (0-30)
            df['stabilizer_network_strength'] * 20 +     # Network support (0-20)
            -(df['density_gradient_factor'] * 100)       # Density penalty (0 to -10)
        ).clip(0, 100)
        
        # Phase separation risk
        df['phase_separation_risk'] = (
            df['density_gradient_factor'] * 50 +         # Density effects
            df['over_emulsification_risk'] * 30 +        # Over-emulsification
            (1 - df['emulsifier_efficiency']) * 30 +     # Poor emulsification
            df['particle_suspension_challenge'] * 20     # Particle effects
        ).clip(0, 100)
        
        # Temperature sensitivity
        df['temperature_sensitivity_index'] = (
            df['total_fat_content'] * 10 +               # Fat crystallization effects
            df['emulsifier_to_fat_ratio'] * 200 +        # Emulsifier temperature response
            df['protein_concentration_gL'] * 2           # Protein interactions
        ).clip(0, 100)
        
        return df
    
    def compute_physical_performance_predictors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute predictors for physical performance and flow characteristics"""
        
        # Viscosity prediction (simplified model)
        df['viscosity_predicted'] = (
            df['total_fat_content'] ** 0.4 * 3.5 +       # Fat contribution
            df['protein_concentration_gL'] ** 0.3 * 1.2 + # Protein network
            df['Stabilizer'] * 0.8 +                     # Direct thickening
            df['cocoa_particle_loading'] * 100 +         # Particle contribution
            15  # Base viscosity
        )
        
        # Pour characteristics score
        optimal_viscosity = np.mean(self.constants.OPTIMAL_VISCOSITY_RANGE)
        df['pour_characteristics_score'] = (
            10 / (1 + abs(df['viscosity_predicted'] - optimal_viscosity) * 0.15) *
            (1 - df['particle_suspension_challenge'] * 0.02) *  # Particle penalty
            (1 + df['process_efficiency_score'] * 0.1)          # Process bonus
        )
        
        # Mouthfeel quality index
        df['mouthfeel_quality_index'] = (
            df['total_fat_content'] * 12 +               # Richness
            df['emulsifier_efficiency'] * 20 +           # Smoothness
            df['protein_network_potential'] * 8 +        # Body
            -(df['cocoa_particle_loading'] * 150)        # Grittiness penalty
        ).clip(0, 100)
        
        # Surface tension predictor
        df['surface_tension_predicted'] = (
            45 - df['emulsifier_to_fat_ratio'] * 1000 +  # Emulsifier reduces surface tension
            df['protein_concentration_gL'] * 0.2 +       # Protein affects surface
            df['temperature_sensitivity_index'] * 0.05   # Temperature effects
        ).clip(25, 55)
        
        return df
    
    def compute_quality_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute overall quality indicators and formulation complexity"""
        
        # Formulation complexity score
        total_ingredients = (
            (df['Stabilizer'] > 0).astype(int) +
            (df['SKIMMED MILK POWDER'] > 0).astype(int) +
            (df['AMF'] > 0).astype(int) +
            (df['FROZEN CREAM'] > 0).astype(int) +
            (df['WHOLE MILK POWDER'] > 0).astype(int) +
            (df['Sugar Crystal ICUMSA 45'] > 0).astype(int) +
            (df['Flavour Chocolate'] > 0).astype(int) +
            (df['COMPOUND COCOA'] > 0).astype(int) +
            (df['EMULSIFIER'] > 0).astype(int) +
            (df['Strawberry Flavouring'] > 0).astype(int) +
            (df['Red Color'] > 0).astype(int)
        )
        df['formulation_complexity'] = total_ingredients / 11
        
        # Premium ingredient indicator
        df['premium_ingredient_score'] = (
            (df['FROZEN CREAM'] > 0).astype(int) * 0.4 +      # Premium fat source
            (df['Solid Milk Conc 100% AUSFIN (SFG)'] > 0).astype(int) * 0.3 +  # Concentrate
            (df['VITAMIN A POWDER Palmitate CWS GF'] > 0).astype(int) * 0.15 +  # Fortification
            (df['VITAMIN D3 POWDER CWS Food Grade'] > 0).astype(int) * 0.15     # Fortification
        )
        
        # Technical risk assessment
        df['technical_risk_score'] = (
            df['over_emulsification_risk'] * 0.4 +       # Emulsification issues
            df['phase_separation_risk'] * 0.003 +        # Separation risk  
            df['particle_suspension_challenge'] * 0.02 + # Particle issues
            (df['formulation_complexity'] > 0.8).astype(int) * 0.2  # Complexity risk
        ).clip(0, 10)
        
        # Fat level classification
        df['fat_level_category'] = pd.cut(
            df['total_fat_content'],
            bins=[0, 1.5, 2.5, 4.0, float('inf')],
            labels=['Low Fat', 'Reduced Fat', 'Standard', 'High Fat'],
            include_lowest=True
        )
        
        return df
    
    def compute_all_predictors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all milk drink predictors
        
        Args:
            df: DataFrame with raw ingredient and process data
            
        Returns:
            DataFrame with all computed predictors
        """
        
        print("Computing milk drink predictors...")
        
        # Apply all predictor computations
        df_processed = df.copy()
        
        print("  Computing fat system predictors...")
        df_processed = self.compute_fat_system_predictors(df_processed)
        
        print("  Computing protein system predictors...")
        df_processed = self.compute_protein_system_predictors(df_processed)
        
        print("  Computing emulsion system predictors...")
        df_processed = self.compute_emulsion_system_predictors(df_processed)
        
        print("  Computing particle system predictors...")
        df_processed = self.compute_particle_system_predictors(df_processed)
        
        print("  Computing flavor system predictors...")
        df_processed = self.compute_flavor_system_predictors(df_processed)
        
        print("  Computing process predictors...")
        df_processed = self.compute_process_predictors(df_processed)
        
        print("  Computing stability predictors...")
        df_processed = self.compute_stability_predictors(df_processed)
        
        print("  Computing physical performance predictors...")
        df_processed = self.compute_physical_performance_predictors(df_processed)
        
        print("  Computing quality indicators...")
        df_processed = self.compute_quality_indicators(df_processed)
        
        # Count computed predictors
        original_cols = len(df.columns)
        new_cols = len(df_processed.columns) - original_cols
        
        print(f"Successfully computed {new_cols} predictors for milk drinks")
        
        return df_processed
    
    def get_predictor_groups(self) -> Dict[str, List[str]]:
        """
        Return dictionary of predictor groups for easy analysis
        
        Returns:
            Dictionary with predictor group names and corresponding column lists
        """
        
        predictor_groups = {
            'fat_system': [
                'total_fat_content', 'fat_source_diversity', 'added_vs_natural_fat_ratio',
                'fat_distribution_uniformity', 'fat_from_cream', 'fat_from_milk_FF'
            ],
            
            'protein_system': [
                'total_protein_content', 'protein_concentration_gL', 'protein_network_potential'
            ],
            
            'emulsion_system': [
                'emulsifier_to_fat_ratio', 'emulsifier_efficiency', 'emulsion_complexity_index',
                'over_emulsification_risk', 'total_emulsifier_equivalent'
            ],
            
            'particle_system': [
                'cocoa_particle_loading', 'particle_to_stabilizer_ratio', 'particle_suspension_challenge',
                'color_particle_loading'
            ],
            
            'flavor_system': [
                'is_chocolate', 'is_strawberry', 'chocolate_flavor_intensity', 'strawberry_flavor_intensity',
                'sugar_concentration', 'color_flavor_harmony', 'cocoa_fat_interaction'
            ],
            
            'process_parameters': [
                'pasteurization_intensity', 'homogenization_energy_density', 'homogenization_pressure_ratio',
                'fat_globule_size_predicted', 'process_efficiency_score'
            ],
            
            'stability_predictors': [
                'emulsion_stability_index', 'phase_separation_risk', 'temperature_sensitivity_index',
                'stabilizer_network_strength', 'density_gradient_factor'
            ],
            
            'physical_performance': [
                'viscosity_predicted', 'pour_characteristics_score', 'mouthfeel_quality_index',
                'surface_tension_predicted'
            ],
            
            'quality_indicators': [
                'formulation_complexity', 'premium_ingredient_score', 'technical_risk_score',
                'fat_level_category'
            ]
        }
        
        return predictor_groups
    
    def generate_predictor_summary(self, df_with_predictors: pd.DataFrame) -> str:
        """Generate summary report of computed predictors"""
        
        predictor_groups = self.get_predictor_groups()
        all_predictors = [pred for group in predictor_groups.values() for pred in group]
        
        # Filter existing predictors
        existing_predictors = [pred for pred in all_predictors if pred in df_with_predictors.columns]
        
        report = "MILK DRINKS PREDICTOR SUMMARY REPORT\n"
        report += "=" * 50 + "\n\n"
        
        report += f"Dataset size: {len(df_with_predictors)} formulations\n"
        report += f"Original columns: {len(df_with_predictors.columns) - len(existing_predictors)}\n"
        report += f"Computed predictors: {len(existing_predictors)}\n\n"
        
        report += "PREDICTOR STATISTICS BY GROUP:\n"
        report += "-" * 40 + "\n"
        
        for group_name, predictors in predictor_groups.items():
            existing_group_predictors = [p for p in predictors if p in df_with_predictors.columns]
            if existing_group_predictors:
                report += f"\n{group_name.upper()} ({len(existing_group_predictors)} predictors):\n"
                for predictor in existing_group_predictors[:5]:  # Show first 5
                    values = df_with_predictors[predictor].dropna()
                    if len(values) > 0 and pd.api.types.is_numeric_dtype(values):
                        report += f"  {predictor:<25}: mean={values.mean():.3f}, std={values.std():.3f}, "
                        report += f"range=[{values.min():.3f}, {values.max():.3f}]\n"
                    else:
                        report += f"  {predictor:<25}: categorical/non-numeric\n"
                if len(existing_group_predictors) > 5:
                    report += f"  ... and {len(existing_group_predictors) - 5} more predictors\n"
        
        return report

def main():
    """Example usage of MilkDrinkPredictorEngine"""
    
    print("Milk Drink Predictor Engine")
    print("=" * 40)
    
    # Initialize predictor engine
    engine = MilkDrinkPredictorEngine()
    
    # Example usage
    # df = engine.load_and_validate_data('milk_drinks_dataset.xlsx')
    # df_with_predictors = engine.compute_all_predictors(df)
    # predictor_groups = engine.get_predictor_groups()
    # summary = engine.generate_predictor_summary(df_with_predictors)
    # print(summary)
    
    # Show available predictor groups
    predictor_groups = engine.get_predictor_groups()
    print("\nAvailable predictor groups:")
    for group_name, predictors in predictor_groups.items():
        print(f"\n{group_name}: {len(predictors)} predictors")
        print(f"  Key predictors: {predictors[:3]}")
    
    print(f"\nTotal predictors available: {sum(len(p) for p in predictor_groups.values())}")
    print("Engine ready for processing your milk drink dataset!")
    print("\nUsage:")
    print("  engine = MilkDrinkPredictorEngine()")
    print("  df = engine.load_and_validate_data('your_dataset.xlsx')")
    print("  df_with_predictors = engine.compute_all_predictors(df)")

if __name__ == "__main__":
    main()