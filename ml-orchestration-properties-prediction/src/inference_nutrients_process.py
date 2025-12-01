#!/usr/bin/env python3
"""
nutrition_process_mapper.py

Production-ready utility to map nutritional composition and process parameters
into a structured DataFrame for further analysis or modeling.

Author: Your Name
Date: YYYY-MM-DD
"""

import pandas as pd
import logging
from typing import Dict, Any

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("NutritionProcessMapper")

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def map_nutrients(plp_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Map nutritional composition from the product DataFrame.

    Args:
        plp_df (pd.DataFrame): Must contain nutrient columns.

    Returns:
        dict: Nutrient -> values (array or scalar).
    """
    required_cols = [
        "Energy (kcal)",
        #"Protein",
        #"Total Fat",
        #"Saturated Fat",
        "Total Carbohydrates",
        #"Total sugars",
        "Vitamin A",
        "Vitamin D"
    ]

    missing = [col for col in required_cols if col not in plp_df.columns]
    if missing:
        raise ValueError(f"Missing required nutrient columns: {missing}")

    logger.info("Mapping nutritional composition...")

    nutrients_map = {
        "Energy_kcal": plp_df["Energy (kcal)"].values,
        #"protein": plp_df["Protein"].values,
        #"fat": plp_df["Total Fat"].values,
        #"SFA": plp_df["Saturated Fat"].values,
        "carbohydrates": plp_df["Total Carbohydrates"].values,
        #"Total_sugars": plp_df["Total sugars"].values,
        "Vitamin_A": plp_df["Vitamin A"].values,
        "Vitamin_D": plp_df["Vitamin D"].values
    }

    return nutrients_map
  
  
def map_process_parameters(plp_globals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map process parameters from global configuration.

    Args:
        plp_globals (dict): Dictionary of process control values.

    Returns:
        dict: Process parameter -> values.
    """
    logger.info("Mapping process parameters...")
    
    process_map = {
        'heat_treatment_temp': plp_globals["vc_heat_treatment_temp"],
        'heat_treatment_time': plp_globals["vc_heat_treatment_time"],
        'homog_pressure_1': plp_globals["vc_homog_pressure_1"],
        'homog_pressure_2': plp_globals["vc_homog_pressure_2"],
        'homog_temp': plp_globals["vc_homog_temp"],
        'outlet_temp_target': plp_globals["vc_outlet_temp_target"],
        'flow_rate': plp_globals["vc_flow_rate"]
	}

    return process_map


def build_feature_dataframe(plp_df: pd.DataFrame,
                            plp_globals: Dict[str, Any]) -> pd.DataFrame:
    """
    Combine nutritional and process parameters into a unified DataFrame.

    Args:
        plp_df (pd.DataFrame): Product-level data (nutrients).
        plp_globals (dict): Process parameters.

    Returns:
        pd.DataFrame: Single-row DataFrame of mapped features.
    """
    nutrients_map = map_nutrients(plp_df)
    process_map = map_process_parameters(plp_globals)

    # Merge dictionaries
    features = {**nutrients_map, **process_map}
    
    feature_df = pd.DataFrame(process_map, index=[0])
    logger.info("Feature DataFrame constructed successfully.")
    return feature_df


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

try:
  plp_df = build_feature_dataframe(plp_df, plp_globals)
  logger.info(f"Final feature DataFrame:\n{feature_df}")
except Exception as e:
  logger.error(f"Feature mapping failed: {e}")