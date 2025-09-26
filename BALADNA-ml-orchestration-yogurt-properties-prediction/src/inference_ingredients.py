#!/usr/bin/env python3
"""
ingredient_mapper.py

Production-ready utility to map formulation materials to ingredient categories.

Author: MJI11
Date: 2025-09-25
"""

import pandas as pd
import logging
from typing import Dict, List, Any

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("IngredientMapper")



# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def map_ingredients(formula_df: pd.DataFrame,
                    mapping: Dict[str, List[Any]]) -> Dict[str, float]:
    """
    Map material numbers in the formulation to ingredient categories.

    Args:
        formula_df (pd.DataFrame): DataFrame containing at least:
            - 'materialNumber': material identifier codes
            - 'targetPercentWet': percentage contribution (wet basis)
        mapping (dict): Dictionary of ingredient -> [[material IDs], placeholder]

    Returns:
        dict: Ingredient -> percentage contribution (scaled * 100)
    """
    if "materialNumber" not in formula_df.columns or "targetPercentWet" not in formula_df.columns:
        raise ValueError("Input DataFrame must contain 'materialNumber' and 'targetPercentWet' columns.")

    logger.info("Starting ingredient mapping...")
    result: Dict[str, float] = {}

    for ingredient, (material_ids, _) in mapping.items():
        total_percent = 0.0

        for mat_id in material_ids:
            qty_sum = formula_df.loc[
                formula_df["materialNumber"] == mat_id, "targetPercentWet"
            ].astype(float).sum()
            total_percent += qty_sum

        # Store scaled percentage
        result[ingredient] = float(total_percent) * 100.0
        
    logger.info("Ingredient mapping complete.")
    return result
  

# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
# Define ingredient mapping
INGREDIENT_MAPPING: Dict[str, List[Any]] = {
  "VITAMIN_A_POWDER": [["MAT-00000843", "MAT-00000670"], 0],
  "VITAMIN_D3_POWDER": [["MAT-00000844", "MAT-00000569"], 0],
  "RO_Water_SFG": [["MAT-00000847"], 0],
  "Past_Milk_Cow_FF": [["MAT-00000848", "MAT-00000304"], 0],
  "Past_Milk_2_4_Fat": [["MAT-00000857"], 0],
  "Past_Milk_Cow_SKM": [["MAT-00000858", "MAT-00000422"], 0],
  "Sugar_Crystal_ICUMSA_45": [["MAT-00000853"], 0],
  "Strawberry_Flavouring": [["MAT-00000852"], 0],
  "Flavour_Chocolate": [["MAT-00000854"], 0],
  "COMPOUND_COCOA": [["MAT-00000855"], 0],
  "Solid_Milk_Conc_100": [["MAT-00000859"], 0],
  "SKIMMED_MILK_POWDER": [["MAT-00000842", "MAT-00000542"], 0],
  "WHOLE_MILK_POWDER": [["MAT-00000850", "MAT-00000764"], 0],
  "FROZEN_CREAM": [["MAT-00000849"], 0],
  "RO_Water": [["MAT-00000561"], 0],
  "Red_Color": [["MAT-00000851"], 0],
  "Stabilizer": [["MAT-00000841"], 0],
  "AMF": [["MAT-00000845"], 0],
  "CULTURE": [["MAT-00000846", "MAT-00000087", "MAT-00000121",
               "MAT-00000097", "MAT-00000421", "MAT-00000369"], 0],
  "EMULSIFIER": [["MAT-00000856"], 0],
}

# Compute ingredient contributions
ingredients_map = map_ingredients(formula_df=plp_df, mapping=INGREDIENT_MAPPING)
plp_df = pd.DataFrame([ingredients_map], index=[0])