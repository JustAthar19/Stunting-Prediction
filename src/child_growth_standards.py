from __future__ import annotations
from pathlib import Path
from typing import Literal

import math
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEIGHT_AGE_DIR = PROJECT_ROOT / "data" / "height-Age"
WEIGHT_AGE_DIR = PROJECT_ROOT / "data" / "weight-Age"
WEIGHT_HEIGHT_DIR = PROJECT_ROOT / "data" / "weight-Height"

Sex = Literal[0, 1]  # 0 = female, 1 = male

def who_z_score(measure: float, L: float, M: float, S: float) -> float:
    """Compute WHO LMS z-score."""
    if L != 0:
        return (measure / (M ** L) - 1) / (L * S)
    return math.log(measure / M) / S


def who_haz(sex: Sex, age_months: int, height_cm: float) -> str:
    if sex == 0:
        path = HEIGHT_AGE_DIR / "Monthly-girls-height-z-score.csv"
    else:
        path = HEIGHT_AGE_DIR / "Monthly-boys-height-z-score.csv"

    data = pd.read_csv(path)
    row = data[data["Month"] == age_months]
    z = who_z_score(
        height_cm,
        float(row["L"].iloc[0]),
        float(row["M"].iloc[0]),
        float(row["S"].iloc[0]),
    )
    if z < -3:
        return "Sangat Pendek (Stunting Berat)"
    if -3 <= z < -2:
        return "Pendek (Stunting)"
    if -2 <= z <= 2:
        return "Normal"
    return "Sangat Tinggi"


def who_waz(sex: Sex, age_months: int, weight_kg: float) -> str:
    if sex == 0:
        path = WEIGHT_AGE_DIR / "Monthly-girls-weight-z-score.csv"
    else:
        path = WEIGHT_AGE_DIR / "Monthly-boys-weight-z-score.csv"

    data = pd.read_csv(path)
    row = data[data["Month"] == age_months]
    z = who_z_score(
        weight_kg,
        float(row["L"].iloc[0]),
        float(row["M"].iloc[0]),
        float(row["S"].iloc[0]),
    )

    if z < -3:
        return "Gizi Buruk"
    if -3 <= z < -2:
        return "Gizi Kurang"
    if -2 <= z <= 2:
        return "Gizi Baik"
    return "Berat Badan Lebih"


def who_whz(sex: Sex, weight_kg: float, height_cm: float) -> str:
    
    if sex == 0:
        path = WEIGHT_HEIGHT_DIR / "girls-zscore-weight-height.xlsx"
    else:
        path = WEIGHT_HEIGHT_DIR / "boys-zscore-weight-height-table.xlsx"

    data = pd.read_excel(path)
    row = data[data["Length"] == height_cm]
    z = who_z_score(
        weight_kg,
        float(row["L"].iloc[0]),
        float(row["M"].iloc[0]),
        float(row["S"].iloc[0]),
    )

    if z < -3:
        return "Gizi Buruk (Sangat Kurus)"
    if -3 <= z < -2:
        return "Gizi Kurang (Kurus)"
    if -2 <= z <= 1:
        return "Normal"
    if 1 < z <= 2:
        return "Berisiko Gizi Lebih"
    if 2 < z <= 3:
        return "Gizi Lebih"
    return "Obesitas"

