"""
Mechanics Feature Engineering (Level 0 Kinematics & Physical Features)
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def compute_kinematics_and_vaa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Level 0 physical mechanics features including:
    - Vertical Approach Angle (VAA)
    - Horizontal Approach Angle (HAA)
    - 3D Release Position vector & euclidean distance
    """
    res = df.copy()

    # 1. 3D Release vector magnitude
    # Using release_pos_x, release_pos_y (or 54.5 - extension), release_pos_z
    rel_x = res["release_pos_x"].fillna(0)
    rel_z = res["release_pos_z"].fillna(0)
    res["release_3d_dist_from_origin"] = np.sqrt(rel_x**2 + rel_z**2)

    # 2. Exact VAA (Vertical Approach Angle) calculation at home plate (y = 1.417 ft)
    # y0 = 50.0 ft (standard tracking origin)
    y0 = 50.0
    y_plate = 17.0 / 12.0 # 1.417 ft

    vy0 = res["vy0"].values
    ay = res["ay"].values
    vz0 = res["vz0"].values
    az = res["az"].values
    vx0 = res["vx0"].values
    ax = res["ax"].values

    # Time to plate: t = (-vy0 - sqrt(vy0^2 - 2*ay*(y0 - y_plate))) / ay
    # For numeric stability with zero or very small ay:
    with np.errstate(invalid='ignore', divide='ignore'):
        disc = vy0**2 - 2 * ay * (y0 - y_plate)
        # fallback if disc < 0
        disc = np.maximum(disc, 1e-6)
        
        # When ay != 0
        t_plate = (-vy0 - np.sqrt(disc)) / np.where(np.abs(ay) < 1e-4, 1e-4, ay)
        # fallback for linear when ay == 0
        t_plate_linear = (y0 - y_plate) / np.maximum(np.abs(vy0), 1e-4)
        t_plate = np.where(np.abs(ay) < 1e-4, t_plate_linear, t_plate)

        vy_plate = vy0 + ay * t_plate
        vz_plate = vz0 + az * t_plate
        vx_plate = vx0 + ax * t_plate

        # Approach angles in degrees
        vaa_deg = np.arctan(vz_plate / -vy_plate) * (180.0 / np.pi)
        haa_deg = np.arctan(vx_plate / -vy_plate) * (180.0 / np.pi)

    # If kinematic vectors are missing, estimate VAA using empirical regression
    # VAA ≈ (plate_z - release_pos_z) / 54.0 ...
    nan_mask = np.isnan(vaa_deg)
    if nan_mask.any():
        approx_vaa = np.arctan((res["plate_z"] - res["release_pos_z"]) / 54.0) * (180.0 / np.pi)
        vaa_deg = np.where(nan_mask, approx_vaa, vaa_deg)

    res["vaa"] = np.round(vaa_deg, 3)
    res["haa"] = np.round(haa_deg, 3)

    return res
