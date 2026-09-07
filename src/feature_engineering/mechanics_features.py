"""
Mechanics Feature Engineering (Level 0 Kinematics, Circular Spin & Physical Features)
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def compute_kinematics_and_vaa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Level 0 physical mechanics features:
    - Vertical Approach Angle (VAA) & Horizontal Approach Angle (HAA)
    - 3D Release Position vector & euclidean distance
    - Angular Circular Spin Components: spin_axis_cos, spin_axis_sin (avoids 359 vs 1 deg discontinuity)
    """
    res = df.copy()

    # 1. 3D Release vector magnitude
    rel_x = res["release_pos_x"].fillna(0)
    rel_z = res["release_pos_z"].fillna(0)
    res["release_3d_dist_from_origin"] = np.sqrt(rel_x**2 + rel_z**2).round(4)

    # 2. Circular decomposition of spin_axis (in degrees [0, 360))
    # Eliminates the 359° vs 1° numerical boundary issue in Euclidean / Covariance distance
    spin_axis_rad = np.radians(res["spin_axis"].fillna(0.0))
    res["spin_axis_cos"] = np.cos(spin_axis_rad).round(4)
    res["spin_axis_sin"] = np.sin(spin_axis_rad).round(4)

    # 3. Exact VAA (Vertical Approach Angle) calculation at home plate (y = 1.417 ft)
    y0 = 50.0
    y_plate = 17.0 / 12.0 # 1.417 ft

    vy0 = res["vy0"].fillna(-130.0).values
    ay = res["ay"].fillna(28.0).values
    vz0 = res["vz0"].fillna(-6.0).values
    az = res["az"].fillna(-20.0).values
    vx0 = res["vx0"].fillna(4.0).values
    ax = res["ax"].fillna(-10.0).values

    with np.errstate(invalid='ignore', divide='ignore'):
        disc = vy0**2 - 2 * ay * (y0 - y_plate)
        disc = np.maximum(disc, 1e-6)
        
        t_plate = (-vy0 - np.sqrt(disc)) / np.where(np.abs(ay) < 1e-4, 1e-4, ay)
        t_plate_linear = (y0 - y_plate) / np.maximum(np.abs(vy0), 1e-4)
        t_plate = np.where(np.abs(ay) < 1e-4, t_plate_linear, t_plate)

        vy_plate = vy0 + ay * t_plate
        vz_plate = vz0 + az * t_plate
        vx_plate = vx0 + ax * t_plate

        vaa_deg = np.arctan(vz_plate / -vy_plate) * (180.0 / np.pi)
        haa_deg = np.arctan(vx_plate / -vy_plate) * (180.0 / np.pi)

    nan_mask = np.isnan(vaa_deg)
    if nan_mask.any():
        approx_vaa = np.arctan((res["plate_z"].fillna(2.5) - res["release_pos_z"].fillna(5.8)) / 54.0) * (180.0 / np.pi)
        vaa_deg = np.where(nan_mask, approx_vaa, vaa_deg)

    res["vaa"] = np.round(vaa_deg, 3)
    res["haa"] = np.round(haa_deg, 3)

    return res
