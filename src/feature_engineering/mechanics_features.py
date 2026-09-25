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
    rel_x = pd.to_numeric(res["release_pos_x"], errors="coerce")
    rel_z = pd.to_numeric(res["release_pos_z"], errors="coerce")
    res["release_3d_dist_from_origin"] = np.sqrt(rel_x**2 + rel_z**2).round(4)

    # 2. Circular decomposition of spin_axis (in degrees [0, 360))
    # Eliminates the 359° vs 1° numerical boundary issue in Euclidean / Covariance distance
    spin_axis_rad = np.radians(pd.to_numeric(res["spin_axis"], errors="coerce"))
    res["spin_axis_cos"] = np.cos(spin_axis_rad).round(4)
    res["spin_axis_sin"] = np.sin(spin_axis_rad).round(4)

    # 3. Exact VAA (Vertical Approach Angle) calculation at home plate (y = 1.417 ft)
    y0 = 50.0
    y_plate = 17.0 / 12.0 # 1.417 ft

    vy0 = pd.to_numeric(res["vy0"], errors="coerce").to_numpy()
    ay = pd.to_numeric(res["ay"], errors="coerce").to_numpy()
    vz0 = pd.to_numeric(res["vz0"], errors="coerce").to_numpy()
    az = pd.to_numeric(res["az"], errors="coerce").to_numpy()
    vx0 = pd.to_numeric(res["vx0"], errors="coerce").to_numpy()
    ax = pd.to_numeric(res["ax"], errors="coerce").to_numpy()

    with np.errstate(invalid='ignore', divide='ignore'):
        disc = vy0**2 - 2 * ay * (y0 - y_plate)
        disc = np.where(disc >= 0, disc, np.nan)
        
        t_plate = (-vy0 - np.sqrt(disc)) / np.where(np.abs(ay) < 1e-4, 1e-4, ay)
        t_plate_linear = (y0 - y_plate) / np.maximum(np.abs(vy0), 1e-4)
        t_plate = np.where(np.abs(ay) < 1e-4, t_plate_linear, t_plate)

        vy_plate = vy0 + ay * t_plate
        vz_plate = vz0 + az * t_plate
        vx_plate = vx0 + ax * t_plate

        vaa_deg = np.arctan(vz_plate / -vy_plate) * (180.0 / np.pi)
        haa_deg = np.arctan(vx_plate / -vy_plate) * (180.0 / np.pi)

    res["vaa"] = np.round(vaa_deg, 3)
    res["haa"] = np.round(haa_deg, 3)

    return res
