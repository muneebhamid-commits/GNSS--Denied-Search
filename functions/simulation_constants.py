# ------------------------------------------------------------
# 1. SIMULATION CONSTANTS WITH DENSITY CONTROL
# ------------------------------------------------------------
def get_simulation_constants(density_factor=1.50):
    """
    Returns all global parameters in a structured dictionary.

    Args:
        density_factor: Controls candidate point density
            >1.0 = higher density (more points, closer spacing)
            <1.0 = lower density (fewer points, wider spacing)
            Default: 1.0 (normal density)
    """

    # Base spacing values that control density
    BASE_UGV_SPACING = 150
    BASE_BEACON_SPACING = 150

    # Apply density factor (inverse relationship with spacing)
    ugv_spacing = BASE_UGV_SPACING / density_factor
    beacon_spacing = BASE_BEACON_SPACING / density_factor

    # Safety limits to prevent impossible configurations
    MIN_SPACING = 50  # Don't go below 50m spacing (technical limit)
    MAX_SPACING = 2000  # Don't go above 2000m spacing (practical limit)

    ugv_spacing = max(MIN_SPACING, min(ugv_spacing, MAX_SPACING))
    beacon_spacing = max(MIN_SPACING, min(beacon_spacing, MAX_SPACING))

    # Warn if limits are hit
    if density_factor > 2.0 and ugv_spacing == MIN_SPACING:
        print(f"⚠️  Density factor {density_factor} would require spacing < {MIN_SPACING}m")
        print(f"   Using minimum technical spacing: {MIN_SPACING}m")

    if density_factor < 0.3 and ugv_spacing == MAX_SPACING:
        print(f"⚠️  Density factor {density_factor} would require spacing > {MAX_SPACING}m")
        print(f"   Using maximum practical spacing: {MAX_SPACING}m")

    constants = {
        # Density-controlled parameters
        "AREA": 5500.0,
        "UGV_SPACING": ugv_spacing,  # Now controlled by density_factor
        "BEACON_SPACING": beacon_spacing,  # Now controlled by density_factor
        "DENSITY_FACTOR": density_factor,  # Track the factor used
        "BASE_UGV_SPACING": BASE_UGV_SPACING,  # For reference
        "BASE_BEACON_SPACING": BASE_BEACON_SPACING,  # For reference

        # Fixed parameters (unchanged)
        "INTERNAL_MARGIN": 1000,
        "COVERAGE_RADIUS_UGV": 300,
        "COVERAGE_RADIUS_BEACON": 850,
        "UGV_COST": 3000,
        "BEACON_COST": 600,
        "UAV_COST": 10000,
        "TIME_BASED_OPERATION_COST": 5000,  # cost paid per hour of operation
        "UAV_OPERATIONAL_INCREMENT_COST": 1000,  # additional operational cost per UAV
        "ENERGY_COST": 10,                      # cost per kWh
        "DRIFT_RATE": 25,
        "DRIFT_TOLERANCE": 100,
        "UAV_CRUISE_SPEED": 15,
        "UAV_MIN_SPEED": 6,
        "SENSOR_RADIUS": 200,
        "OVERLAP_RATIO": 0.2,
        "MISSION_ALTITUDE": 50,
        "DESCENT_SPEED": 2,
        "CLIMB_SPEED": 4,
        "NUM_UAVS_MAX": 20,
        "MIN_UAV": 5,
        "BATTERY_CAPACITY": 3283200,
        "BATTERY_RESERVE_FRACTION": 0.6,
        "AVERAGE_FLIGHT_POWER": 3033,
    }



    return constants
