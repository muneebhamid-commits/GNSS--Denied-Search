import matplotlib.pyplot as plt

def plot_milp_output(milp_result, ugv_candidates, beacon_candidates):
    plt.figure(figsize=(8,8))
    plt.scatter(ugv_candidates[:,0], ugv_candidates[:,1], c='lightblue', s=20, label='UGV candidates')
    plt.scatter(beacon_candidates[:,0], beacon_candidates[:,1], c='lightgreen', s=20, label='Beacon candidates')
    for i in milp_result['selected_ugvs']:
        plt.scatter(ugv_candidates[i,0], ugv_candidates[i,1], c='blue', s=100, edgecolor='k', label='Selected UGV')
    for j in milp_result['selected_beacons']:
        plt.scatter(beacon_candidates[j,0], beacon_candidates[j,1], c='green', marker='^', s=100, edgecolor='k', label='Selected Beacon')
    plt.title('MILP Output: Selected Support Nodes')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

# Test for git
# # mission_simulator.py
import numpy as np
import matplotlib.pyplot as plt
from pulp import (LpProblem, LpVariable, LpMinimize, lpSum,
                  LpBinary, LpInteger, value, PULP_CBC_CMD)


def solve_milp(params, ugv_valid, beacon_valid, uav_vars, plot=True, time_limit=120):
    """
    MILP solver with drift-aware coverage (Option A).
    Signature: solve_milp(params, ugv_valid, beacon_valid, uav_vars)
    - ugv_valid, beacon_valid: numpy arrays of candidate coords (N x 2)
    - uav_vars: list of pulp LpVariable binary vars (length = params["num_uavs_max"])
    """

    # Defensive conversions
    ugv_candidates = np.asarray(ugv_valid) if ugv_valid is not None and len(ugv_valid) > 0 else np.zeros((0, 2))
    beacon_candidates = np.asarray(beacon_valid) if beacon_valid is not None and len(beacon_valid) > 0 else np.zeros((0, 2))

    AREA = params["AREA"]
    GRID_SPACING = params.get("GRID_SPACING", 100.0)

    # Build scan grid (same approach as your existing code)
    x_points = np.arange(0, AREA + 1e-9, GRID_SPACING)
    y_points = np.arange(0, AREA + 1e-9, GRID_SPACING)
    X, Y = np.meshgrid(x_points, y_points)
    scan_points = np.column_stack((X.ravel(), Y.ravel()))
    num_scan_points = scan_points.shape[0]

    UAV_alt = params.get("MISSION_ALTITUDE", 50.0)
    UGV_alt = 0.0

    n_ugv = ugv_candidates.shape[0]
    n_beacon = beacon_candidates.shape[0]
    n_uav = len(uav_vars)

    # ---------------------------
    # Compute conservative drift radius (Option A)
    # ---------------------------
    r = params["DRIFT_RATE"]            # m/min
    D_max = params["DRIFT_TOLERANCE"]   # m
    v_min = params["UAV_MIN_SPEED"]     # m/s

    # Convert and compute:
    # R_drift = v_min * 60 * D_max / r
    R_drift = float(v_min * 60.0 * D_max / r)
    R_ugv =  R_drift + params["COVERAGE_RADIUS_UGV"]
    R_beacon = R_drift + params["COVERAGE_RADIUS_BEACON"]

    print(f"[INFO] Computed conservative drift radius R_drift = {R_drift:.2f} m (v_min={v_min} m/s)")

    # ---------------------------
    # Precompute distances (for coverage checks)
    # ---------------------------
    if n_ugv > 0:
        dist_UGV_3D = np.sqrt((scan_points[:, 0, None] - ugv_candidates[:, 0][None, :]) ** 2 +
                              (scan_points[:, 1, None] - ugv_candidates[:, 1][None, :]) ** 2 +
                              (UAV_alt - UGV_alt) ** 2)
        dist_UGV_2D = np.sqrt((scan_points[:, 0, None] - ugv_candidates[:, 0][None, :]) ** 2 +
                              (scan_points[:, 1, None] - ugv_candidates[:, 1][None, :]) ** 2)
    else:
        dist_UGV_3D = np.zeros((num_scan_points, 0))
        dist_UGV_2D = np.zeros((num_scan_points, 0))

    if n_beacon > 0:
        dist_Beacon_2D = np.sqrt((scan_points[:, 0, None] - beacon_candidates[:, 0][None, :]) ** 2 +
                                 (scan_points[:, 1, None] - beacon_candidates[:, 1][None, :]) ** 2)
    else:
        dist_Beacon_2D = np.zeros((num_scan_points, 0))

    # ---------------------------
    # Build MILP
    # ---------------------------
    prob = LpProblem("UAV_UGV_Beacon_Placement_Drift", LpMinimize)

    # decision variables
    x_ugv = [LpVariable(f"x_ugv_{i}", cat=LpBinary) for i in range(n_ugv)]
    y_beacon = [LpVariable(f"y_beacon_{j}", cat=LpBinary) for j in range(n_beacon)]
    # uav_vars are passed in (list of binary LpVariables)
    uav_vars_list = uav_vars

    # N_UAV as sum of provided uav vars (so main controls number of available UAV slots)
    N_UAV = lpSum(uav_vars_list)

    # objective: keep original cost structure names from params (fallbacks if missing)
    COST_UGV = params.get("UGV_COST", params.get("COST_UGV", 3000))
    COST_BEACON = params.get("BEACON_COST", params.get("COST_BEACON", 600))
    COST_UAV = params.get("UAV_COST", params.get("COST_UAV", 1000))

    prob += (lpSum([COST_UGV * x_ugv[i] for i in range(n_ugv)]) +
             lpSum([COST_BEACON * y_beacon[j] for j in range(n_beacon)])), "Total_Cost"

    # ---------------------------
    # Constraints
    # ---------------------------

    # 1) Minimum UAVs
    MIN_UAV = int(params.get("MIN_UAV", 1))
    prob += N_UAV >= MIN_UAV, "Min_UAV_Required"

    # 2) Energy-based battery swap feasibility constraint (70% reserve)
    P_avg = params["AVERAGE_FLIGHT_POWER"]  # Watts
    v_min = params["UAV_MIN_SPEED"]  # m/s
    battery_cap = params["BATTERY_CAPACITY"]  # Joules

    reserve_fraction = params["BATTERY_RESERVE_FRACTION"]  # UAV must keep this much reserve
    energy_limit = (1 - reserve_fraction) * battery_cap
    # => energy_limit = 0.30 * Emax

    # Energy per meter (constant)
    energy_per_meter = P_avg / v_min

    # Energy cost matrix: shape (num_scan_points, n_ugv)
    energy_UGV = dist_UGV_3D * energy_per_meter

    for idx in range(num_scan_points):

        reachable_ugvs = [
            j for j in range(n_ugv)
            if energy_UGV[idx, j] <= energy_limit
            # i.e.: E <= 0.3 * Emax
        ]

        if n_ugv == 0:
            # no UGVs → hard infeasible constraint
            prob += lpSum([]) >= 1, f"BatterySwap_Point_{idx}_noUGV"

        elif len(reachable_ugvs) > 0:
            # at least one UGV reachable
            prob += lpSum(x_ugv[j] for j in reachable_ugvs) >= 1, \
                f"BatterySwap_Point_{idx}"

        else:
            # unreachable → symbolic infeasible constraint
            prob += lpSum([]) >= 1, f"BatterySwap_Point_{idx}_unreachable"

    # 3) Drift correction: each scan point must have at least one support node (UGV or Beacon) within R_drift (2D)
    for idx in range(num_scan_points):
        reachable_ugvs = [j for j in range(n_ugv) if dist_UGV_2D[idx, j] <= R_ugv] if n_ugv > 0 else []
        reachable_beacons = [j for j in range(n_beacon) if dist_Beacon_2D[idx, j] <= R_beacon] if n_beacon > 0 else []
        if (len(reachable_ugvs) + len(reachable_beacons)) > 0:
            prob += lpSum([x_ugv[j] for j in reachable_ugvs] + [y_beacon[j] for j in reachable_beacons]) >= 1, f"DriftCorrection_Point_{idx}"
        else:
            prob += lpSum([]) >= 1, f"DriftCorrection_Point_{idx}_unreachable"

    # 4) Non-overlap of drift circles (UGV-UGV, Beacon-Beacon, UGV-Beacon)
    # Use simple linear exclusion x_i + x_j <= 1 when centers are closer than sum of radii
    EPS = params.get("NONOVERLAP_EPS", 1e-6)

    # UGV-UGV
    # for i in range(n_ugv):
        # for j in range(i + 1, n_ugv):
            # dij = float(np.linalg.norm(ugv_candidates[i] - ugv_candidates[j]))
            # if dij < (2 * R_ugv) + EPS:
                # prob += x_ugv[i] + x_ugv[j] <= 1, f"UGV_NoOverlap_{i}_{j}"

    # Beacon-Beacon
    # for i in range(n_beacon):
        # for j in range(i + 1, n_beacon):
            # dij = float(np.linalg.norm(beacon_candidates[i] - beacon_candidates[j]))
            # if dij < (2 * R_beacon) + EPS:
                # prob += y_beacon[i] + y_beacon[j] <= 1, f"Beacon_NoOverlap_{i}_{j}"

    # UGV-Beacon
    # for i in range(n_ugv):
        # for j in range(n_beacon):
            # dij = float(np.linalg.norm(ugv_candidates[i] - beacon_candidates[j]))
            # if dij < (R_ugv + R_beacon) + EPS:
                # prob += x_ugv[i] + y_beacon[j] <= 1, f"UGV_Beacon_NoOverlap_{i}_{j}"

    # ---------------------------
    # Solve
    # ---------------------------
    solver = PULP_CBC_CMD(msg=True, timeLimit=time_limit)
    prob.solve(solver)
    print("\n" + "=" * 60)
    print("DEBUG: COST ANALYSIS")
    print("=" * 60)

    # Check which UGVs are selected
    selected_ugv_indices = [i for i in range(n_ugv) if x_ugv[i].varValue == 1]
    selected_beacon_indices = [j for j in range(n_beacon) if y_beacon[j].varValue == 1]

    print(f"UGV cost per unit: {COST_UGV}")
    print(f"Beacon cost per unit: {COST_BEACON}")
    print(f"UGVs selected: {len(selected_ugv_indices)}")
    print(f"Beacons selected: {len(selected_beacon_indices)}")

    # Manual cost calculation
    manual_ugv_cost = len(selected_ugv_indices) * COST_UGV
    manual_beacon_cost = len(selected_beacon_indices) * COST_BEACON
    manual_total_cost = manual_ugv_cost + manual_beacon_cost

    print(f"Manual UGV cost: {manual_ugv_cost}")
    print(f"Manual Beacon cost: {manual_beacon_cost}")
    print(f"Manual total cost: {manual_total_cost}")

    # Check what the solver reports
    solver_total_cost = value(prob.objective)
    print(f"Solver reported total cost: {solver_total_cost}")

    # Check if there's a discrepancy
    if abs(manual_total_cost - solver_total_cost) > 1e-6:
        print("❌ DISCREPANCY DETECTED: Manual vs Solver costs don't match!")
        print(f"   Difference: {abs(manual_total_cost - solver_total_cost)}")
    else:
        print("✅ Costs match: Manual calculation = Solver reported cost")

    # Debug individual variable values
    print(f"\nDEBUG: First 10 UGV variables:")
    for i in range(min(10, n_ugv)):
        print(f"  x_ugv[{i}] = {x_ugv[i].varValue}")

    print(f"DEBUG: First 10 Beacon variables:")
    for j in range(min(10, n_beacon)):
        print(f"  y_beacon[{j}] = {y_beacon[j].varValue}")

    # Check if the objective function is what we think it is
    print(f"\nDEBUG: Objective function components:")
    ugv_cost_in_obj = sum(COST_UGV * x_ugv[i].varValue for i in range(n_ugv))
    beacon_cost_in_obj = sum(COST_BEACON * y_beacon[j].varValue for j in range(n_beacon))
    print(f"  UGV cost component: {ugv_cost_in_obj}")
    print(f"  Beacon cost component: {beacon_cost_in_obj}")
    print(f"  Sum: {ugv_cost_in_obj + beacon_cost_in_obj}")

    print("=" * 60)
    # ---------------------------
    # Extract solution
    # ---------------------------
    selected_ugvs = [i for i in range(n_ugv) if value(x_ugv[i]) is not None and value(x_ugv[i]) > 0.5]
    selected_beacons = [j for j in range(n_beacon) if value(y_beacon[j]) is not None and value(y_beacon[j]) > 0.5]
    num_uavs = int(sum(1 for v in uav_vars_list if value(v) is not None and value(v) > 0.5))
    total_cost = float(value(prob.objective)) if value(prob.objective) is not None else None
    status = prob.status

    # ---------------------------
    # Optional plotting (clean) -> THREE-SUBPLOTS figure (Drift | Coverage | Battery Swap)
    # ---------------------------
    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(26, 9))

        # Common settings for all subplots
        for ax in axs:
            ax.set_xlim(0, AREA)
            ax.set_ylim(0, AREA)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        # Draw restricted zone on all 3 subplots
        margin = params.get("INTERNAL_MARGIN", None)
        if margin:
            restricted_min = margin
            restricted_max = AREA - margin
            from matplotlib.patches import Rectangle
            for ax in axs:
                ax.add_patch(Rectangle(
                    (restricted_min, restricted_min),
                    restricted_max - restricted_min,
                    restricted_max - restricted_min,
                    facecolor='lightcoral', alpha=0.18
                ))

        # ------------------------------------------------------------
        # SUBPLOT 1: DRIFT RADIUS
        # ------------------------------------------------------------
        axs[0].set_title(f"Drift Correction Zones (R_drift={R_drift:.1f} m)")

        if n_ugv > 0:
            axs[0].scatter(ugv_candidates[:, 0], ugv_candidates[:, 1],
                           c='lightblue', s=20, alpha=0.5, label='UGV candidates')
        if n_beacon > 0:
            axs[0].scatter(beacon_candidates[:, 0], beacon_candidates[:, 1],
                           c='lightgreen', s=20, alpha=0.5, label='Beacon candidates')

        # selected UGV drift circles
        for k, i in enumerate(selected_ugvs):
            x, y = ugv_candidates[i]
            axs[0].scatter([x], [y], c='blue', s=100, edgecolor='k',
                           label='Selected UGV' if k == 0 else None)
            axs[0].add_patch(plt.Circle((x, y), R_ugv, color='blue', alpha=0.18))

        # selected Beacons drift circles
        for k, j in enumerate(selected_beacons):
            x, y = beacon_candidates[j]
            axs[0].scatter([x], [y], c='green', marker='^', s=100,
                           edgecolor='k', label='Selected Beacon' if k == 0 else None)
            axs[0].add_patch(plt.Circle((x, y), R_beacon, color='green', alpha=0.18))

        axs[0].legend(loc='upper right', fontsize=9)

        # ------------------------------------------------------------
        # SUBPLOT 2: COVERAGE RADIUS (existing)
        # ------------------------------------------------------------
        axs[1].set_title("Coverage Zones (UGV vs Beacon)")

        COV_UGV = params["COVERAGE_RADIUS_UGV"]
        COV_BEACON = params["COVERAGE_RADIUS_BEACON"]

        if n_ugv > 0:
            axs[1].scatter(ugv_candidates[:, 0], ugv_candidates[:, 1],
                           c='lightblue', s=20, alpha=0.5, label='UGV candidates')
        if n_beacon > 0:
            axs[1].scatter(beacon_candidates[:, 0], beacon_candidates[:, 1],
                           c='lightgreen', s=20, alpha=0.5, label='Beacon candidates')

        for k, i in enumerate(selected_ugvs):
            x, y = ugv_candidates[i]
            axs[1].scatter([x], [y], c='blue', s=100, edgecolor='k',
                           label='Selected UGV' if k == 0 else None)
            axs[1].add_patch(plt.Circle((x, y), COV_UGV,
                                        color='blue', fill=False, alpha=0.22, linestyle='--', linewidth=2))

        for k, j in enumerate(selected_beacons):
            x, y = beacon_candidates[j]
            axs[1].scatter([x], [y], c='green', marker='^', s=100, edgecolor='k',
                           label='Selected Beacon' if k == 0 else None)
            axs[1].add_patch(plt.Circle((x, y), COV_BEACON,
                                        color='green', fill=False, alpha=0.22, linestyle='--', linewidth=2))

        axs[1].legend(loc='upper right', fontsize=9)

        # ------------------------------------------------------------
        # SUBPLOT 3: BATTERY-SWAP RANGE (UGVs only)
        # ------------------------------------------------------------
        BATTERY_SAFE_RANGE = (1 - params["BATTERY_RESERVE_FRACTION"]) * params["BATTERY_CAPACITY"] * params["UAV_MIN_SPEED"] / params["AVERAGE_FLIGHT_POWER"]
        axs[2].set_title(f"Battery Swap Range (radius={BATTERY_SAFE_RANGE} m)")

        # show all UGV candidates
        if n_ugv > 0:
            axs[2].scatter(ugv_candidates[:, 0], ugv_candidates[:, 1],
                           c='lightblue', s=20, alpha=0.5, label='UGV candidates')

        # highlight selected UGVs + swap range
        for k, i in enumerate(selected_ugvs):
            x, y = ugv_candidates[i]
            axs[2].scatter([x], [y], c='blue', s=100, edgecolor='k',
                           label='Selected UGV' if k == 0 else None)

            axs[2].add_patch(plt.Circle((x, y), BATTERY_SAFE_RANGE,
                                        color='purple', alpha=0.18, label='Battery range' if k == 0 else None))

        axs[2].legend(loc='upper right', fontsize=9)

        plt.tight_layout()
        plt.show()

    # ---------------------------
    # Return
    # ---------------------------
    return {
        "selected_ugvs": selected_ugvs,
        "selected_beacons": selected_beacons,
        "num_uavs": num_uavs,
        "total_cost": total_cost,
        "status": status
    }
