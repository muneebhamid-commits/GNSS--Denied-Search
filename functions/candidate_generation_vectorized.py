import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import math
import pulp
# Test for git
# ------------------------------------------------------------
# IMPROVED Poisson-Disk Sampling
# ------------------------------------------------------------
def generate_hexagonal_grid(area_side, spacing):
    """
    Generate candidate positions in a hexagonal grid covering a square area.
    """
    dx = spacing
    dy = spacing * math.sqrt(3) / 2
    nx = int(area_side // dx) + 2
    ny = int(area_side // dy) + 2
    points = []
    for iy in range(ny):
        y = iy * dy
        x_offset = (dx / 2) if (iy % 2) else 0
        for ix in range(nx):
            x = ix * dx + x_offset
            if 0 <= x < area_side and 0 <= y < area_side:
                points.append((x, y))
    return np.array(points)

# ------------------------------------------------------------
# FULLY VECTORIZED Filtering Function
# ------------------------------------------------------------
def filter_candidates_vectorized(candidates, coverage_radius, margin, area_side):
    """
    Remove candidates inside the internal restricted zone and those whose coverage extends beyond the area.
    """
    candidates = np.asarray(candidates)
    x, y = candidates[:, 0], candidates[:, 1]
    # Internal restricted zone
    in_restricted = (x >= margin) & (x <= area_side - margin) & (y >= margin) & (y <= area_side - margin)
    # Coverage containment
    contained = (x >= coverage_radius) & (x <= area_side - coverage_radius) & (y >= coverage_radius) & (y <= area_side - coverage_radius)
    valid_mask = (~in_restricted) & contained
    valid_candidates = candidates[valid_mask]
    return valid_candidates, valid_mask

# ------------------------------------------------------------
# VECTORIZED Plotting (Minimal Loops)
# ------------------------------------------------------------
def plot_candidates_vectorized(area_side, ugv_raw, beacon_raw, ugv_valid, beacon_valid,
                               margin, coverage_radius_ugv, coverage_radius_beacon, params):
    if ugv_raw.size == 0 and beacon_raw.size == 0:
        return None
    if ugv_valid.size == 0 and beacon_valid.size == 0:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    # Plot raw
    ax1.set_title("Raw Candidates")
    if ugv_raw.size > 0:
        ax1.scatter(ugv_raw[:, 0], ugv_raw[:, 1], c='blue', label='UGV Raw', s=30, marker='o', alpha=0.7, edgecolors='black', linewidths=0.5)
    if beacon_raw.size > 0:
        ax1.scatter(beacon_raw[:, 0], beacon_raw[:, 1], c='green', label='Beacon Raw', s=30, marker='^', alpha=0.7, edgecolors='black', linewidths=0.5)
    # Restricted zone
    if margin > 0:
        rect = plt.Rectangle((margin, margin), area_side-2*margin, area_side-2*margin, color='red', alpha=0.2, label='Restricted Zone')
        ax1.add_patch(rect)
    ax1.set_xlim(0, area_side)
    ax1.set_ylim(0, area_side)
    ax1.legend()
    ax1.set_aspect('equal')
    # Plot filtered
    ax2.set_title("Filtered Candidates")
    if ugv_valid.size > 0:
        ax2.scatter(ugv_valid[:, 0], ugv_valid[:, 1], c='blue', label='UGV Valid', s=30, marker='o', alpha=0.7, edgecolors='black', linewidths=0.5)
    if beacon_valid.size > 0:
        ax2.scatter(beacon_valid[:, 0], beacon_valid[:, 1], c='green', label='Beacon Valid', s=30, marker='^', alpha=0.7, edgecolors='black', linewidths=0.5)
    if margin > 0:
        rect2 = plt.Rectangle((margin, margin), area_side-2*margin, area_side-2*margin, color='red', alpha=0.2, label='Restricted Zone')
        ax2.add_patch(rect2)
    ax2.set_xlim(0, area_side)
    ax2.set_ylim(0, area_side)
    ax2.legend()
    ax2.set_aspect('equal')
    plt.tight_layout()
    plt.show(block=True)
    return fig

# ------------------------------------------------------------
# MAIN VECTORIZED PIPELINE
# ------------------------------------------------------------
def generate_and_filter_candidates_vectorized(params, plot_results=True):
    area_side = params["AREA"]
    ugv_spacing = params["UGV_SPACING"]
    beacon_spacing = params["BEACON_SPACING"]
    margin = params["INTERNAL_MARGIN"]
    coverage_radius_ugv = params["COVERAGE_RADIUS_UGV"]
    coverage_radius_beacon = params["COVERAGE_RADIUS_BEACON"]

    print(f"UGV candidate spacing: {ugv_spacing}")
    print(f"Beacon candidate spacing: {beacon_spacing}")

    # Generate candidates
    ugv_raw = generate_hexagonal_grid(area_side, ugv_spacing)
    beacon_raw = generate_hexagonal_grid(area_side, beacon_spacing)

    # Filter candidates
    ugv_valid, _ = filter_candidates_vectorized(ugv_raw, coverage_radius_ugv, margin, area_side)
    beacon_valid, _ = filter_candidates_vectorized(beacon_raw, coverage_radius_beacon, margin, area_side)

    print(f"UGV candidates after filtering: {len(ugv_valid)}")
    print(f"Beacon candidates after filtering: {len(beacon_valid)}")

    if plot_results:
        plot_candidates_vectorized(area_side, ugv_raw, beacon_raw, ugv_valid, beacon_valid, margin, coverage_radius_ugv, coverage_radius_beacon, params)
        plt.savefig("candidates_generated.png") # Save the plot

    return ugv_valid, beacon_valid

    ugv_raw = generate_hexagonal_grid(area_side, ugv_spacing)
    beacon_raw = generate_hexagonal_grid(area_side, beacon_spacing)

    ugv_valid, _ = filter_candidates_vectorized(ugv_raw, coverage_radius_ugv, margin, area_side)
    beacon_valid, _ = filter_candidates_vectorized(beacon_raw, coverage_radius_beacon, margin, area_side)

    if plot_results:
        plot_candidates_vectorized(area_side, ugv_raw, beacon_raw, ugv_valid, beacon_valid, margin, coverage_radius_ugv, coverage_radius_beacon, params)

    return ugv_valid, beacon_valid

# ------------------------------------------------------------
# DEBUG FUNCTION FOR COST ANALYSIS
# ------------------------------------------------------------
def debug_cost_analysis(prob, x_ugv, y_beacon, n_ugv, n_beacon, COST_UGV, COST_BEACON):
    # ...function body as provided...
    # ...
    pass
