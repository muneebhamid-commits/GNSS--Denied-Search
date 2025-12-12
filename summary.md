## Drift Plotting in VNS Dashboard

In the current code, each UAV tracks its own drift value, which accumulates as it moves. The "maximum drift" shown in the VNS dashboard refers to the highest drift value among all UAVs at a given time step (i.e., the worst-case drift for any single UAV). Drift can exceed the threshold (e.g., 100 meters) if a UAV does not visit a support node in time, because the code does not forcibly cap the drift variable itself. The threshold is used to trigger correction, but drift continues to accumulate until reset.

## Confidence Update Formula

When a UAV visits a grid cell, the confidence of that cell is increased by:

    gain = max(0, 100 - UAV drift)

This means the higher the UAV's drift, the less confidence is gained per visit. When the cell's confidence reaches 95% or more, it is considered covered.
# Project Summary: GNSS-Denied UAV Search Simulation

## Overview
This project simulates and optimizes a **Cooperative UAV Search Mission** in a GNSS-denied environment. In such environments, UAVs cannot rely on GPS for positioning and suffer from navigation drift (accumulating position error) over time. To maintain effective coverage, UAVs must periodically visit support infrastructure (**UGVs** or **Beacons**) to reset their drift and recharge their batteries.

The goal is to find the optimal configuration of support nodes (UGVs and Beacons) and the optimal fleet size (5-13 UAVs) to cover **>97% of the target area** within **100 minutes** at the **lowest possible cost**.

## System Architecture

### 1. Initialization: Static Placement (MILP)
*   **File**: `functions/milp_placement.py`
*   **Algorithm**: Mixed-Integer Linear Programming (using `pulp`).
*   **Purpose**: Generates a mathematically optimal *initial* placement of UGVs and Beacons based purely on geometric coverage. This provides a "good enough" starting point for the dynamic optimization.

### 2. Dynamic Simulation (The "Judge")
*   **File**: `functions/mission_optimizer.py`
*   **Algorithm**: Centroidal Voronoi Tessellation (CVT).
*   **Key Mechanics**:
    *   **CVT Movement**: UAVs continuously move to the centroid of their Voronoi cell, ensuring optimal area distribution.
    *   **Drift Modeling**: UAVs accumulate position error linearly ($0.5$ m/s approx) when outside the range of a support node.
    *   **Probabilistic Coverage**: A grid cell is only considered "covered" if the confidence exceeds 95%. Confidence is a function of the UAV's current drift (higher drift = lower detection probability).
    *   **State Machine**: UAVs autonomously switch states: `SEARCH` $\to$ `RETURN_DRIFT` (if drift high) $\to$ `RETURN_BATTERY` (if energy low) $\to$ `CHARGING`.

### 3. Optimization Pipeline (VNS)
*   **File**: `functions/vns_optimizer.py`
*   **Algorithm**: Variable Neighborhood Search (VNS).
*   **Workflow**:
    1.  **Evaluation**: Every solution is tested by running **5 Monte Carlo simulations** with random start positions. A solution is valid only if $\ge 80\%$ of runs succeed.
    2.  **Neighborhoods**:
        *   **Swap**: Exchanges a selected node with an unselected candidate (Intensification).
        *   **Shift**: Moves a node to a nearby location (Local Search).
        *   **Shake**: Randomly perturbs the solution to escape local optima.
        *   **Add/Remove**: Adjusts the number of support nodes to balance cost vs. performance.

## Key Constraints & Parameters
*   **Area**: Defined square grid (default 1000m x 1000m).
*   **UAVs**: 5 to 13 agents.
*   **Drift Tolerance**: UAVs must reset drift before it exceeds a threshold (e.g., 100m).
*   **Battery**: UAVs calculate 3D energy costs (climb/descent/cruise) and must physically fly to a UGV to swap batteries.
*   **Cost Function**: Total Cost = Infrastructure Cost (UGVs/Beacons) + Operational Cost (UAVs) + Time Cost + Energy Cost.

## Outputs
*   **JSON Results**: `vns_result.json` containing the optimal configuration and mission metrics.
*   **Visualizations**:
    *   `vns_solution_map.png`: Final placement of nodes.
    *   `vns_coverage_heatmap.png`: Confidence map of the area.
    *   `vns_dashboard.png`: Graphs of drift, battery, and coverage over time.
*   **Animation**: Optional GIF generation of the mission execution.
