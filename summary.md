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

## Mathematical Formulation

### 1. Mixed-Integer Linear Programming (MILP)
The initialization phase solves a Facility Location Problem to minimize infrastructure cost while ensuring coverage.

**Objective Function:**
$$ \min \sum_{i \in I} C_{UGV} x_i + \sum_{j \in J} C_{Beacon} y_j $$
Where:
*   $x_i \in \{0, 1\}$: Binary variable for UGV at candidate location $i$.
*   $y_j \in \{0, 1\}$: Binary variable for Beacon at candidate location $j$.
*   $C_{UGV}, C_{Beacon}$: Costs of respective units.

**Constraints:**
1.  **Coverage Constraint**: Every grid point $k$ must be covered by at least one selected node.
    $$ \sum_{i \in I} a_{ik} x_i + \sum_{j \in J} b_{jk} y_j \ge 1 \quad \forall k \in K $$
    Where $a_{ik} = 1$ if point $k$ is within coverage radius of UGV $i$, else 0.
2.  **Minimum UAVs**:
    $$ \sum_{u \in U} z_u \ge N_{min} $$
3.  **Battery Feasibility**:
    $$ R_{swap} \le \frac{(E_{cap} \times (1 - R_{reserve})) \times v_{min}}{P_{avg}} $$

### 2. Bayesian Coverage Update
The simulation uses a probabilistic grid map. The confidence $P(C_k)$ of cell $k$ being covered is updated using a Bayesian rule based on the UAV's detection probability $P_{det}$.

**Detection Probability:**
$$ P_{det}(t) = \max \left(0, 1 - \frac{\delta(t)}{\delta_{tol}} \right) $$
Where $\delta(t)$ is the current drift and $\delta_{tol}$ is the drift tolerance.

**Update Rule:**
$$ P(C_k | z_t) = 1 - (1 - P(C_k | z_{t-1})) \times (1 - P_{det}(t)) $$
A cell is considered "covered" when $P(C_k) \ge 0.95$.

### 3. Drift Dynamics
UAV position error (drift) accumulates linearly over time when outside support coverage:
$$ \delta(t + \Delta t) = \delta(t) + r_{drift} \times \Delta t $$
When a UAV enters the coverage radius of a UGV or Beacon, drift is reset:
$$ \delta(t) \to 0 $$

### 4. CVT Energy Function (Movement Logic)
The UAVs move according to Lloyd's Algorithm to minimize the Centroidal Voronoi Tessellation (CVT) energy function. This ensures optimal distribution of agents across the uncovered area.

**Energy Function:**
$$ J = \sum_{i=1}^{N} \int_{V_i} \rho(q) \| q - p_i \|^2 dq $$
Where:
*   $V_i$: Voronoi cell of UAV $i$.
*   $p_i$: Position of UAV $i$.
*   $\rho(q)$: Density function (higher weight for uncovered areas).
*   $\| q - p_i \|^2$: Squared Euclidean distance.

The control input moves each UAV toward the centroid $C_i$ of its cell:
$$ p_i[k+1] = p_i[k] + k_p (C_i - p_i[k]) $$

### 5. Battery Consumption Model
Energy consumption is modeled based on flight state, derived from rotorcraft power equations.

**Power Consumption:**
$$ P(t) = \begin{cases} P_{hover} & \text{if } v \approx 0 \\ P_{cruise} & \text{if } v > 0 \\ P_{climb} & \text{if } v_z > 0 \\ P_{descent} & \text{if } v_z < 0 \end{cases} $$

**Total Energy:**
$$ E_{total} = \int_{0}^{T} P(t) dt $$
UAVs must return to a UGV for battery swapping when $E_{remaining} \le E_{threshold}$.

### 6. VNS Objective Function
The Variable Neighborhood Search minimizes a weighted total cost function to balance infrastructure investment against mission performance.

**Total Cost:**
$$ J_{VNS} = w_1 C_{infra} + w_2 C_{ops} + w_3 T_{mission} + w_4 E_{fleet} + P_{penalty} $$
Where:
*   $C_{infra}$: Cost of UGVs + Beacons.
*   $C_{ops}$: Operational cost per UAV.
*   $T_{mission}$: Time to reach 97% coverage.
*   $E_{fleet}$: Total energy consumed by all UAVs.
*   $P_{penalty}$: Large penalty if mission fails (coverage < 97% or time > limit).

## Outputs
*   **JSON Results**: `vns_result.json` containing the optimal configuration and mission metrics.
*   **Visualizations**:
    *   `vns_solution_map.png`: Final placement of nodes (3 subplots: Drift, Coverage, Battery).
    *   `vns_coverage_heatmap.png`: Confidence map of the area.
    *   `vns_dashboard.png`: Graphs of drift, battery, and coverage over time.
    *   `candidates_generated.png`: Initial candidate grid.
    *   `milp_solution.png`: Initial static solution.
*   **Animation**: Optional GIF generation of the mission execution.
## Drift Plotting in VNS Dashboard

In the current code, each UAV tracks its own drift value, which accumulates as it moves. The "maximum drift" shown in the VNS dashboard refers to the highest drift value among all UAVs at a given time step (i.e., the worst-case drift for any single UAV). Drift can exceed the threshold (e.g., 100 meters) if a UAV does not visit a support node in time, because the code does not forcibly cap the drift variable itself. The threshold is used to trigger correction, but drift continues to accumulate until reset.

## Confidence Update Formula

When a UAV visits a grid cell, the confidence of that cell is increased by:

    gain = max(0, 100 - UAV drift)

This means the higher the UAV's drift, the less confidence is gained per visit. When the cell's confidence reaches 95% or more, it is considered covered.