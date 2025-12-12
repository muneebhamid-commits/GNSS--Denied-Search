# Literature Review: GNSS-Denied UAV Search & Coverage Optimization

This document outlines the theoretical foundations and academic context for the algorithms implemented in this project. The system integrates **Centroidal Voronoi Tessellation (CVT)** for distributed control, **Mixed-Integer Linear Programming (MILP)** for static resource allocation, and **Variable Neighborhood Search (VNS)** for combinatorial optimization.

## 1. Distributed Coverage Control (CVT)
The core UAV movement logic is based on Centroidal Voronoi Tessellations (CVT). In this framework, agents (UAVs) continuously move towards the centroid of their Voronoi cell. This method provides an optimal distribution of sensors over a target area, minimizing the average distance from any point in the domain to the nearest sensor.

*   **Key Reference**:
    > **Cortes, J., Martinez, S., Karatas, T., & Bullo, F. (2004).** "Coverage control for mobile sensing networks." *IEEE Transactions on Robotics and Automation*, 20(2), 243-255.
    *   *Relevance*: This is the seminal paper establishing the gradient descent control law $u = -k(p_i - C_{V_i})$ used in `mission_optimizer.py`, where agents move toward the centroid of their Voronoi partition.

*   **Supporting Reference**:
    > **Du, Q., Faber, V., & Gunzburger, M. (1999).** "Centroidal Voronoi Tessellations: Applications and Algorithms." *SIAM Review*, 41(4), 637-676.
    *   *Relevance*: Provides the mathematical foundation for Lloyd's algorithm, which is the iterative process simulated in the code.

## 2. Optimization Metaheuristics (VNS)
To optimize the placement of support infrastructure (UGVs and Beacons), the project employs Variable Neighborhood Search (VNS). VNS is a metaheuristic that systematically explores different "neighborhoods" (Swap, Shift, Shake, Add/Remove) to escape local optima.

*   **Key Reference**:
    > **Mladenović, N., & Hansen, P. (1997).** "Variable neighborhood search." *Computers & Operations Research*, 24(11), 1097-1100.
    *   *Relevance*: Defines the core structure of the VNS algorithm implemented in `vns_optimizer.py`, specifically the systematic change of neighborhood structures during the search.

*   **Application Reference**:
    > **Hansen, P., & Mladenović, N. (2001).** "Variable neighborhood search: Principles and applications." *European Journal of Operational Research*, 130(3), 449-467.
    *   *Relevance*: Discusses the "Shaking" step used in the pipeline to perturb solutions and explore new areas of the solution space.

## 3. Facility Location & MILP
The initialization phase uses Mixed-Integer Linear Programming (MILP) to solve a variation of the Facility Location Problem. The goal is to select a minimal set of support nodes (UGVs/Beacons) that cover the maximum area, subject to budget or quantity constraints.

*   **Key Reference**:
    > **Daskin, M. S. (1995).** *Network and Discrete Location: Models, Algorithms, and Heuristics*. John Wiley & Sons.
    *   *Relevance*: Provides the standard formulations for the Set Covering Problem (SCP) and Maximal Covering Location Problem (MCLP) used in `milp_placement.py`.

## 4. GNSS-Denied Navigation & Drift
The simulation explicitly models navigation drift, which accumulates over time and requires periodic correction (loop closure) at support nodes. This models the reality of Inertial Navigation Systems (INS) where error grows unbounded without external corrections.

*   **Key Reference**:
    > **Borenstein, J., & Ojeda, L. (2010).** "Heuristic Reduction of Gyro Drift in IMU-based Personnel Tracking Systems." *2010 International Conference on Indoor Positioning and Indoor Navigation*.
    *   *Relevance*: While focused on personnel, the principles of linear drift accumulation and the necessity of "Zero Velocity Updates" (ZUPT) or landmark updates (visiting UGVs in our simulation) are central to the drift logic.

## 5. Probabilistic Search & Detection
The project uses a probabilistic coverage map where detection confidence decays with drift. This aligns with Bayesian search theory, where the probability of detection is conditioned on the sensor's state (position uncertainty).

*   **Key Reference**:
    > **Stone, L. D. (1975).** *Theory of Optimal Search*. Academic Press.
    *   *Relevance*: The classical text on search theory. The update rule $P_{new} = 1 - (1 - P_{old})(1 - P_{det})$ used in the simulation is a direct application of the standard Bayesian update for independent looks.
