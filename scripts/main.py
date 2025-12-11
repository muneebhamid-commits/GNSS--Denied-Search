# Test for git
# main.py
# Entry point for simulation

from functions import get_simulation_constants, optimize_mission_parameters, vns_pipeline
from functions.candidate_generation_vectorized import generate_and_filter_candidates_vectorized
from functions.milp_placement import solve_milp
import pulp

def main():
    # Get simulation constants with default density
    constants = get_simulation_constants()
    
    # Generate candidate positions
    ugv_candidates, beacon_candidates = generate_and_filter_candidates_vectorized(constants, plot_results=True)
    
    # Initialize UAV variables for MILP
    uav_vars = [pulp.LpVariable(f"UAV_{i}", cat="Binary") for i in range(constants["NUM_UAVS_MAX"])]
    
    # Solve MiLP for support node selection and visualize
    milp_result = solve_milp(constants, ugv_candidates, beacon_candidates, uav_vars)
    
    # Run VNS Pipeline
    # This wraps the mission optimization logic in a VNS loop
    vns_pipeline(constants, milp_result, ugv_candidates, beacon_candidates)




if __name__ == "__main__":
    main()
