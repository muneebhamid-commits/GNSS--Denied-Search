
import numpy as np
import random
import copy
import json
import matplotlib.pyplot as plt
from .mission_optimizer import run_cvt_simulation

def evaluate_solution_cost(constants, ugv_indices, beacon_indices, ugv_candidates, beacon_candidates, n_monte_carlo=None):
    """
    Evaluates a specific configuration of support nodes.
    Iterates through 5-13 UAVs.
    For each UAV count, runs n_monte_carlo simulations.
    Returns the minimum total cost found that satisfies constraints.
    """
    if n_monte_carlo is None:
        n_monte_carlo = constants.get("MONTE_CARLO_RUNS", 5)
    
    # Prepare locations
    ugv_locs = ugv_candidates[list(ugv_indices)] if len(ugv_indices) > 0 else np.empty((0, 2))
    beacon_locs = beacon_candidates[list(beacon_indices)] if len(beacon_indices) > 0 else np.empty((0, 2))
    
    n_ugv = len(ugv_indices)
    n_beacon = len(beacon_indices)
    infrastructure_cost = (n_ugv * constants["UGV_COST"]) + (n_beacon * constants["BEACON_COST"])
    
    best_cost = float('inf')
    best_details = None
    
    # Iterate UAV counts
    for n_uavs in range(5, 14):
        avg_time = 0
        avg_energy = 0
        avg_coverage = 0
        success_count = 0
        
        # Monte Carlo Runs
        for _ in range(n_monte_carlo):
            t, e, success, _ = run_cvt_simulation(n_uavs, constants, ugv_locs, beacon_locs, record_history=False)
            avg_time += t
            avg_energy += e
            if success:
                success_count += 1
        
        avg_time /= n_monte_carlo
        avg_energy /= n_monte_carlo
        
        # 80% Success Rule
        if success_count >= (n_monte_carlo * 0.8):
            # Calculate Cost
            cost_uav_base = n_uavs * constants["UAV_COST"]
            cost_uav_add = n_uavs * constants["UAV_OPERATIONAL_INCREMENT_COST"]
            cost_time = (avg_time / 60.0) * constants["TIME_BASED_OPERATION_COST"]
            energy_kwh = avg_energy / 3.6e6
            cost_energy = energy_kwh * constants["ENERGY_COST"]
            
            total_cost = infrastructure_cost + cost_uav_base + cost_uav_add + cost_time + cost_energy
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_details = {
                    "n_uavs": n_uavs,
                    "cost": total_cost,
                    "time": avg_time,
                    "ugvs": n_ugv,
                    "beacons": n_beacon
                }
                
    return best_cost, best_details

def get_neighbors_indices(candidates, current_indices, radius):
    """Finds unselected candidates within radius of selected ones."""
    # This is expensive to compute every time. 
    # For "Shift", we want to move a selected node to a nearby unselected node.
    pass

def vns_pipeline(constants, milp_result, ugv_candidates, beacon_candidates):
    """
    VNS Pipeline to optimize support node placement.
    """
    print("\n" + "="*60)
    print("STARTING VNS OPTIMIZATION PIPELINE")
    print("="*60)
    
    # Initial Solution from MILP
    current_ugv_indices = set(milp_result["selected_ugvs"])
    current_beacon_indices = set(milp_result["selected_beacons"])
    
    # All available indices
    all_ugv_indices = set(range(len(ugv_candidates)))
    all_beacon_indices = set(range(len(beacon_candidates)))
    
    # Evaluate Initial
    print("Evaluating Initial MILP Solution (5 Monte Carlo runs)...")
    current_cost, current_details = evaluate_solution_cost(
        constants, current_ugv_indices, current_beacon_indices, 
        ugv_candidates, beacon_candidates
    )
    
    if current_cost == float('inf'):
        print("Initial MILP solution did not meet criteria in simulation!")
        # Try to add nodes? Or just proceed?
        # We'll proceed, hoping VNS finds something.
    else:
        print(f"Initial Cost: ${current_cost:,.2f} (UAVs: {current_details['n_uavs']})")

    best_ugv_indices = copy.deepcopy(current_ugv_indices)
    best_beacon_indices = copy.deepcopy(current_beacon_indices)
    best_cost = current_cost
    best_details = current_details
    
    max_iterations = 10 # Total VNS loops
    # Neighborhoods: 1=Swap, 2=Shift, 3=Shake, 4=Add/Remove
    
    for iteration in range(max_iterations):
        print(f"\n--- VNS Iteration {iteration + 1}/{max_iterations} ---")
        improved = False
        
        # --- Neighborhood 1: SWAP (Intensification) ---
        # Keep swapping as long as it is finding better solutions
        print("  > Neighborhood 1: Swap (Intensification)")
        while True:
            swap_improved = False
            k_swaps = 3
            for _ in range(k_swaps):
                # Decide to swap UGV or Beacon
                if random.random() < 0.5 and len(current_ugv_indices) > 0:
                    # Swap UGV
                    u_remove = random.choice(list(current_ugv_indices))
                    available = list(all_ugv_indices - current_ugv_indices)
                    if not available: continue
                    u_add = random.choice(available)
                    
                    new_ugv = copy.deepcopy(current_ugv_indices)
                    new_ugv.remove(u_remove)
                    new_ugv.add(u_add)
                    
                    cost, details = evaluate_solution_cost(constants, new_ugv, current_beacon_indices, ugv_candidates, beacon_candidates)
                    if cost < current_cost:
                        print(f"    Found Better Swap! Cost: ${cost:,.2f}")
                        current_cost = cost
                        current_ugv_indices = new_ugv
                        current_details = details
                        swap_improved = True
                        improved = True
                        break # Break k_swaps loop to restart while loop
                elif len(current_beacon_indices) > 0:
                    # Swap Beacon
                    b_remove = random.choice(list(current_beacon_indices))
                    available = list(all_beacon_indices - current_beacon_indices)
                    if not available: continue
                    b_add = random.choice(available)
                    
                    new_beacon = copy.deepcopy(current_beacon_indices)
                    new_beacon.remove(b_remove)
                    new_beacon.add(b_add)
                    
                    cost, details = evaluate_solution_cost(constants, current_ugv_indices, new_beacon, ugv_candidates, beacon_candidates)
                    if cost < current_cost:
                        print(f"    Found Better Swap! Cost: ${cost:,.2f}")
                        current_cost = cost
                        current_beacon_indices = new_beacon
                        current_details = details
                        swap_improved = True
                        improved = True
                        break # Break k_swaps loop to restart while loop
            
            if not swap_improved:
                break # Stop swapping if no improvement in this batch
        
        # Update best if improved during swap
        if improved:
            if current_cost < best_cost:
                best_cost = current_cost
                best_ugv_indices = copy.deepcopy(current_ugv_indices)
                best_beacon_indices = copy.deepcopy(current_beacon_indices)
                best_details = current_details
            # Do NOT continue here. Fall through to Shift/Shake if Swap is exhausted.
            
        # --- Neighborhood 2: SHIFT ---
        # Move a selected node to a nearby unselected node (Local Search)
        print("  > Neighborhood 2: Shift")
        # Try k shifts
        k_shifts = 3
        shift_radius = 500 # meters
        
        for _ in range(k_shifts):
             if random.random() < 0.5 and len(current_ugv_indices) > 0:
                # Shift UGV
                u_idx = random.choice(list(current_ugv_indices))
                u_pos = ugv_candidates[u_idx]
                
                # Find neighbors
                available = list(all_ugv_indices - current_ugv_indices)
                if not available: continue
                
                # Simple distance check (can be optimized with KDTree but list is small enough)
                avail_locs = ugv_candidates[available]
                dists = np.linalg.norm(avail_locs - u_pos, axis=1)
                nearby_mask = dists < shift_radius
                
                if np.any(nearby_mask):
                    # Pick one nearby
                    nearby_indices = np.array(available)[nearby_mask]
                    u_add = random.choice(nearby_indices)
                    
                    new_ugv = copy.deepcopy(current_ugv_indices)
                    new_ugv.remove(u_idx)
                    new_ugv.add(u_add)
                    
                    cost, details = evaluate_solution_cost(constants, new_ugv, current_beacon_indices, ugv_candidates, beacon_candidates)
                    if cost < current_cost:
                        print(f"    Found Better Shift! Cost: ${cost:,.2f}")
                        current_cost = cost
                        current_ugv_indices = new_ugv
                        current_details = details
                        improved = True
                        break
             elif len(current_beacon_indices) > 0:
                # Shift Beacon
                b_idx = random.choice(list(current_beacon_indices))
                b_pos = beacon_candidates[b_idx]
                
                available = list(all_beacon_indices - current_beacon_indices)
                if not available: continue
                
                avail_locs = beacon_candidates[available]
                dists = np.linalg.norm(avail_locs - b_pos, axis=1)
                nearby_mask = dists < shift_radius
                
                if np.any(nearby_mask):
                    nearby_indices = np.array(available)[nearby_mask]
                    b_add = random.choice(nearby_indices)
                    
                    new_beacon = copy.deepcopy(current_beacon_indices)
                    new_beacon.remove(b_idx)
                    new_beacon.add(b_add)
                    
                    cost, details = evaluate_solution_cost(constants, current_ugv_indices, new_beacon, ugv_candidates, beacon_candidates)
                    if cost < current_cost:
                        print(f"    Found Better Shift! Cost: ${cost:,.2f}")
                        current_cost = cost
                        current_beacon_indices = new_beacon
                        current_details = details
                        improved = True
                        break

        if improved:
            if current_cost < best_cost:
                best_cost = current_cost
                best_ugv_indices = copy.deepcopy(current_ugv_indices)
                best_beacon_indices = copy.deepcopy(current_beacon_indices)
                best_details = current_details
            continue

        # --- Neighborhood 3: SHAKE ---
        # Perturb a cluster. We'll implement this as multiple random swaps.
        print("  > Neighborhood 3: Shake")
        # Perform shake but only accept if better (VNS usually shakes to escape local optima, 
        # but here we are wrapping it as a pipeline step. 
        # Standard VNS: Shake -> Local Search -> Accept if better.
        # We will just try a random perturbation and see if it's better directly for simplicity,
        # or we can accept it even if worse? The prompt says "then shakes... and as last step adds/removes".
        # This implies a sequential pipeline of heuristics.
        # Let's try to find a better solution by shaking.
        
        shake_strength = 2
        new_ugv = copy.deepcopy(current_ugv_indices)
        new_beacon = copy.deepcopy(current_beacon_indices)
        
        # Randomly swap 'shake_strength' nodes
        for _ in range(shake_strength):
            if random.random() < 0.5 and len(new_ugv) > 0:
                u_rem = random.choice(list(new_ugv))
                new_ugv.remove(u_rem)
                avail = list(all_ugv_indices - new_ugv)
                if avail: new_ugv.add(random.choice(avail))
            elif len(new_beacon) > 0:
                b_rem = random.choice(list(new_beacon))
                new_beacon.remove(b_rem)
                avail = list(all_beacon_indices - new_beacon)
                if avail: new_beacon.add(random.choice(avail))
                
        cost, details = evaluate_solution_cost(constants, new_ugv, new_beacon, ugv_candidates, beacon_candidates)
        if cost < current_cost:
            print(f"    Found Better Shake! Cost: ${cost:,.2f}")
            current_cost = cost
            current_ugv_indices = new_ugv
            current_beacon_indices = new_beacon
            current_details = details
            improved = True
            
        if improved:
            if current_cost < best_cost:
                best_cost = current_cost
                best_ugv_indices = copy.deepcopy(current_ugv_indices)
                best_beacon_indices = copy.deepcopy(current_beacon_indices)
                best_details = current_details
            continue

        # --- Neighborhood 4: ADD / REMOVE ---
        print("  > Neighborhood 4: Add/Remove")
        # Try adding or removing a node
        # Prioritize removing if we have many, adding if we have few? Random for now.
        
        action = random.choice(['add', 'remove'])
        node_type = random.choice(['ugv', 'beacon'])
        
        new_ugv = copy.deepcopy(current_ugv_indices)
        new_beacon = copy.deepcopy(current_beacon_indices)
        
        valid_op = False
        if action == 'add':
            if node_type == 'ugv':
                avail = list(all_ugv_indices - new_ugv)
                if avail:
                    new_ugv.add(random.choice(avail))
                    valid_op = True
            else:
                avail = list(all_beacon_indices - new_beacon)
                if avail:
                    new_beacon.add(random.choice(avail))
                    valid_op = True
        else: # remove
            if node_type == 'ugv' and len(new_ugv) > 1: # Keep at least 1?
                new_ugv.remove(random.choice(list(new_ugv)))
                valid_op = True
            elif node_type == 'beacon' and len(new_beacon) > 0:
                new_beacon.remove(random.choice(list(new_beacon)))
                valid_op = True
                
        if valid_op:
            cost, details = evaluate_solution_cost(constants, new_ugv, new_beacon, ugv_candidates, beacon_candidates)
            if cost < current_cost:
                print(f"    Found Better Add/Remove! Cost: ${cost:,.2f}")
                current_cost = cost
                current_ugv_indices = new_ugv
                current_beacon_indices = new_beacon
                current_details = details
                improved = True
        
        if improved:
            if current_cost < best_cost:
                best_cost = current_cost
                best_ugv_indices = copy.deepcopy(current_ugv_indices)
                best_beacon_indices = copy.deepcopy(current_beacon_indices)
                best_details = current_details
            continue
            
        print("  No improvement in this iteration.")
        
    print("="*60)
    print("VNS OPTIMIZATION COMPLETE")
    print(f"Best Cost: ${best_cost:,.2f}")
    if best_details:
        print(f"Configuration: {best_details['ugvs']} UGVs, {best_details['beacons']} Beacons, {best_details['n_uavs']} UAVs")
        
        # --- Final Visualization Run ---
        print("\nRunning Final Simulation for Visualization...")
        ugv_locs = ugv_candidates[list(best_ugv_indices)] if len(best_ugv_indices) > 0 else np.empty((0, 2))
        beacon_locs = beacon_candidates[list(best_beacon_indices)] if len(best_beacon_indices) > 0 else np.empty((0, 2))
        
        # Pick a seed for reproducibility
        showcase_seed = random.randint(0, 10000)
        
        t, e, success, history = run_cvt_simulation(
            best_details['n_uavs'], constants, ugv_locs, beacon_locs, record_history=True, seed=showcase_seed
        )
        
        # Save VNS Result
        result = {
            "selected_ugvs": [int(x) for x in best_ugv_indices],
            "selected_beacons": [int(x) for x in best_beacon_indices],
            "num_uavs": int(best_details['n_uavs']),
            "cost": float(best_cost),
            "time": float(t),
            "energy": float(e),
            "success": bool(success),
            "seed": int(showcase_seed)
        }
        with open("vns_result.json", "w") as f:
            json.dump(result, f, indent=4)
            
        # --- PLOTTING ---
        area_side = constants["AREA"]
        
        # Figure 1: Final Solution Map
        fig1, ax1 = plt.subplots(figsize=(10, 10))
        ax1.set_title(f"Final VNS Solution (Cost: ${best_cost:,.0f})")
        ax1.set_xlim(0, area_side)
        ax1.set_ylim(0, area_side)
        
        # Candidates (faint)
        ax1.scatter(ugv_candidates[:, 0], ugv_candidates[:, 1], c='lightblue', s=10, alpha=0.3)
        ax1.scatter(beacon_candidates[:, 0], beacon_candidates[:, 1], c='lightgreen', s=10, alpha=0.3)
        
        # Selected
        if len(ugv_locs) > 0:
            ax1.scatter(ugv_locs[:, 0], ugv_locs[:, 1], c='blue', s=100, label='UGV', edgecolors='k')
            for u in ugv_locs:
                ax1.add_patch(plt.Circle((u[0], u[1]), constants["COVERAGE_RADIUS_UGV"], color='blue', fill=False, linestyle='--', alpha=0.3))
        
        if len(beacon_locs) > 0:
            ax1.scatter(beacon_locs[:, 0], beacon_locs[:, 1], c='green', s=100, marker='^', label='Beacon', edgecolors='k')
            for b in beacon_locs:
                ax1.add_patch(plt.Circle((b[0], b[1]), constants["COVERAGE_RADIUS_BEACON"], color='green', fill=False, linestyle='--', alpha=0.3))
                
        ax1.legend()
        plt.savefig("vns_solution_map.png")
        
        # Figure 2: Coverage Heatmap
        if history and "grid_confidence" in history:
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            ax2.set_title("Coverage Confidence Heatmap")
            
            # Reshape grid
            grid_res = constants.get("GRID_SPACING", 100.0)
            nx = int(area_side / grid_res)
            ny = int(area_side / grid_res)
            
            conf_map = history["grid_confidence"].reshape((ny, nx))
            
            im = ax2.imshow(conf_map, origin='lower', extent=[0, area_side, 0, area_side], cmap='hot', vmin=0, vmax=1)
            plt.colorbar(im, ax=ax2, label='Confidence')
            plt.savefig("vns_coverage_heatmap.png")
            
        # Figure 3: Dashboard
        if history:
            fig3, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
            time_axis = history["time"]
            
            # Drift
            axs[0].plot(time_axis, history["avg_drift"], label="Avg Drift")
            axs[0].plot(time_axis, history["max_drift"], label="Max Drift", linestyle='--')
            axs[0].set_ylabel("Drift (m)")
            axs[0].set_title("Drift Statistics")
            axs[0].legend()
            axs[0].grid(True)
            
            # Battery
            # battery_levels is list of arrays (steps, n_uavs)
            batt_data = np.array(history["battery_levels"])
            for i in range(batt_data.shape[1]):
                axs[1].plot(time_axis, batt_data[:, i], alpha=0.5)
            axs[1].set_ylabel("Battery (J)")
            axs[1].set_title("UAV Battery Levels")
            axs[1].grid(True)
            
            # Coverage
            axs[2].plot(time_axis, np.array(history["coverage_pct"]) * 100, color='green')
            axs[2].set_ylabel("Coverage (%)")
            axs[2].set_xlabel("Time (min)")
            axs[2].set_title("Area Coverage Progress")
            axs[2].axhline(y=97, color='r', linestyle='--', label='Threshold (97%)')
            axs[2].legend()
            axs[2].grid(True)
            
            plt.tight_layout()
            plt.savefig("vns_dashboard.png")
            
        plt.show()
        
        # --- GIF Prompt ---
        print("\n" + "="*60)
        save_gif = input("Do you want to save a GIF animation of the mission? (y/n): ").strip().lower()
        if save_gif == 'y':
            print("Generating GIF with the same random seed...")
            run_cvt_simulation(
                best_details['n_uavs'], constants, ugv_locs, beacon_locs, 
                save_gif=True, gif_filename="mission_animation.gif", seed=showcase_seed
            )
        print("="*60)

    print("="*60)
    
    return {
        "selected_ugvs": list(best_ugv_indices),
        "selected_beacons": list(best_beacon_indices),
        "best_details": best_details
    }
