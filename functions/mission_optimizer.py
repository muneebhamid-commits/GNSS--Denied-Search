
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import distance
import pulp
from .milp_placement import solve_milp
from .candidate_generation_vectorized import generate_and_filter_candidates_vectorized

def run_cvt_simulation(num_uavs, constants, ugv_locs, beacon_locs, max_time_min=100, coverage_threshold=0.97, record_history=False, save_gif=False, gif_filename="simulation.gif", seed=None):
    """
    Simulates UAV mission using CVT for area allocation.
    Includes Drift and Battery constraints.
    Returns: time_taken (min), energy_consumed (J), success (bool), history (dict, optional)
    """
    if seed is not None:
        np.random.seed(seed)
        
    area_side = constants["AREA"]
    sensor_radius = constants["SENSOR_RADIUS"]
    cruise_speed = constants["UAV_CRUISE_SPEED"]
    avg_power = constants["AVERAGE_FLIGHT_POWER"]
    
    # Drift parameters
    drift_rate_min = constants["DRIFT_RATE"] # m/min
    drift_rate_sec = drift_rate_min / 60.0 # m/s
    drift_tolerance = constants["DRIFT_TOLERANCE"]
    
    # Battery parameters
    capacity = constants["BATTERY_CAPACITY"]
    reserve_fraction = constants["BATTERY_RESERVE_MISSION"]
    min_safe_energy = capacity * reserve_fraction
    
    # 3D parameters
    altitude = constants["MISSION_ALTITUDE"]
    descent_speed = constants["DESCENT_SPEED"]
    climb_speed = constants["CLIMB_SPEED"]
    
    # Coverage Radii
    r_ugv = constants["COVERAGE_RADIUS_UGV"]
    r_beacon = constants["COVERAGE_RADIUS_BEACON"]
    
    # Simulation parameters
    dt = 1.0 # seconds per step
    grid_res = constants.get("GRID_SPACING", 100.0) # meters
    
    # Create grid for coverage and CVT
    x = np.arange(0, area_side, grid_res)
    y = np.arange(0, area_side, grid_res)
    gx, gy = np.meshgrid(x, y)
    grid_points = np.column_stack((gx.ravel(), gy.ravel()))
    num_grid_points = len(grid_points)
    
    # Initialize UAVs
    uav_pos = np.random.rand(num_uavs, 2) * area_side
    uav_energy = np.random.uniform(min_safe_energy, capacity, num_uavs)
    uav_drift = np.zeros(num_uavs)
    
    # States: 0=SEARCH, 1=RETURN_DRIFT, 2=RETURN_BATTERY, 3=CHARGING
    uav_state = np.zeros(num_uavs, dtype=int)
    uav_target = np.zeros((num_uavs, 2))
    
    # Coverage tracking
    grid_confidence = np.zeros(num_grid_points, dtype=float)
    covered_mask = np.zeros(num_grid_points, dtype=bool)
    
    current_time = 0.0
    total_energy_consumed = 0.0
    
    # History recording
    history = {
        "time": [],
        "avg_drift": [],
        "max_drift": [],
        "battery_levels": [],
        "coverage_pct": [],
        "grid_confidence": [] # Store confidence maps for animation/heatmap
    } if record_history else None
    
    # Animation frames
    frames = [] if save_gif else None
    
    while current_time < (max_time_min * 60):
        # Record history
        if record_history and (int(current_time) % 10 == 0): # Record every 10s to save memory
            history["time"].append(current_time / 60.0)
            history["avg_drift"].append(np.mean(uav_drift))
            history["max_drift"].append(np.max(uav_drift))
            history["battery_levels"].append(uav_energy.copy())
            history["coverage_pct"].append(np.sum(covered_mask) / num_grid_points)
            
        # Record frame for GIF (every 60s simulation time to keep file size manageable)
        if save_gif and (int(current_time) % 60 == 0):
            frames.append({
                "uav_pos": uav_pos.copy(),
                "uav_state": uav_state.copy(),
                "grid_confidence": grid_confidence.copy(), # Save confidence for coloring
                "time": current_time
            })

        # --- 1. Update Drift & Check Coverage ---
        # Calculate distances to all support nodes
        if len(ugv_locs) > 0:
            d_ugv = distance.cdist(uav_pos, ugv_locs)
            in_ugv_cov = np.any(d_ugv <= r_ugv, axis=1)
        else:
            d_ugv = np.full((num_uavs, 0), np.inf)
            in_ugv_cov = np.zeros(num_uavs, dtype=bool)
            
        if len(beacon_locs) > 0:
            d_beacon = distance.cdist(uav_pos, beacon_locs)
            in_beacon_cov = np.any(d_beacon <= r_beacon, axis=1)
        else:
            d_beacon = np.full((num_uavs, 0), np.inf)
            in_beacon_cov = np.zeros(num_uavs, dtype=bool)
            
        in_coverage = in_ugv_cov | in_beacon_cov
        
        # Update drift
        uav_drift[in_coverage] = 0.0
        uav_drift[~in_coverage] += drift_rate_sec * dt
        
        # --- 2. Determine States & Targets ---
        
        # CVT Calculation (for SEARCH state)
        # Only consider points that are NOT yet fully covered (confidence < 0.95)
        # This focuses the search on uncovered areas
        uncovered_indices = np.where(~covered_mask)[0]
        if len(uncovered_indices) > 0:
            uncovered_points = grid_points[uncovered_indices]
            dists_grid = distance.cdist(uncovered_points, uav_pos)
            nearest_uav_idx = np.argmin(dists_grid, axis=1)
            
            cvt_targets = np.zeros_like(uav_pos)
            for i in range(num_uavs):
                my_points = uncovered_points[nearest_uav_idx == i]
                if len(my_points) > 0:
                    cvt_targets[i] = np.mean(my_points, axis=0)
                else:
                    # Random walk if crowded or no points assigned
                    angle = np.random.rand() * 2 * np.pi
                    cvt_targets[i] = uav_pos[i] + np.array([np.cos(angle), np.sin(angle)]) * cruise_speed * 10
        else:
             # All covered, just hover or random walk (simulation should end soon)
             cvt_targets = uav_pos.copy()
        
        for i in range(num_uavs):
            # Skip if charging (handled in movement)
            if uav_state[i] == 3:
                continue
                
            # --- Safety Checks ---
            
            # A. Battery Check
            # Find nearest UGV
            if len(ugv_locs) > 0:
                d_to_ugvs = d_ugv[i]
                nearest_ugv_idx = np.argmin(d_to_ugvs)
                dist_ugv_2d = d_to_ugvs[nearest_ugv_idx]
                nearest_ugv_pos = ugv_locs[nearest_ugv_idx]
            else:
                # Should not happen if we have UGVs
                dist_ugv_2d = float('inf')
                nearest_ugv_pos = uav_pos[i]

            # Energy to reach UGV and land
            # Time = 2D_Time + Descent_Time
            time_to_ugv = (dist_ugv_2d / cruise_speed) + (altitude / descent_speed)
            energy_req = time_to_ugv * avg_power
            
            # If we are already returning for battery, keep doing it
            if uav_state[i] == 2:
                uav_target[i] = nearest_ugv_pos
            # If not, check if we need to start
            elif (uav_energy[i] - energy_req) < min_safe_energy:
                uav_state[i] = 2 # RETURN_BATTERY
                uav_target[i] = nearest_ugv_pos
            
            # B. Drift Check (Only if not Critical Battery)
            elif uav_state[i] != 2:
                # Find nearest coverage boundary
                min_dist_cov = float('inf')
                target_cov = uav_pos[i]
                
                # Check UGVs
                if len(ugv_locs) > 0:
                    closest_ugv_idx = np.argmin(d_ugv[i])
                    d = d_ugv[i][closest_ugv_idx]
                    if d > r_ugv:
                        dist_to_boundary = d - r_ugv
                        vec = ugv_locs[closest_ugv_idx] - uav_pos[i]
                        target_cov = uav_pos[i] + (vec / d) * dist_to_boundary
                    else:
                        dist_to_boundary = 0
                        target_cov = uav_pos[i]
                    
                    if dist_to_boundary < min_dist_cov:
                        min_dist_cov = dist_to_boundary
                        
                # Check Beacons
                if len(beacon_locs) > 0:
                    closest_beacon_idx = np.argmin(d_beacon[i])
                    d = d_beacon[i][closest_beacon_idx]
                    if d > r_beacon:
                        dist_to_boundary = d - r_beacon
                        vec = beacon_locs[closest_beacon_idx] - uav_pos[i]
                        target_cov_b = uav_pos[i] + (vec / d) * dist_to_boundary
                    else:
                        dist_to_boundary = 0
                        target_cov_b = uav_pos[i]
                        
                    if dist_to_boundary < min_dist_cov:
                        min_dist_cov = dist_to_boundary
                        target_cov = target_cov_b

                # Predict drift at arrival
                time_to_cov = min_dist_cov / cruise_speed
                predicted_drift = uav_drift[i] + (time_to_cov * drift_rate_sec)
                
                if uav_state[i] == 1:
                    # Already returning, update target to nearest coverage
                    uav_target[i] = target_cov
                    # If we are inside coverage now, switch back to search
                    if in_coverage[i]:
                        uav_state[i] = 0
                        uav_target[i] = cvt_targets[i]
                elif predicted_drift >= drift_tolerance:
                    uav_state[i] = 1 # RETURN_DRIFT
                    uav_target[i] = target_cov
                else:
                    uav_state[i] = 0 # SEARCH
                    uav_target[i] = cvt_targets[i]

        # --- 3. Move UAVs ---
        for i in range(num_uavs):
            if uav_state[i] == 3: # CHARGING
                # Energy cost of climb
                time_climb = altitude / climb_speed
                energy_climb = time_climb * avg_power
                
                uav_energy[i] = capacity - energy_climb # Start with full minus climb cost
                total_energy_consumed += energy_climb
                
                # Reset state
                uav_state[i] = 0
                uav_drift[i] = 0
                continue

            # Move towards target
            curr = uav_pos[i]
            target = uav_target[i]
            dist = np.linalg.norm(target - curr)
            
            step_dist = cruise_speed * dt
            
            if dist <= step_dist:
                uav_pos[i] = target
                # If target was UGV for battery
                if uav_state[i] == 2:
                    # Arrived at UGV (2D)
                    # Apply Descent Cost
                    time_desc = altitude / descent_speed
                    energy_desc = time_desc * avg_power
                    uav_energy[i] -= energy_desc
                    total_energy_consumed += energy_desc
                    
                    # Switch to Charging (will reset next step)
                    uav_state[i] = 3 
            else:
                dir_vec = (target - curr) / dist
                uav_pos[i] = curr + dir_vec * step_dist
                
            # Boundary check
            uav_pos[i] = np.clip(uav_pos[i], 0, area_side)

        # --- 4. Update Coverage (Probabilistic) ---
        # Calculate distances from all grid points to all UAVs
        dists_new = distance.cdist(grid_points, uav_pos)
        
        for i in range(num_uavs):
            # Only active UAVs contribute to coverage
            if uav_state[i] == 3: # Charging
                continue
                
            # Find points within sensor radius
            in_range_indices = np.where(dists_new[:, i] <= sensor_radius)[0]
            
            if len(in_range_indices) > 0:
                # Calculate detection probability based on drift
                # drift_pct = current_drift / tolerance
                # prob = 1.0 * (1 - drift_pct)
                drift_pct = uav_drift[i] / drift_tolerance
                detection_prob = max(0.0, 1.0 - drift_pct)
                
                # Update confidence: P_new = 1 - (1 - P_old) * (1 - P_det)
                # Bayesian Update
                current_conf = grid_confidence[in_range_indices]
                new_conf = 1.0 - (1.0 - current_conf) * (1.0 - detection_prob)
                grid_confidence[in_range_indices] = new_conf
                
        # Update covered mask
        covered_mask = grid_confidence >= 0.95
        grid_confidence[in_range_indices] = new_conf
        
        # Update covered mask based on 95% threshold
        covered_mask = grid_confidence >= 0.95
        
        coverage_pct = np.sum(covered_mask) / num_grid_points
        
        # --- 5. Update Energy & Time ---
        active_mask = (uav_state != 3)
        step_energy = np.sum(active_mask) * avg_power * dt
        
        uav_energy[active_mask] -= (avg_power * dt)
        total_energy_consumed += step_energy
        
        current_time += dt
        
        if coverage_pct >= coverage_threshold:
            if record_history:
                history["grid_confidence"] = grid_confidence # Save final confidence map
            if save_gif:
                _save_animation(frames, constants, ugv_locs, beacon_locs, grid_points, gif_filename)
            return current_time / 60.0, total_energy_consumed, True, history
            
    if record_history:
        history["grid_confidence"] = grid_confidence
    if save_gif:
        _save_animation(frames, constants, ugv_locs, beacon_locs, grid_points, gif_filename)
    return current_time / 60.0, total_energy_consumed, False, history

def _save_animation(frames, constants, ugv_locs, beacon_locs, grid_points, filename):
    print(f"Saving animation to {filename}...")
    fig, ax = plt.subplots(figsize=(10, 10))
    area_side = constants["AREA"]
    ax.set_xlim(0, area_side)
    ax.set_ylim(0, area_side)
    
    # Static elements
    if len(ugv_locs) > 0:
        ax.scatter(ugv_locs[:, 0], ugv_locs[:, 1], c='blue', s=100, label='UGV', edgecolors='k')
        for u in ugv_locs:
            ax.add_patch(plt.Circle((u[0], u[1]), constants["COVERAGE_RADIUS_UGV"], color='blue', fill=False, linestyle='--', alpha=0.3))
    
    if len(beacon_locs) > 0:
        ax.scatter(beacon_locs[:, 0], beacon_locs[:, 1], c='green', s=100, marker='^', label='Beacon', edgecolors='k')
        for b in beacon_locs:
            ax.add_patch(plt.Circle((b[0], b[1]), constants["COVERAGE_RADIUS_BEACON"], color='green', fill=False, linestyle='--', alpha=0.3))
            
    # Dynamic elements
    # Use scatter for coverage to "color the regions"
    # We will update the color/alpha of grid points based on confidence
    scat_cov = ax.scatter([], [], c=[], cmap='Greens', vmin=0, vmax=1, s=15, marker='s', alpha=0.5)
    scat_uav = ax.scatter([], [], c='red', s=50, label='UAV', edgecolors='white')
    
    title = ax.set_title("Time: 0.0 min")
    
    def update(frame):
        # Update UAVs
        scat_uav.set_offsets(frame["uav_pos"])
        
        # Update Coverage
        # frame["grid_confidence"] contains values 0..1
        # We only want to plot points with some confidence to avoid clutter?
        # Or plot all with color mapping.
        conf = frame["grid_confidence"]
        mask = conf > 0.01
        if np.any(mask):
            pts = grid_points[mask]
            c_vals = conf[mask]
            scat_cov.set_offsets(pts)
            scat_cov.set_array(c_vals)
        
        title.set_text(f"Time: {frame['time']/60.0:.1f} min")
        return scat_uav, scat_cov, title
        
    anim = FuncAnimation(fig, update, frames=frames, interval=200, blit=False)
    anim.save(filename, writer='pillow', fps=5)
    plt.close(fig)
    print("Animation saved.")

def optimize_mission_parameters(constants, milp_result, ugv_candidates, beacon_candidates):
    """
    Iterates through 5-13 UAVs to find the cheapest solution meeting requirements.
    """
    # Extract selected nodes from MILP result
    selected_ugv_indices = milp_result["selected_ugvs"]
    selected_beacon_indices = milp_result["selected_beacons"]
    
    ugv_locs = ugv_candidates[selected_ugv_indices] if len(selected_ugv_indices) > 0 else np.empty((0, 2))
    beacon_locs = beacon_candidates[selected_beacon_indices] if len(selected_beacon_indices) > 0 else np.empty((0, 2))
    
    n_ugv = len(selected_ugv_indices)
    n_beacon = len(selected_beacon_indices)
    infrastructure_cost = (n_ugv * constants["UGV_COST"]) + (n_beacon * constants["BEACON_COST"])
    
    print(f"Infrastructure Cost: ${infrastructure_cost:,.2f} ({n_ugv} UGVs, {n_beacon} Beacons)")
    
    print(f"{'# UAVs':<8} | {'Time (min)':<10} | {'Cov %':<8} | {'Cost ($)':<15} | {'Status'}")
    print("-" * 65)
    
    best_solution = None
    min_cost = float('inf')
    
    for n_uavs in range(5, 14): # 5 to 13
        # Run Simulation
        time_taken, energy_joules, success = run_cvt_simulation(n_uavs, constants, ugv_locs, beacon_locs)
        
        # Calculate Costs
        cost_milp = infrastructure_cost
        cost_uav_base = n_uavs * constants["UAV_COST"]
        cost_uav_add = n_uavs * constants["UAV_OPERATIONAL_INCREMENT_COST"]
        cost_time = (time_taken / 60.0) * constants["TIME_BASED_OPERATION_COST"]
        energy_kwh = energy_joules / 3.6e6
        cost_energy = energy_kwh * constants["ENERGY_COST"]
        
        total_cost = cost_milp + cost_uav_base + cost_uav_add + cost_time + cost_energy
        
        status_str = "Success" if success else "Failed (>100m)"
        
        print(f"{n_uavs:<8} | {time_taken:<10.1f} | {'>97%' if success else '<97%':<8} | ${total_cost:,.2f} | {status_str}")
        
        if success:
            if total_cost < min_cost:
                min_cost = total_cost
                best_solution = {
                    "num_uavs": n_uavs,
                    "total_cost": total_cost,
                    "time": time_taken,
                    "energy_kwh": energy_kwh,
                    "infrastructure_cost": infrastructure_cost,
                    "breakdown": {
                        "milp": cost_milp,
                        "uav_base": cost_uav_base,
                        "uav_add": cost_uav_add,
                        "time": cost_time,
                        "energy": cost_energy
                    }
                }
                
    print("-" * 65)
    if best_solution:
        print(f"\n🏆 Best Solution Found:")
        print(f"   UAVs: {best_solution['num_uavs']}")
        print(f"   Total Cost: ${best_solution['total_cost']:,.2f}")
        print(f"   Time: {best_solution['time']:.1f} min")
        print(f"   Energy: {best_solution['energy_kwh']:.2f} kWh")
    else:
        print("\n❌ No solution met the criteria (97% coverage within 100 min).")
        
    return best_solution
