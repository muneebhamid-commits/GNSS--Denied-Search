import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from .simulation_constants import get_simulation_constants

class GridPoint:
    def __init__(self, x, y, z=0):
        self.coord = np.array([x, y, z])
        self.confidence = 0.0

class UAV:
    def __init__(self, start_pos, battery, params):
        self.pos = np.array(start_pos)
        self.battery = battery
        self.drift = 0.0
        self.path = [self.pos.copy()]
        self.params = params
        self.waypoint = None

    def move_to(self, target, velocity):
        dist = np.linalg.norm(target - self.pos)
        energy = dist * self.params['AVERAGE_FLIGHT_POWER'] / velocity
        self.battery -= energy
        self.pos = target
        self.path.append(target.copy())
        return dist, energy

    def reset_drift(self):
        self.drift = 0.0

    def swap_battery(self):
        self.battery = self.params['BATTERY_CAPACITY']

class VNSSearchPipeline:
    def __init__(self, params, grid_spacing):
        self.params = params
        self.grid_spacing = grid_spacing
        self.grid_points = self._create_grid()
        self.ugvs = []
        self.beacons = []

    def _create_grid(self):
        AREA = self.params['AREA']
        x_points = np.arange(0, AREA + 1e-9, self.grid_spacing)
        y_points = np.arange(0, AREA + 1e-9, self.grid_spacing)
        grid = [GridPoint(x, y) for x in x_points for y in y_points]
        return grid

    def initialize_agents(self, num_uavs, num_ugvs, num_beacons):
        self.uavs = []
        for _ in range(num_uavs):
            start_pos = np.random.uniform(0, self.params['AREA'], size=2)
            battery = np.random.uniform(0.5, 1.0) * self.params['BATTERY_CAPACITY']
            self.uavs.append(UAV(np.append(start_pos, self.params['MISSION_ALTITUDE']), battery, self.params))
        # UGVs and Beacons: place at fixed/random positions (for demo)
        self.ugvs = [np.array([self.params['AREA']/4, self.params['AREA']/4, 0]),
                     np.array([3*self.params['AREA']/4, 3*self.params['AREA']/4, 0])]
        self.beacons = [np.array([self.params['AREA']/2, self.params['AREA']/2, 0])]

    def run_cvt_simulation(self, max_steps=1000, coverage_threshold=95, max_time=100):
        # CVT: assign grid points to UAVs using KMeans (Voronoi cells)
        grid_coords = np.array([gp.coord[:2] for gp in self.grid_points])
        kmeans = KMeans(n_clusters=len(self.uavs), n_init=1).fit(grid_coords)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        for i, uav in enumerate(self.uavs):
            uav.waypoint = np.append(centroids[i], self.params['MISSION_ALTITUDE'])

        step = 0
        time_elapsed = 0
        while step < max_steps and time_elapsed < max_time:
            for i, uav in enumerate(self.uavs):
                # Move UAV to centroid of its cell
                velocity = self.params['UAV_MIN_SPEED']
                dist, energy = uav.move_to(uav.waypoint, velocity)
                uav.drift += self.params['DRIFT_RATE'] * dist / velocity
                uav.battery -= energy
                # Support node logic
                for sn in self.ugvs + self.beacons:
                    if np.linalg.norm(uav.pos[:2] - sn[:2]) <= self.params['COVERAGE_RADIUS_UGV']:
                        uav.reset_drift()
                        if sn in self.ugvs:
                            uav.swap_battery()
                # Update grid point confidence for points in UAV's cell
                cell_points = grid_coords[labels == i]
                for pt in cell_points:
                    gp_idx = np.where((grid_coords == pt).all(axis=1))[0][0]
                    gain = max(0, 100 - uav.drift)
                    self.grid_points[gp_idx].confidence = min(100, self.grid_points[gp_idx].confidence + gain)
            # Coverage calculation
            covered_points = sum(1 for gp in self.grid_points if gp.confidence >= coverage_threshold)
            coverage_percent = 100 * covered_points / len(self.grid_points)
            if coverage_percent >= 97:
                print(f"Coverage achieved: {coverage_percent:.2f}% at step {step}, time {time_elapsed:.2f} min")
                break
            if all(uav.battery <= 0 for uav in self.uavs):
                print("All UAVs out of battery. Mission failed.")
                break
            step += 1
            time_elapsed += self.params['DT']
        # Cost calculation
        total_cost = (len(self.uavs) * self.params['UAV_COST'] +
                      len(self.ugvs) * self.params['UGV_COST'] +
                      len(self.beacons) * self.params['BEACON_COST'])
        print(f"Total mission cost: {total_cost}")
        return coverage_percent, total_cost, time_elapsed

    def visualize(self):
        plt.figure(figsize=(8,8))
        for gp in self.grid_points:
            plt.scatter(gp.coord[0], gp.coord[1], c=plt.cm.viridis(gp.confidence/100), s=10)
        for uav in self.uavs:
            path = np.array(uav.path)
            if len(path) > 0:
                plt.plot(path[:,0], path[:,1], label='UAV Path')
        for sn in self.ugvs:
            plt.scatter(sn[0], sn[1], c='blue', marker='s', s=100, label='UGV')
        for sn in self.beacons:
            plt.scatter(sn[0], sn[1], c='green', marker='^', s=100, label='Beacon')
        plt.title('UAV Coverage and Confidence Map')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()

# Example usage: iterate UAVs 5-13, select cheapest config with >=97% coverage in <=100 min
if __name__ == "__main__":
    params = get_simulation_constants()
    best_cost = float('inf')
    best_n = None
    best_time = None
    best_coverage = None
    for n_uav in range(5, 14):
        print(f"\n--- Running simulation for {n_uav} UAVs ---")
        pipeline = VNSSearchPipeline(params, params['GRID_SPACING'])
        pipeline.initialize_agents(num_uavs=n_uav, num_ugvs=2, num_beacons=1)
        coverage, cost, elapsed = pipeline.run_cvt_simulation(max_steps=1000, coverage_threshold=95, max_time=100)
        pipeline.visualize()
        if coverage >= 97 and elapsed <= 100 and cost < best_cost:
            best_cost = cost
            best_n = n_uav
            best_time = elapsed
            best_coverage = coverage
    print(f"\nBest configuration: {best_n} UAVs, Coverage: {best_coverage:.2f}%, Time: {best_time:.2f} min, Cost: {best_cost}")

    def visualize(self):
        # Basic visualization stub
        plt.figure(figsize=(8,8))
        for gp in self.grid_points:
            plt.scatter(gp.coord[0], gp.coord[1], c=plt.cm.viridis(gp.confidence/100), s=10)
        for uav in self.uavs:
            path = np.array(uav.path)
            if len(path) > 0:
                plt.plot(path[:,0], path[:,1], label='UAV Path')
        plt.title('UAV Coverage and Confidence Map')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()

# Example usage (to be removed or moved to main.py)
if __name__ == "__main__":
    params = get_simulation_constants()
    pipeline = VNSSearchPipeline(params, params['GRID_SPACING'])
    pipeline.initialize_agents(num_uavs=5, num_ugvs=2, num_beacons=3)
    pipeline.run_simulation()
    pipeline.visualize()
