import numpy as np
import math

class CostEvaluator:
    def __init__(self):
        # --- COST WEIGHTS (Tuning Parameters) ---
        self.W_COLLISION = 9999999.0  # Safety: Hitting an obstacle is unacceptable
        self.W_OFF_ROAD  = 9999999.0  # Safety: Leaving the drivable area is unacceptable
        
        self.W_JERK      = 10.0       # Comfort: Penalize violent steering/braking
        self.W_LATERAL_D = 5.0        # Efficiency: Penalize lingering on lane lines (prefer d=0, 3.5, -3.5)
        self.W_SPEED     = 2.0        # Efficiency: Penalize driving too slow compared to target speed

        # Vehicle dimensions for collision checking
        self.VEHICLE_RADIUS = 1.5     # Safe bubble around the ego car in meters

    def calculate_total_cost(self, trajectory, obstacles, target_speed):
        """
        Grades a single trajectory. 
        Returns the total penalty score (Lower is better).
        """
        cost = 0.0
        
        # 1. SAFETY COST (Obstacle Avoidance)
        if self._check_collision(trajectory, obstacles):
            return self.W_COLLISION
            
        # 2. COMFORT COST (Jerk and Acceleration)
        cost += self._calculate_comfort_cost(trajectory)
        
        # 3. EFFICIENCY COST (Lane Centering and Speed)
        cost += self._calculate_efficiency_cost(trajectory, target_speed)
        
        return cost

    def _check_collision(self, trajectory, obstacles):
        """
        Checks if any point in the trajectory overlaps with any obstacle.
        obstacles: List of dictionaries [{'s': 50.0, 'd': 0.0}, ...]
        """
        if not obstacles:
            return False
            
        for i in range(len(trajectory['t_points'])):
            ego_s = trajectory['s_points'][i]
            ego_d = trajectory['d_points'][i]
            
            for obs in obstacles:
                # Simple Euclidean distance check in Frenet space
                dist = math.hypot(ego_s - obs['s'], ego_d - obs['d'])
                if dist < self.VEHICLE_RADIUS:
                    return True # COLLISION DETECTED!
                    
        return False

    def _calculate_comfort_cost(self, trajectory):
        """Penalizes high acceleration and high jerk."""
        comfort_penalty = 0.0
        
        # Assuming we have calculated accelerations and jerks in the trajectory dict
        if 's_jerk' in trajectory and 'd_jerk' in trajectory:
            max_s_jerk = np.max(np.abs(trajectory['s_jerk']))
            max_d_jerk = np.max(np.abs(trajectory['d_jerk']))
            comfort_penalty += (max_s_jerk + max_d_jerk) * self.W_JERK
            
        return comfort_penalty

    def _calculate_efficiency_cost(self, trajectory, target_speed):
        """Penalizes slow speeds and driving between lanes."""
        efficiency_penalty = 0.0
        
        # Speed penalty: We want the final speed of the trajectory to match target_speed
        final_speed = trajectory['s_dot'][-1] if 's_dot' in trajectory else 0.0
        speed_diff = abs(target_speed - final_speed)
        efficiency_penalty += speed_diff * self.W_SPEED
        
        # Lateral penalty: We want the final 'd' to be exactly in the center of a lane
        final_d = trajectory['d_points'][-1]
        
        # Modulo math to find out how far we are from a lane center (0, 3.5, 7.0, etc.)
        lane_width = 3.5
        dist_from_center = abs((final_d % lane_width) - (lane_width / 2.0))
        # We invert it so that being ON the line (dist = 1.75) gives a high penalty
        lane_penalty = lane_width / 2.0 - dist_from_center
        
        efficiency_penalty += lane_penalty * self.W_LATERAL_D
        
        return efficiency_penalty
