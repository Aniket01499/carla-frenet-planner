import numpy as np

# Define lateral quintic equation
class QuinticPolynom:
    def __init__(self, start_d, start_v, start_a, end_d, end_v, end_a, T):
        
        #define coeficients
        self.a0 = start_d
        self.a1 = start_v
        self.a2 = start_a / 2.0

        a = np.matrix([
            [T**3, T**4, T**5],
            [3*T**2, 4*T**3, 5*T**4],
            [6*T, 12*T**2, 20*T**3]
        ])

        A = np.linalg.inv(a)

        B = np.matrix([
            [end_d - (self.a0 + self.a1*T + self.a2*T**2)],
            [end_v - (self.a1 + 2*self.a2*T)],
            [end_a - (2*self.a2)]
        ])

        x = np.linalg.solve(A, B)

        self.a3 = x[0, 0]
        self.a4 = x[1, 0]
        self.a5 = x[2, 0]

        print("Target point", end_d, end_v, end_a)  

    def cal_lateral_cord(self, t):       # Defined a method
        d = self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4 + self.a5*t**5
        return d    

def generate_frenet_paths(current_d, current_v, current_a):
    # Generate parallel trajectories and return smooth paths as list.

    paths = []

    # Left, center and right lanes
    target_lanes = [3.5, 0, -3.5]

    # Times of maneuvers
    target_times = [3, 4, 5]

    for target_d in target_lanes:
        for T in target_times:
        
            #Create the quintic equation and end velocity and accelration is 0.
            poly = QuinticPolynom(start_d = current_d, start_v = current_v, start_a = current_a,
                                end_d = target_d, end_v = 0, end_a = 0, T = T)
            
            # Time steps and points plotting
            path_d = []
            path_t = []

            # Time step of 0.1 seconds untill T
            for t in np.arange(0, T, 0.1):
                path_d.append(poly.cal_lateral_cord(t))
                path_t.append(t)

            paths.append({
                "target_lane": target_d,
                "time_to_complete": T,
                "d_points": path_d,
                "t_points": path_t,                
            })

    return paths

# aryan = QuinticPolynom(0, 40, 0, 3.5, 0, 0, 3)

# ani = generate_frenet_paths(0, 40, 0)
# print( ani)

# print("0done")

# Quick test to make sure our math works before we plug it into CARLA!
    print("Testing Quintic Trajectory Generator...")
# Assume we are in the center lane (d=0), going straight, and want to evaluate options
bundle = generate_frenet_paths(current_d=0.0, current_v=40.0, current_a=0.0)

print(f"Generated {len(bundle)} possible trajectories.")
for path in bundle:
    print(f"Aiming for d={path['target_lane']}m in {path['time_to_complete']}s. (Generated {len(path['d_points'])} waypoints)")