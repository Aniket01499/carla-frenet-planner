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

        print("Lateral Target point wrt time", end_d, end_v, end_a)  

    def cal_lateral_cord(self, t):       # Defined a method
        d = self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4 + self.a5*t**5
        v = self.a1 + 2*self.a2*t + 3*self.a3*t**2 + 4*self.a4*t**3 + 5*self.a5*t**4
        a = 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2 + 20*self.a5*t**3
        j = 6*self.a3 + 24*self.a4*t + 60*self.a5*t**2
        return d, v, a, j

# Longitudinal Quartic equation
class QuarticPolynom:
    def __init__(self, start_ld, start_lv, start_la, end_lv, end_la, T):
        # Using 'ld' for longitudinal distance (s), 'lv' for velocity (s_dot), 'la' for acceleration (s_ddot)
        self.a_0 = start_ld
        self.a_1 = start_lv
        self.a_2 = start_la / 2.0

        a_mat = np.matrix([
            [T**3, T**4],
            [3*T**2, 4*T**3],
            [6*T, 12*T**2]
        ])

        A_mat = np.linalg.inv(a_mat)

        B_mat = np.matrix([
            [end_lv - (self.a_1 + 2*self.a_2*T)],
            [end_la - (2*self.a_2)]
        ])

        x1 = np.linalg.solve(A_mat, B_mat)
        
        self.a3 = x1[0, 0]
        self.a4 = x1[1, 0]

        # print("Longitudinal Target velociy and acceleration wrt time", end_lv, end_la)

    def cal_longitudinal_cord(self, t):
        s = self.a_0 + self.a_1*t + self.a_2*t**2 + self.a3*t**3 + self.a4*t**4
        vl = self.a_1 + 2*self.a_2*t + 3*self.a3*t**2 + 4*self.a4*t**3
        al = 2*self.a_2 + 6*self.a3*t + 12*self.a4*t**2
        jl = 6*self.a3 + 24*self.a4*t
        return s, vl, al, jl
        
def generate_frenet_paths(c_s, c_s_v, c_s_a, c_d, c_d_v, c_d_a):
    """
    Generates parallel (s, d, t) trajectories.
    c_s: current s position (longitudinal)
    c_s_v: current s velocity
    c_s_a: current s acceleration
    c_d: current d position (lateral)
    c_d_v: current d velocity
    c_d_a: current d acceleration
    """

    paths = []

    # Left, center and right lanes
    target_lanes = [3.5, 0, -3.5]

    # Times of maneuvers
    target_times = [3, 4, 5]

    # Target speeds (Maintain speed, speed up slightly, slow down slightly)
    target_speeds = [c_s_v - 5.0, c_s_v, c_s_v + 5.0]

    for target_d in target_lanes:
        for T in target_times:
        
            # Create the quintic equation. Target lateral velocity and accelration is 0.
            lat_poly = QuinticPolynom(start_d = c_d, start_v = c_d_v, start_a = c_d_a,
                                end_d = target_d, end_v = 0, end_a = 0, T = T)
            
            for target_v in target_speeds:
                # Create the quartic equation. Target longitudinal acceleration is 0.
                lon_poly = QuarticPolynom(start_ld = c_s, start_lv = c_s_v, start_la = c_s_a,
                                    end_lv = target_v, end_la = 0, T = T)

                # Time steps and points plotting
                path_t = []
                path_d, path_d_jerk = [], []
                path_s, path_s_vel, path_s_jerk = [], [], []

                # Time step of 0.1 seconds untill T
                for t in np.arange(0, T, 0.1):
                    d, d_v, d_a, d_j = lat_poly.cal_lateral_cord(t)
                    s, s_v, s_a, s_j = lon_poly.cal_longitudinal_cord(t)
                    
                    path_t.append(t)
                    
                    path_d.append(d)
                    path_d_jerk.append(d_j)
                    
                    path_s.append(s)
                    path_s_vel.append(s_v)
                    path_s_jerk.append(s_j)

                paths.append({
                    "target_lane": target_d,
                    "target_speed": target_v,
                    "time_to_complete": T,
                    "t_points": path_t,                
                    "d_points": path_d,
                    "d_jerk": path_d_jerk,
                    "s_points": path_s,
                    "s_dot": path_s_vel,
                    "s_jerk": path_s_jerk
                })

    return paths

# Quick test to make sure the math works before plugging it into CARLA!
print("Testing Combined Trajectory Generator...")

# Assume we are at s=0, going 20m/s straight down the center lane (d=0)
bundle = generate_frenet_paths(c_s=0.0, c_s_v=20.0, c_s_a=0.0, c_d=0.0, c_d_v=0.0, c_d_a=0.0)

print(f"Generated {len(bundle)} possible trajectories.")
for path in bundle:
    print(f"Aiming for d={path['target_lane']}m at {path['target_speed']}m/s in {path['time_to_complete']}s. (Generated {len(path['d_points'])} waypoints)")