import carla
import numpy as np

def get_forward_centerline(vehicle, carla_map, distance=100, interval=1.0):
    """
    Extracts the lane centerline ahead of the vehicle.
    Returns map_x, map_y, map_s as numpy arrays for Frenet calculations.
    """
    # 1. Find the invisible map waypoint closest to the car's current location
    waypoint = carla_map.get_waypoint(vehicle.get_location())
    
    map_x = []
    map_y = []
    map_s = []
    
    current_s = 0.0
    
    # 2. Project forward along the lane center
    for _ in range(int(distance / interval)):
        map_x.append(waypoint.transform.location.x)
        map_y.append(waypoint.transform.location.y)
        map_s.append(current_s)
        
        # Ask CARLA for the next waypoint in the center of this specific lane
        next_wps = waypoint.next(interval)
        
        # If the road ends, stop extracting
        if not next_wps:
            break
            
        waypoint = next_wps[0]
        current_s += interval
        
    return np.array(map_x), np.array(map_y), np.array(map_s)

def visualize_path(world, map_x, map_y, z_height=1.5):
    """
    Draws a green dotted line in the CARLA simulator so we can see the reference path.
    """
    for x, y in zip(map_x, map_y):
        loc = carla.Location(x=x, y=y, z=z_height)
        # Draw a small green dot that lasts for 0.1 seconds (1 frame)
        world.debug.draw_point(loc, size=0.1, color=carla.Color(0, 255, 0), life_time=0.1)