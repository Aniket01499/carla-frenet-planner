import numpy as np

def get_closest_waypoint(x, y, map_x, map_y):
    """Finds the index of the closest map waypoint to the car."""
    distances = np.sqrt((map_x - x)**2 + (map_y - y)**2)
    return np.argmin(distances)

def get_next_waypoint(x, y, theta, map_x, map_y):
    """Finds the index of the waypoint immediately in front of the car."""
    closest_idx = get_closest_waypoint(x, y, map_x, map_y)
    
    map_vec_x = map_x[closest_idx] - x
    map_vec_y = map_y[closest_idx] - y
    
    # Heading vector of the car
    heading_vec_x = np.cos(theta)
    heading_vec_y = np.sin(theta)
    
    # Dot product to check if the closest waypoint is behind us
    dot_product = (map_vec_x * heading_vec_x) + (map_vec_y * heading_vec_y)
    
    if dot_product < 0:
        closest_idx += 1
        
    return closest_idx

def get_frenet(x, y, theta, map_x, map_y, map_s):
    """
    Transforms Cartesian (x, y) to Frenet (s, d).
    map_x, map_y: Arrays of the reference path coordinates.
    map_s: Array of cumulative distances along the reference path.
    """
    next_wp = get_next_waypoint(x, y, theta, map_x, map_y)
    prev_wp = next_wp - 1
    
    # Edge case: if we are at the very beginning of the map
    if next_wp == 0:
        prev_wp = len(map_x) - 1

    # The Road Vector (from previous waypoint to next waypoint)
    n_x = map_x[next_wp] - map_x[prev_wp]
    n_y = map_y[next_wp] - map_y[prev_wp]
    
    # The Car Vector (from previous waypoint to the car)
    x_x = x - map_x[prev_wp]
    x_y = y - map_y[prev_wp]
    
    # Find the projection of the Car Vector onto the Road Vector
    proj_norm = (x_x * n_x + x_y * n_y) / (n_x**2 + n_y**2)
    proj_x = proj_norm * n_x
    proj_y = proj_norm * n_y
    
    # Calculate d (Lateral distance) using the cross product logic
    d = np.sqrt((x_x - proj_x)**2 + (x_y - proj_y)**2)
    
    # Determine if d is positive (right) or negative (left) of the center line
    cross_product = (n_x * x_y) - (n_y * x_x)
    if cross_product > 0:
        d *= -1
        
    # Calculate s (Longitudinal distance)
    s = map_s[prev_wp] + np.sqrt(proj_x**2 + proj_y**2)
    
    return s, d


def get_cartesian(s, d, map_x, map_y, map_s):
    """
    Transforms Frenet (s, d) to Cartesian (x, y).
    map_x, map_y: Arrays of the reference path coordinates.
    map_s: Array of cumulative distances along the reference path.
    """
    # 1. Find the map segment that contains our target 's'
    prev_wp = 0
    # Loop through the map's cumulative 's' distances.
    # Stop when the NEXT waypoint's 's' distance is strictly greater than our target 's'.
    # This means our target 's' lies somewhere between prev_wp and prev_wp + 1.
    while prev_wp < len(map_s) - 1 and map_s[prev_wp+1] < s:
        prev_wp += 1
        
    # 2. Set the next waypoint index
    next_wp = prev_wp + 1
    # Edge case: If 's' extends beyond our known map_s array bounds,
    # clamp the indices to the very last segment of the map to avoid an IndexError.
    if next_wp >= len(map_s):
        next_wp = len(map_s) - 1
        prev_wp = next_wp - 1
        
    # Road vector
    n_x = map_x[next_wp] - map_x[prev_wp]
    n_y = map_y[next_wp] - map_y[prev_wp]
    
    # Heading of the road
    heading = np.arctan2(n_y, n_x)
    
    # Interpolate the reference (s=0) Cartesian coordinates
    seg_s = map_s[next_wp] - map_s[prev_wp]
    
    ratio = (s - map_s[prev_wp]) / seg_s if seg_s != 0 else 0
    x_ref = map_x[prev_wp] + ratio * n_x
    y_ref = map_y[prev_wp] + ratio * n_y
        
    # Add the lateral (d) offset perpendicularly
    x = x_ref - d * np.sin(heading)
    y = y_ref + d * np.cos(heading)
    
    return x, y
