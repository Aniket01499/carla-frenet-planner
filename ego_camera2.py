import carla
import numpy as np
import cv2
import queue
import time

def process_img(image):
    # Convert CARLA raw image data to a numpy array
    i = np.array(image.raw_data)
    i2 = i.reshape((image.height, image.width, 4))
    
    # Extract the Semantic tags (Red channel)
    tags = i2[:, :, 2]
    
    # Create a blank canvas
    perception_view = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    
    # Color-code the drivable space and obstacles
    perception_view[tags == 7] = (128, 64, 128)  # Roads (Purple)
    perception_view[tags == 10] = (255, 0, 0)    # Vehicles (Blue)
    perception_view[tags == 4] = (0, 0, 255)     # Pedestrians (Red)
    perception_view[tags == 1] = (70, 70, 70)    # Buildings (Dark Gray)
    
    # Display the perception view
    cv2.imshow("Semantic Perception", perception_view)
    cv2.waitKey(1)

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # --- 1. Set up the Traffic Manager for Safe Driving ---
    tm = client.get_trafficmanager(8000)
    tm_port = tm.get_port()
    
    # Global rules for all TM vehicles
    tm.global_percentage_speed_difference(30.0) # Drive 30% below the speed limit

    # --- 2. Spawn Ego Vehicle ---
    bp = blueprint_library.filter('model3')[0]
    spawn_points = world.get_map().get_spawn_points()
    # Pick a spawn point further down the list to avoid spawning inside traffic
    spawn_point = spawn_points[10] if len(spawn_points) > 10 else spawn_points[0]
    
    vehicle = world.spawn_actor(bp, spawn_point)
    
    # Hand vehicle to Traffic Manager and apply strict ego-rules
    vehicle.set_autopilot(True, tm_port)
    tm.ignore_lights_percentage(vehicle, 0.0)    # NEVER run red lights
    tm.ignore_signs_percentage(vehicle, 0.0)     # Obey stop signs

    print("Cautious Ego vehicle spawned!")

    # --- 3. Spawn Semantic Camera ---
    cam_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    cam_bp.set_attribute('image_size_x', '640')
    cam_bp.set_attribute('image_size_y', '360')
    cam_bp.set_attribute('fov', '90')
    
    spawn_point_cam = carla.Transform(carla.Location(x=1.5, z=1.4))
    camera = world.spawn_actor(cam_bp, spawn_point_cam, attach_to=vehicle)

    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    try:
        print("Streaming Semantic Perception... Press 'q' in window or Ctrl+C in terminal to stop.")
        while True:
            image = image_queue.get()
            process_img(image)

    except KeyboardInterrupt:
        print("\nInterrupted by user...")
    finally:
        print("Destroying actors...")
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
