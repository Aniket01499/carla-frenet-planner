import carla
import numpy as np
import cv2
import queue
import random
from global_planner import get_forward_centerline, visualize_path
from frenet_math import get_frenet

def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((image.height, image.width, 4))
    tags = i2[:, :, 2]
    
    perception_view = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    perception_view[tags == 7] = (128, 64, 128)  # Roads (Purple)
    perception_view[tags == 10] = (255, 0, 0)    # Vehicles (Blue)
    perception_view[tags == 4] = (0, 0, 255)     # Pedestrians (Red)
    perception_view[tags == 1] = (70, 70, 70)    # Buildings (Dark Gray)
    
    cv2.imshow("Semantic Perception", perception_view)
    cv2.waitKey(1)

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    
    # --- 1. ENABLE SYNCHRONOUS MODE ---
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # Force exactly 20 FPS (Perfect for MPC math)
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()

    # Set up Traffic Manager to sync with the server
    tm = client.get_trafficmanager(8000)
    tm_port = tm.get_port()
    tm.set_synchronous_mode(True)

    # Spawn Ego Vehicle
    bp = blueprint_library.filter('model3')[0]
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = spawn_points[10] if len(spawn_points) > 10 else spawn_points[0]
    
    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.set_autopilot(True, tm_port)
    print("Ego vehicle spawned!")

    # --- 2. SPAWN BACKGROUND TRAFFIC ---
    print("Spawning 50 background traffic vehicles...")
    
    # Get all vehicle blueprints and filter for only 4-wheeled vehicles (no bikes)
    blueprints_vehicles = blueprint_library.filter('vehicle.*')
    blueprints_vehicles = [x for x in blueprints_vehicles if int(x.get_attribute('number_of_wheels')) == 4]
    
    # Remove the ego vehicle's spawn point so we don't spawn a car on top of ourselves
    spawn_points.remove(spawn_point)
    random.shuffle(spawn_points)
    
    # Define CARLA batch commands for efficiency
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor
    
    batch = []
    npc_vehicles_list = [] # Keep track of them so we can delete them later
    
    # Ask CARLA to spawn 50 cars at random locations
    for n, transform in enumerate(spawn_points[:50]):
        bp = random.choice(blueprints_vehicles)
        # Spawn the car and immediately hand it over to the Traffic Manager
        batch.append(SpawnActor(bp, transform).then(SetAutopilot(FutureActor, True, tm_port)))
        
    # Send the batch to the server
    responses = client.apply_batch_sync(batch, True)
    
    for response in responses:
        if not response.error:
            npc_vehicles_list.append(response.actor_id)
            
    print(f"Successfully spawned {len(npc_vehicles_list)} traffic vehicles.")

    # Spawn Semantic Camera
    cam_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    cam_bp.set_attribute('image_size_x', '640')
    cam_bp.set_attribute('image_size_y', '360')
    cam_bp.set_attribute('fov', '90')
    
    spawn_point_cam = carla.Transform(carla.Location(x=1.5, z=1.4))
    camera = world.spawn_actor(cam_bp, spawn_point_cam, attach_to=vehicle)

    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    try:
        print("Streaming in SYNCHRONOUS MODE... Press Ctrl+C in terminal to stop.")
        while True:
            # 1. Tell the server to compute exactly ONE frame of physics
            world.tick()
            
            # 2. Extract the HD Map Centerline (100 meters ahead)
            map_x, map_y, map_s = get_forward_centerline(vehicle, world.get_map())
            
            # 3. Draw the invisible centerline in the simulator so we can see it!
            visualize_path(world, map_x, map_y)
            
            # 4. Process Semantic Camera (for obstacles)
            image = image_queue.get()
            process_img(image)

    except KeyboardInterrupt:
        print("\nInterrupted by user...")
    finally:
        print("Restoring Asynchronous mode and destroying actors...")
        # WE MUST TURN OFF SYNC MODE BEFORE EXITING, OR THE SERVER WILL FREEZE
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
