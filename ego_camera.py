import carla
import numpy as np
import cv2
import queue

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # 1. Spawn the Ego Vehicle (Tesla Model 3)
    bp = blueprint_library.filter('model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(bp, spawn_point)
    
    vehicle.set_autopilot(True)
    print("Ego vehicle spawned!")

    # 2. Spawn and attach the Camera
    cam_bp = blueprint_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '640')
    cam_bp.set_attribute('image_size_y', '360')
    cam_bp.set_attribute('fov', '90')
    
    spawn_point_cam = carla.Transform(carla.Location(x=1.5, z=1.4))
    camera = world.spawn_actor(cam_bp, spawn_point_cam, attach_to=vehicle)

    # 3. Create a Queue and listen
    image_queue = queue.Queue()
    camera.listen(image_queue.put)  # Just toss the raw data into the queue

    try:
        print("Streaming camera feed... Press 'q' in the image window or Ctrl+C in terminal to stop.")
        while True:
            # Main thread waits to get an image from the queue
            image = image_queue.get()
            
            # Process the image data on the main thread
            i = np.array(image.raw_data)
            i2 = i.reshape((image.height, image.width, 4))
            i3 = i2[:, :, :3]
            
            # Display it safely
            cv2.imshow("Ego Vehicle Camera", i3)
            
            # cv2.waitKey(1) gives OpenCV the time it needs to render the frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user...")
    finally:
        print("Destroying actors...")
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
