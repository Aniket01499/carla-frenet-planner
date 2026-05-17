import carla
import numpy as np
import cv2
import queue

def process_img(image):
    # Convert CARLA raw image data to a numpy array
    i = np.array(image.raw_data)
    i2 = i.reshape((image.height, image.width, 4))
    
    # In CARLA's Semantic Segmentation, the class ID is stored in the Red channel (index 2 in BGRA)
    tags = i2[:, :, 2]
    
    # Create a blank black canvas to draw our perception data on
    perception_view = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    
    # Color-code the drivable space and obstacles!
    # BGR format for OpenCV
    perception_view[tags == 7] = (128, 64, 128)  # Roads become Purple
    perception_view[tags == 10] = (255, 0, 0)    # Vehicles become Blue
    perception_view[tags == 4] = (0, 0, 255)     # Pedestrians become Red
    perception_view[tags == 1] = (70, 70, 70)    # Buildings become dark gray
    
    # Display the perception view
    cv2.imshow("Semantic Perception", perception_view)
    cv2.waitKey(1)

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.set_autopilot(True)

    # 1. SWAPPED TO SEMANTIC SEGMENTATION
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
        print("Interrupted by user...")
    finally:
        print("Destroying actors...")
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
