import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import time
import numpy as np
import cv2
import pygame
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import h5py

import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu
import tensorflow as tf

from PIL import Image, ImageFont, ImageDraw

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpu = 0
setup_gpu(gpu)

#loading and converting model
model_path = 'resnet50_coco_best_v2.1.0.h5'
model = models.load_model(model_path, backbone_name='resnet50')
# model = models.convert_model(model)

# load label to names mapping for visualization purposes
labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

try:
    import queue
except ImportError:
    import Queue as queue

camera_w=800 #640
camera_h=600 #480

height = 480
width = 640


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))

def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False
    

def obstacle_img(image):

    img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))

    img = img.reshape(image.height, image.width, 4) #4 dimensions because the image also has the 'alpha' channel, that we will ignore for now

    img = img[:,:,:3]
    

    # copy to draw on
    draw = img.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    #image = preprocess_image(img)
    image, scale = resize_image(img)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break
        
        color = label_color(label)
    
        b = box.astype(int)
        draw_box(draw, b, color=color)
    
        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
    
    #plt.figure(figsize=(15, 15))
    #plt.axis('off')
    #plt.imshow(draw)
    #plt.show()
    return draw
  

        
def process_img(image):

    img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))

    img = img.reshape(image.height, image.width, 4) #4 dimensions because the image also has the 'alpha' channel, that we will ignore for now

    img = img[:,:,:3]
    
    img = img[:, :, ::-1]

    return img
    
    
def get_surface(image):

    image_surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
    
    return image_surface
    
def pygame_plot(display, image_list, text_list, screen_width, screen_height, font):

    offset = 6
    
    canvas = pygame.Surface((screen_width, screen_height))

    if len(image_list)==0:
        pass
    if len(image_list)==1:
        img_surf = get_surface(image_list[0])
        img_surf = pygame.transform.scale(img_surf, (screen_width, screen_height))
        display.blit(img_surf, (0,0))
        display.blit(font.render(text_list[0], True, (255, 255, 255)), (offset,offset))
        
    else:
        
        if len(image_list)==2:
            img_surf_1 = get_surface(image_list[0])
            img_surf_2 = get_surface(image_list[1])
            
            img_surf_3 = canvas.subsurface(pygame.Rect(0,int(screen_height/2),int(screen_width/2),int(screen_height/2)))
            img_surf_4 = canvas.subsurface(pygame.Rect(int(screen_width/2),int(screen_height/2),int(screen_width/2),int(screen_height/2)))
            
            text_list.append('')
            text_list.append('')
            
            
            img_surf_1=pygame.transform.scale(img_surf_1, (int(screen_width/2), int(screen_height/2)))
            img_surf_2=pygame.transform.scale(img_surf_2, (int(screen_width/2), int(screen_height/2)))
            
            display.blit(font.render(text_list[0], True, (255, 255, 255)), (offset,offset))
            display.blit(font.render(text_list[1], True, (255, 255, 255)), (int(screen_width/2)+offset,offset))
     
        if len(image_list)==3:
            img_surf_1 = get_surface(image_list[0])
            img_surf_2 = get_surface(image_list[1])
            img_surf_3 = get_surface(image_list[2])
            
            img_surf_4 = canvas.subsurface(pygame.Rect(int(screen_width/2),int(screen_height/2),int(screen_width/2),int(screen_height/2)))
            text_list.append('')
            
            
            img_surf_1=pygame.transform.scale(img_surf_1, (int(screen_width/2), int(screen_height/2)))
            img_surf_2=pygame.transform.scale(img_surf_2, (int(screen_width/2), int(screen_height/2)))
            img_surf_3=pygame.transform.scale(img_surf_2, (int(screen_width/2), int(screen_height/2)))
            
            display.blit(font.render(text_list[0], True, (255, 255, 255)), (offset,offset))
            display.blit(font.render(text_list[1], True, (255, 255, 255)), (int(screen_width/2)+offset,offset))
            display.blit(font.render(text_list[2], True, (255, 255, 255)), (offset, int(screen_height/2)+offset))
        
        if len(image_list)==4:
            img_surf_1 = get_surface(image_list[0])
            img_surf_2 = get_surface(image_list[1])
            img_surf_3 = get_surface(image_list[2])
            img_surf_4 = get_surface(image_list[3])
            
            img_surf_1 = pygame.transform.scale(img_surf_1, (int(screen_width/2), int(screen_height/2)))
            img_surf_2 = pygame.transform.scale(img_surf_2, (int(screen_width/2), int(screen_height/2)))
            img_surf_3 = pygame.transform.scale(img_surf_3, (int(screen_width/2), int(screen_height/2)))
            img_surf_4 = pygame.transform.scale(img_surf_4, (int(screen_width/2), int(screen_height/2)))
            

            
        pygame.draw.line(img_surf_2, (255,255,255), (0,0), (0,int(screen_height/2)), 4)
        pygame.draw.line(img_surf_4, (255,255,255), (0,0), (0,int(screen_height/2)), 4)
        pygame.draw.line(img_surf_3, (255,255,255), (0,0), (int(screen_width/2),0), 4)
        pygame.draw.line(img_surf_4, (255,255,255), (0,0), (int(screen_width/2),0), 4)
        
        display.blit(img_surf_1, (0,0))
        display.blit(img_surf_2, (int(screen_width/2),0))
        display.blit(img_surf_3, (0, int(screen_height/2)))
        display.blit(img_surf_4, (int(screen_width/2), int(screen_height/2)))
        
        display.blit(font.render(text_list[0], True, (255, 255, 255)), (offset,offset))
        display.blit(font.render(text_list[1], True, (255, 255, 255)), (int(screen_width/2)+offset,offset))
        display.blit(font.render(text_list[2], True, (255, 255, 255)), (offset, int(screen_height/2)+offset))
        display.blit(font.render(text_list[3], True, (255, 255, 255)), (int(screen_width/2)+offset, int(screen_height/2)+offset))
        



def main():
    actor_list = []
    pygame.init()

    #Define pygame display
    screen_width=1000
    screen_height=600
    window = (screen_width ,screen_height)
    display = pygame.display.set_mode(window, pygame.HWSURFACE | pygame.DOUBLEBUF)
        
        
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()

    #world = client.load_world('Town02')
    
    #world.set_weather(carla.WeatherParameters.ClearNoon)

    try:
        m = world.get_map()
        start_pose = random.choice(m.get_spawn_points())
        waypoint = m.get_waypoint(start_pose.location)

        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(blueprint_library.find('vehicle.tesla.model3'), start_pose)
        actor_list.append(vehicle)
        #vehicle.set_simulate_physics(False)
        
        #Set spectator location
        spectator = world.get_spectator()
        transform = vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=10)))
        carla.Rotation(pitch=-90)

        
        #Define sensors
        #First RGB camera to perform object and lane detection

        cam_bp = blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x", f"{camera_w}")
        cam_bp.set_attribute("image_size_y", f"{camera_h}")
        cam_bp.set_attribute("fov", "120")

        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=2.5, z=1.5)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)
        
        #Second RGB camera to follow the vehicle
        #camera_view = world.spawn_actor(
        #    blueprint_library.find('sensor.camera.rgb'),
        #    carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
        #    attach_to=vehicle)
        #actor_list.append(camera_view)
        
        #Basic agent
        #agent = BasicAgent(vehicle, target_speed=30)
        vehicle.set_autopilot(True)
        #Set agent destination
        #spawn_point = m.get_spawn_points()[0]
        #destination = (spawn_point.location.x, spawn_point.location.y, spawn_point.location.z)
        #agent.set_destination(destination)

        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, fps=30) as sync_mode:   #camera_view,
            while True:
                if should_quit():
                    return
                clock.tick()

                # Advance the simulation and wait for the data.
                snapshot, image_rgb = sync_mode.tick(timeout=1.0) #, img_view
                
                #Reshape the images
                image_rgb = obstacle_img(image_rgb)
                #img_view = process_img(img_view)

                #Process sensor data
                #image_rgb = cv2.rectangle(np.array(image_rgb) ,(10,10), (100,100), (255,255,255), 2)
                
                #Control vehicle
                #control = agent.run_step()
                #vehicle.apply_control(control)
                
                #Visualize outputs

                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                #image list to visualize
                image_list=[image_rgb] #, img_view
                #text list to display
                text_list=['image_rgb']  #, 'img_view'
                
                pygame_plot(display, image_list, text_list, screen_width, screen_height, font)
                
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (screen_width-120, screen_height-50))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (screen_width-120, screen_height-30))
                
                pygame.display.flip()

                print('Client FPS:', clock.get_fps())
                
                #if vehicle.get_location() == destination:
                #    print("Target reached, mission accomplished...")
                #    break
              

    finally:

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')