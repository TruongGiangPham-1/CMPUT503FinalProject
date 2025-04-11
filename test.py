import gym_duckietown
from gym_duckietown.simulator import Simulator
import cv2
import numpy as np
env = Simulator(
        seed=123, # random seed
        map_name="loop_empty",
        max_steps=500001, # we don't want the gym to reset itself
        domain_rand=0,
        camera_width=640,
        camera_height=480,
        accept_start_angle_deg=4, # start close to straight
        full_transparency=True,  # additional robot info returned in info dict
        distortion=True,
    )   
while True:
    action = [0.1,0.1]
    observation, reward, done, misc = env.step(action)


    img = env.render()  # (640, 480, 3) (RGB)
    # grayscale the image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)  # downsample to 84x84 like atari

    # Assuming `img` is your (H, W, 3) NumPy array
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #img = cv2.flip(img, 0)  # 0 = flip vertically
    cv2.imshow("Image", img)
    cv2.waitKey(0)  # Waits indefinitely until a key is pressed
    cv2.destroyAllWindows()

    print(f'image shape: {img.shape} type {type(img)}')  # (640, 480, 3)
    if done:
        env.reset()
    break