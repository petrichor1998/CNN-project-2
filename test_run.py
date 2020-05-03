import numpy as np
from PIL import ImageGrab
import cv2
import time
from getkeys import key_check
import keyboard
from AlexNet import alexnet

WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 3
MODEL_NAME = 'T-rex{}-{}-{}-epochs.model'.format(LR, 'alexnetv2',EPOCHS)

def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array

    [Jump,Not_jump] boolean values.
    '''
    output = [0, 0]

    if ' ' in keys:
        output[0] = 1
    else:
        output[1] = 1
    return output

def jump():
    keyboard.press(' ')

def main():
    model = alexnet(WIDTH, HEIGHT, LR)
    model.load(MODEL_NAME)
    while(True):
            # 800x600 windowed mode
            screen_grab = ImageGrab.grab(bbox=(123, 499, 408, 570))
            screen_arr = np.array(screen_grab)
            last_time = time.time()
            screen_gray = cv2.cvtColor(screen_arr, cv2.COLOR_BGR2GRAY)
            # resize to something a bit more acceptable for a CNN
            screen_resized = cv2.resize(screen_gray, (80, 60))
            moves = list(np.around(model.predict([screen_resized.reshape(80, 60, 1)])[0]))
            print(moves)
            if moves == [1,0]:
                jump()
main()
