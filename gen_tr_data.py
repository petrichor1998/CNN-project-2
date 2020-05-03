import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os
from PIL import ImageGrab


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

def main():
    
    file_name = 'training_data.npy'

    if os.path.isfile(file_name):
        print('File exists, loading previous data!')
        training_data = list(np.load(file_name), allow_pickle = True)
    else:
        print('File does not exist, starting fresh!')
        training_data = []



    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    while (True):
        screen_grab = ImageGrab.grab(bbox=(123, 499, 408, 570))
        screen_arr = np.array(screen_grab)
        screen_gray = cv2.cvtColor(screen_arr, cv2.COLOR_BGR2GRAY)
        # resize to something a bit more acceptable for a CNN
        screen_resized = cv2.resize(screen_gray, (80, 60))
        keys = key_check()
        output = keys_to_output(keys)
        training_data.append([screen_resized, output])


        if len(training_data) % 500 == 0:
            print(len(training_data))
            np.save(file_name, training_data)
main()
