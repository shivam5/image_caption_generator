import object_detection
import cv2
import tensorflow as tf
import numpy as np
import os
import copy
import sys

net = object_detection.Net(graph_fp='ConvNets/frozen_inference_graph.pb',
                           labels_fp='data/label.pbtxt',
                           num_classes=90,
                           threshold=0.6)
CAMERA_MODE = 'camera'
STATIC_MODE = 'static'
IMAGE_SIZE = 320

if __name__ == '__main__':


    if len(sys.argv)!=3:
        print("The correct syntax is: python det_object.py 'source_image_folder' 'target_image_folder'")
        exit(0)

    load_folder = str(sys.argv[1])
    save_folder = str(sys.argv[2])

    image_names = set(os.listdir(load_folder)).difference(set(os.listdir(save_folder)))


    # img_name = '23445819.jpg'
    for img_name in image_names:

        load_path = os.path.join(load_folder, img_name)
        save_path = os.path.join(save_folder, img_name)  

        img = cv2.imread(load_path)
        # im = copy.deepcopy(img)
        height, width, channels = img.shape

        detections = net.predict(img=img, display_img=img)
        
        new_image = np.zeros((height,width,channels), np.uint8)

        for det in detections:
            bounding_box = det["bb"]
            print(bounding_box)

            y1 = int(bounding_box[0] * height)
            x1 = int(bounding_box[1] * width)
            y2 = int(bounding_box[2] * height)
            x2 = int(bounding_box[3] * width)

            new_image[y1:y2, x1:x2, :] = img[y1:y2, x1:x2, :]

        # image_to_write = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, new_image.astype(np.uint8) )

        # image_to_write = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(save_path, image_to_write)
        