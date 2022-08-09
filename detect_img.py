import os
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
from object_detection.utils import label_map_util,visualization_utils as v_utils,config_util
from object_detection.builders import model_builder
import matplotlib
from matplotlib import pyplot as plt

# taken from: 
# https://github.com/jakkcoder/Widows-Object-Detection-Setup/blob/main/object_detection_tutorial.ipynb
wd = os.getcwd()
#MODEL_PATH = os.path.join(wd,'model_graph','new_graph','saved_model');
#LABELS_PATH = os.path.join(wd,'Annotations','label_map.pbtxt');

MODEL_PATH = "" # add "model_graph" path from the drive
LABELS_PATH = "" # add the "Annotations" folder path 

# load the trained model onto memory
detection_model = tf.saved_model.load(MODEL_PATH)

# list of strings to add correct label for each box
category_index = label_map_util.create_category_index_from_labelmap(LABELS_PATH,use_display_name=True)

# checks the model input signature which expects a batch of 3-color images of type uint8
def single_model_inference(model,image):
    # conver image to numpy array
    image = np.asarray(image)
    # the model takes a tensor of image
    input_tensor = tf.convert_to_tensor(image)
    # model expects batch of images so tf.newaxis
    input_tensor = input_tensor[tf.newaxis,...]
    # running inference
    detect_fn = model.signatures['serving_default']
    output_dict = detect_fn(input_tensor)
    
    # convert all batch tensor output to numpy array
    # taking index [0] to remove the batch dimension
    # only interested in first number of detection
    num_detection = int(output_dict.pop('num_detections'))
    
    output_dict = {key:value[0, :num_detection].numpy() for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detection
    # detection classes must be an integer
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
    
    return output_dict;


def show_inference(model,image_np):
    # perform the detection
    output_dict = single_model_inference(model, image_np)
    # visualize the detected object
    final_img = v_utils.visualize_boxes_and_labels_on_image_array(image_np, 
                                                                  output_dict['detection_boxes'],
                                                                  output_dict['detection_classes'],
                                                                  output_dict['detection_scores'], 
                                                                  category_index,
                                                                  use_normalized_coordinates=True,
                                                                  line_thickness=8)
    return (final_img)



# cap = cv2.VideoCapture(0)
# while 1:
#     _,img = cap.read()
#     img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
#     final_img = show_inference(detection_model, img)
#     final_img = cv2.cvtColor(final_img,cv2.COLOR_RGB2BGR)
    
#     cv2.imshow('object detection',final_img)
    
#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


#img_path = [os.path.join(wd,"Test","IMG22.jpg"),
#            os.path.join(wd,"Test","IMG15.jpg"),
#            os.path.join(wd,"Test","IMG14.jpg")
            
#            ]
img_path = [] # add all your image path here Example : "E:\Train\IMG22.jpg"
            
i = 0
for p in img_path:
    i += 1
    img = cv2.imread(p)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    final_Img = show_inference(detection_model, img)
    final_Img = cv2.cvtColor(final_Img,cv2.COLOR_RGB2BGR)
    image_name = 'Saved_img_{}.jpg'.format(str(i));
    cv2.imwrite(image_name,final_Img)
    print(image_name,' saved sucessfully');


    


    
    
