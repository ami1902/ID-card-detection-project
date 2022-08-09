import os
import sys
import pandas as pd
import glob
import xml.etree.ElementTree as ET
import io


import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util,label_map_util,config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from collections import namedtuple,OrderedDict

wd = os.getcwd();
custom_model_name = 'my_faster_rcnn';

#for local training
paths = {
        'pre_trained_model' : os.path.join(wd, "Pre-trained_model"),
        'train_img_path' : os.path.join(wd,"Train"),
        'test_img_path' : os.path.join(wd,"Test"),
        'protoc_path' : os.path.join(wd,"Protoc"),
        'annotation_path' : os.path.join(wd,'Annotations'),
        'final_model_path' : os.path.join(wd,"Trained_model"),
        'api_model_path' : os.path.join(wd,"API_model")
    };

# paths = {
#     'annotation_path' : os.path.join(wd,'Annotations'),
#     'final_model_path' : os.path.join(wd,"Trained_model"),
#     }

files = {
    'pipeline' : os.path.join(paths['final_model_path'],custom_model_name,'pipeline.config'),
    }

# will go through each of the paths in 'paths' dictonary
# and create a folder for it
def makeDir(): 
    for p in paths:
        path = paths[p]
        f_path = f'"{path}"';
        if not os.path.exists(path):
            print(f_path)
            if os.name == 'nt':
                os.system('mkdir '+f_path)
        
#creates a label_map.pbtxt file
# the structure of label map is 
# item { 
#   id : 1
#   name: 'id_card'
#}
# will help the feature extractor to collect the class name from label map
# in case the annotation file doesn't clearly define the class name   
def createLabelMap():
    label = {'name' : 'id_card','id':1}
    try:
        if not os.path.exists(paths["annotation_path"]+"\\label_map.pbtxt"):
            f = open(paths["annotation_path"]+"\\label_map.pbtxt","x");
            f.write("item {\n");
            f.write("\tid: {}\n".format(label['id']));
            f.write("\tname: \'{}\'\n".format(label['name']));
            f.write("}\n");
            f.close()
    except:
        print("file already created");        
    
# takes all the annotation xml file 
# converts them to pandas dataframe to later be used to convert to .csv

def convert_xml_to_DataFrame(path):
    # a list to hold the values
    x_list = [];
    for x_file in glob.glob(path+"\\*.xml"):
        # parse the xml file and get root
        tree = ET.parse(x_file);
        root = tree.getroot();
        # find a children called object which holds the class name
        for m in root.findall('object'):
            # collect all the required values from the xml file
            value = (root.find('filename').text,
            int(root.find('size')[0].text),
            int(root.find('size')[1].text),
            m[0].text,
            int(m[4][0].text),
            int(m[4][1].text),
            int(m[4][2].text),
            int(m[4][3].text))
            x_list.append(value)
    col_name = ['filename','width','height','class','xmin','ymin','xmax','ymax']
    # convert to pandas dataframe datastructure 
    # a 2d tabluar data with mutable size alogn with axes labeled 
    
    x_dataframe = pd.DataFrame(x_list,columns=col_name)
    return x_dataframe;

# create two csv from the dataframe one for train other for test
def convert_df_to_csv():
    dataFrame= convert_xml_to_DataFrame(paths["train_img_path"]);
    if not os.path.exists(paths["annotation_path"]+"\\id_card_labels_train.csv"):
        #conversion to csv file
        dataFrame.to_csv(paths["annotation_path"]+"\\id_card_labels_train.csv",index=None)
        print("Generated train csv sucessfully");
    else:
        print("Train CSV already generated")
    
    dataFrame= convert_xml_to_DataFrame(paths["test_img_path"]);
    if not os.path.exists(paths["annotation_path"]+"\\id_card_labels_test.csv"):
        #conversion to csv file
        dataFrame.to_csv(paths["annotation_path"]+"\\id_card_labels_test.csv",index=None)
        print("Generated test csv sucessfully");
    else:
        print("Test CSV already generated")

def split(df,group):
    # will give a new subclass of tuple with named feilds
    # and iterable dictonary along with key/value pair
    data = namedtuple('data', ['filename','object'])

    
    # groups the pandas data frame to filename
    g = df.groupby(group)
    # gives a list of classes with filename and single pandas dataframe row
    return [data(filename,g.get_group(x)) for filename,x in zip(g.groups.keys(),g.groups)]

# gives the id of the label 
def cls_text_to_int(class_name):
    label_map = label_map_util.load_labelmap(paths["annotation_path"]+"\\label_map.pbtxt")
    label_map_dictonary = label_map_util.get_label_map_dict(label_map);
    print(label_map_dictonary);
    return label_map_dictonary[class_name]
    

def tf_example(group,path):
    # reading the image content as byte literals
    with tf.io.gfile.GFile(os.path.join(path,'{}'.format(group.filename)),'rb') as f:
        encoded_jpg = f.read();
    # placing the bytes images in memory buffer
    enc_jpg_io = io.BytesIO(encoded_jpg)
    # opening the image from the buffer
    img = Image.open(enc_jpg_io)
    # getting the width and height of image 
    width,height = img.size
    # converting the jpeg name to bytes
    filename = group.filename.encode('utf8')
    # storing the image format as byte literal
    image_format = b'jpg';
    x_mins = []
    x_maxs = []
    y_mins = []
    y_maxs = []
    class_texts = []
    classes =[]
    
    for i,r in group.object.iterrows():
        x_mins.append(r['xmin']/width)
        x_maxs.append(r['xmax']/width)
        y_mins.append(r['ymin']/height)
        y_maxs.append(r['ymax']/height)
        class_texts.append(r['class'].encode('utf8'))
        classes.append(cls_text_to_int(r['class']))
    
    # a proto message tf.train.Example felxible proto message type
    # protobuffers are cross platform and language library for serialization of structured data
    # here basically tf.train.Example is a {string : tf.train.feature } mapping
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(x_mins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(x_maxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(y_mins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(y_maxs),
        'image/object/class/text': dataset_util.bytes_list_feature(class_texts),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        }));
    
    return example


def generateTFRecord():
    
    if not os.path.exists(paths['annotation_path']+"\\train.record"):
        w = tf.io.TFRecordWriter(paths['annotation_path']+"\\train.record");
        examples = pd.read_csv(paths['annotation_path']+"\\id_card_labels_train.csv");
        
        grouped = split(examples,'filename');
        
        for group in grouped:
           # gives the proto message
           example = tf_example(group,paths['train_img_path'])
           # converts the proto message to binary string
           w.write(example.SerializeToString())
       
        w.close()
        print("Sucessfully created train record at :", paths['annotation_path']+"\\train.record");
    
    if not os.path.exists(paths['annotation_path']+"\\test.record"):
        w = tf.io.TFRecordWriter(paths['annotation_path']+"\\test.record");
        examples = pd.read_csv(paths['annotation_path']+"\\id_card_labels_test.csv");
        
        grouped = split(examples,'filename');
        
        for group in grouped:
           # gives the proto message
           example = tf_example(group,paths['test_img_path'])
           # converts the proto message to binary string
           w.write(example.SerializeToString())
       
        w.close()
        print("Sucessfully created train record at :", paths['annotation_path']+"\\test.record");
    


# def update_pipline_config():
#     # config = config_util.get_configs_from_pipeline_file(files['pipeline']);
#     pipeline_config = pipeline_pb2.TrainEvalPipelineConfig();
    
#     with tf.io.gfile.GFile(files['pipeline'],"r") as f:
#         p_str = f.read()
#         text_format.Merge(p_str, pipeline_config);
    
#     pipeline_config.model.faster_rcnn.num_classes = 1;
#     pipeline_config.train_config.batch_size = 4;
#     #local training
#     # pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['pre_trained_model'],
#     #                                                                  'checkpoint',
#     #                                                                  'ckpt-0')
#     pipeline_config.train_config.fine_tune_checkpoint = "Pre-trained_model/checkpoint/cktp-0"
#     pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
#     #local training
#     # pipeline_config.train_input_reader.label_map_path = os.path.join(paths['annotation_path'],'labelmap.pbtxt')
#     # pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['annotation_path'],'train.record')];
#     # pipeline_config.eval_input_reader[0].label_map_path = os.path.join(paths['annotation_path'],'labelmap.pbtxt')
#     # pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['annotation_path'],'test.record')];
#     pipeline_config.train_input_reader.label_map_path = "Annotations/label_map.pbtxt"
#     pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = ["Annotations/train.record"]
#     pipeline_config.eval_input_reader[0].label_map_path = "Annotations/label_map.pbtxt"
#     pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = ["Annotations/test.record"]
    
#     cfg_text = text_format.MessageToString(pipeline_config);
   
#     with tf.io.gfile.GFile(files['pipeline'],"wb") as f:
#         f.write(cfg_text)
      
#     print("Config updated sucessfully");
    
    
    
def print_train_command():
    
    train_script = os.path.join(paths['api_model_path'],'research','object_detection','model_main_tf2.py');
    
    command = 'python \"{}\" --model_dir=\"{}\" --pipeline_config_path=\"{}\" --num_train_steps=500'.format(train_script,os.path.join(paths['final_model_path'],custom_model_name),files['pipeline'])
    print(command);


def print_eval_command():
    train_script = os.path.join(paths['api_model_path'],'research','object_detection','model_main_tf2.py');
    
    command = 'python \"{}\" --model_dir=\"{}\" --pipeline_config_path=\"{}\" --checkpoint_dir=\"{}\"'.format(train_script,os.path.join(paths['final_model_path'],custom_model_name),files['pipeline'],os.path.join(paths['final_model_path'],custom_model_name))
    print(command);

if __name__ == '__main__':
    makeDir();
    createLabelMap()
    convert_df_to_csv()
    generateTFRecord();
    print_train_command();
    print_eval_command();
    #update_pipline_config();

    