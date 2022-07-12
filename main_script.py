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
paths = {
        'pre_trained_model' : os.path.join(wd, "Pre-trained_model"),
        'train_img_path' : os.path.join(wd,"Train"),
        'test_img_path' : os.path.join(wd,"Test"),
        'protoc_path' : os.path.join(wd,"Protoc"),
        'annotation_path' : os.path.join(wd,'Annotations'),
        'final_model_path' : os.path.join(wd,"Trained_model"),
        'api_model_path' : os.path.join(wd,"API_model")
    };



files = {
    'pipeline' : os.path.join(paths['final_model_path'],custom_model_name,'pipeline.config'),
    }

def makeDir(): 
    for p in paths:
        path = paths[p]
        f_path = f'"{path}"';
        if not os.path.exists(path):
            print(f_path)
            if os.name == 'nt':
                os.system('mkdir '+f_path)
        
    
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
    

def convert_xml_to_DataFrame(path):
    x_list = [];
    for x_file in glob.glob(path+"\\*.xml"):
        tree = ET.parse(x_file);
        root = tree.getroot();
        for m in root.findall('object'):
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
    x_dataframe = pd.DataFrame(x_list,columns=col_name)
    return x_dataframe;

def convert_df_to_csv():
    dataFrame= convert_xml_to_DataFrame(paths["train_img_path"]);
    if not os.path.exists(paths["annotation_path"]+"\\id_card_labels_train.csv"):
        dataFrame.to_csv(paths["annotation_path"]+"\\id_card_labels_train.csv",index=None)
        print("Generated train csv sucessfully");
    else:
        print("Train CSV already generated")
    
    dataFrame= convert_xml_to_DataFrame(paths["test_img_path"]);
    if not os.path.exists(paths["annotation_path"]+"\\id_card_labels_test.csv"):
        dataFrame.to_csv(paths["annotation_path"]+"\\id_card_labels_test.csv",index=None)
        print("Generated test csv sucessfully");
    else:
        print("Test CSV already generated")

def split(df,group):
    data = namedtuple('data', ['filename','object'])
    g = df.groupby(group)
    return [data(filename,g.get_group(x)) for filename,x in zip(g.groups.keys(),g.groups)]

def cls_text_to_int(class_name):
    label_map = label_map_util.load_labelmap(paths["annotation_path"]+"\\label_map.pbtxt")
    label_map_dictonary = label_map_util.get_label_map_dict(label_map);
    print(label_map_dictonary);
    return label_map_dictonary[class_name]
    

def tf_example(group,path):
    with tf.io.gfile.GFile(os.path.join(path,'{}'.format(group.filename)),'rb') as f:
        encoded_jpg = f.read();
    
    enc_jpg_io = io.BytesIO(encoded_jpg)
    img = Image.open(enc_jpg_io)
    width,height = img.size
    filename = group.filename.encode('utf8')
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
           example = tf_example(group,paths['train_img_path'])
           w.write(example.SerializeToString())
       
        w.close()
        print("Sucessfully created train record at :", paths['annotation_path']+"\\train.record");
    
    if not os.path.exists(paths['annotation_path']+"\\test.record"):
        w = tf.io.TFRecordWriter(paths['annotation_path']+"\\test.record");
        examples = pd.read_csv(paths['annotation_path']+"\\id_card_labels_test.csv");
        
        grouped = split(examples,'filename');
        
        for group in grouped:
           example = tf_example(group,paths['test_img_path'])
           w.write(example.SerializeToString())
       
        w.close()
        print("Sucessfully created train record at :", paths['annotation_path']+"\\test.record");
    


def update_pipline_config():
    # config = config_util.get_configs_from_pipeline_file(files['pipeline']);
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig();
    
    with tf.io.gfile.GFile(files['pipeline'],"r") as f:
        p_str = f.read()
        text_format.Merge(p_str, pipeline_config);
    
    pipeline_config.model.faster_rcnn.num_classes = 1;
    pipeline_config.train_config.batch_size = 24;
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['pre_trained_model'],
                                                                     'checkpoint',
                                                                     'ckpt-0')
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path = os.path.join(paths['annotation_path'],'labelmap.pbtxt')
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['annotation_path'],'train.record')];
    pipeline_config.eval_input_reader[0].label_map_path = os.path.join(paths['annotation_path'],'labelmap.pbtxt')
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['annotation_path'],'test.record')];
    
    cfg_text = text_format.MessageToString(pipeline_config);
   
    with tf.io.gfile.GFile(files['pipeline'],"wb") as f:
        f.write(cfg_text)
      
    print("Config updated sucessfully");
    
    
def print_train_command():
    
    train_script = os.path.join(paths['api_model_path'],'research','object_detection','model_main_tf2.py');
    
    command = 'python \"{}\" --model_dir=\"{}\" --pipeline_config_path=\"{}\" --num_train_steps=500'.format(train_script,os.path.join(paths['final_model_path'],custom_model_name),files['pipeline'])
    print(command);


def print_eval_command():
    train_script = os.path.join(paths['api_model_path'],'research','object_detection','model_main_tf2.py');
    
    command = 'python \"{}\" --model_dir=\"{}\" --pipeline_config_path=\"{}\" --checkpoint_dir=\"{}\"'.format(train_script,os.path.join(paths['final_model_path'],custom_model_name),files['pipeline'],os.path.join(paths['final_model_path'],custom_model_name))
    print(command);

if __name__ == '__main__':
    # print_train_command();
    # print_eval_command();
    # makeDir();
    createLabelMap()
    convert_df_to_csv()
    generateTFRecord();
    #update_pipline_config();

    