import glob
import gzip
import os
import tarfile
import time
import warnings
from urllib.request import urlretrieve

import pandas as pd

# import keras
# from keras.applications import DenseNet121, ResNet50
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.layers import Dense, Flatten
# from keras.metrics import AUC
# from keras.models import load_model, Model
# from keras.preprocessing import image
# from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

import os
import numpy as np
import torch
import pandas as pd

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score
from tqdm import tqdm_notebook

# import tensorflow as tf
# # tf.test.is_gpu_available()
# # Set CPU as available physical device
# my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
# tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

# # To find out which devices your operations and tensors are assigned to
# tf.debugging.set_log_device_placement(True)

from PIL import Image
from torch.autograd import Variable
import re

os.environ["CUDA_VISIBLE_DEVICES"]=""

# with tf.device('cpu:0'):
#     resnet = load_model('resnet-best_new.hdf5', 
#                     compile=False)

CLASSES = [
  'Hernia',
  'Pneumonia',
  'Fibrosis',
  'Edema',
  'Emphysema',
  'Cardiomegaly',
  'Pleural_Thickening',
  'Consolidation',
  'Pneumothorax',
  'Mass',
  'Nodule',
  'Atelectasis',
  'Effusion',
  'Infiltration'
]

classes_1 = [
        'Atelectasis',
        'Cardiomegaly',
        'Effusion',
        'Infiltration',
        'Mass',
        'Nodule',
        'Pneumonia',
        'Pneumothorax',
        'Consolidation',
        'Edema',
        'Emphysema',
        'Fibrosis',
        'Pleural_Thickening',
        'Hernia']


CKPT_PATH = 'ChexNetmodel.pth.tar'
N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']



class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


checkpoint = torch.load(CKPT_PATH,map_location='cpu')

pattern = re.compile(
    r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
state_dict = checkpoint['state_dict']

for key in list(state_dict.keys()):
  res = pattern.match(key)
  if res:
      new_key = res.group(1) + res.group(2)
      state_dict[new_key] = state_dict[key]
      del state_dict[key]

model = DenseNet121(N_CLASSES)#.cuda()
model = torch.nn.DataParallel(model)#.cuda()


model.load_state_dict(state_dict)


def ensemble(ID,model_to=model,classes_1=CLASSES,classes_2=classes_1) ->set: #model_tf=resnet,

    # img = tf.keras.preprocessing.image.load_img(
    #     ID, grayscale=False, color_mode='rgb', target_size=None,
    #     interpolation='nearest'
    #     )


    # image_array  = tf.keras.preprocessing.image.img_to_array(img)
    # image_array = tf.image.resize(image_array, [224,224])/255
    # image_tensor =tf.expand_dims(
    # image_array, 0, name=None)

    # answer_1 = model_tf(image_tensor)
    # answ={}
    # for cls,pred in (zip(CLASSES,range(len(CLASSES)))):
    #     answ[cls] = float(answer_1[:,pred])
    # max_val = max(answ.values())
    # d = []
    # for key,val in zip(answ.keys(),answ.values()):
    #     if val >= max_val*0.6:
    #         d.append(key)


    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])

    image = Image.open(ID)
    image = image.convert('RGB')

    transform=transforms.Compose([
                                        transforms.Resize(224),
                                        transforms.TenCrop(224),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ])

    inp = transform(image)
    b,c, h, w = inp.size()
    input_var = torch.autograd.Variable(inp.view(-1, c, h, w), volatile=True)
    output = model(input_var)
    answers = set(output.max(1)[1].data.numpy())
    answ=[]
    for i in list(answers):
        answ.append(classes_1[i])
    # labels = set(d).union(set(answ))
 
    return answ #labels


# def ensemble(ID,model_tf=resnet,model_to=model,classes_1=CLASSES,classes_2=classes_1) ->set:

#     # img = tf.keras.preprocessing.image.load_img(
#     #     ID, grayscale=False, color_mode='rgb', target_size=None,
#     #     interpolation='nearest'
#     #     )


#     # image_array  = tf.keras.preprocessing.image.img_to_array(img)
#     # image_array = tf.image.resize(image_array, [224,224])/255
#     # image_tensor =tf.expand_dims(
#     # image_array, 0, name=None)

#     # answer_1 = model_tf(image_tensor)
#     # answ={}
#     # for cls,pred in (zip(CLASSES,range(len(CLASSES)))):
#     #     answ[cls] = float(answer_1[:,pred])
#     # max_val = max(answ.values())
#     # d = []
#     # for key,val in zip(answ.keys(),answ.values()):
#     #     if val >= max_val*0.6:
#     #         d.append(key)


#     normalize = transforms.Normalize([0.485, 0.456, 0.406],
#                                     [0.229, 0.224, 0.225])

#     image = Image.open(ID)
#     image = image.convert('RGB')

#     transform=transforms.Compose([
#                                         transforms.Resize(224),
#                                         transforms.TenCrop(224),
#                                         transforms.Lambda
#                                         (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
#                                         transforms.Lambda
#                                         (lambda crops: torch.stack([normalize(crop) for crop in crops]))
#                                     ])

#     inp = transform(image)
#     b,c, h, w = inp.size()
#     input_var = torch.autograd.Variable(inp.view(-1, c, h, w), volatile=True)
#     output = model(input_var)
#     answers = set(output.max(1)[1].data.numpy())
#     answ=[]
#     for i in list(answers):
#         answ.append(classes_1[i])
#     # labels = set(d).union(set(answ))
 
#     return set(answ) #labels
      


# def predict(image_path):
#     CLASSES_1 = [
#     'Hernia',
#     'Pneumonia',
#     'Fibrosis',
#     'Edema',
#     'Emphysema',
#     'Cardiomegaly',
#     'Pleural_Thickening',
#     'Consolidation',
#     'Pneumothorax',
#     'Mass',
#     'Nodule',
#     'Atelectasis',
#     'Effusion',
#     'Infiltration']

#     image = tf.keras.preprocessing.image.load_img(
#         image_path, grayscale=False, color_mode='rgb', target_size=None,
#         interpolation='nearest'
#     )
#     image_array  = tf.keras.preprocessing.image.img_to_array(image)
#     r = tf.image.resize(image_array, [224,224])/255
#     rr =tf.expand_dims( r, 0, name=None )
    
#     answer_1 = resnet(rr)
#     answ={}
#     for cls,pred in (zip(CLASSES_1,range(len(CLASSES_1)))):
#         answ[cls] = float(answer_1[:,pred])
#     max_val = max(list(answ.values()))
#     labels  = [key for key,val in answ.items() if val >= max_val/2]
#     return labels

