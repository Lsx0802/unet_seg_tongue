import numpy as np
import os
import json
from PIL import Image
from shutil import copy
import random
import matplotlib.pyplot as plt

save=r'C:\Users\hello\PycharmProjects\UNet2D_tongue_2\dataset'
path=r'C:\Users\hello\PycharmProjects\UNet2D_tongue_2\original_data'
tongue_image = os.path.join(path,'image')
mask_image=os.path.join(path,'label')

tongue_image1=os.listdir(tongue_image)
mask_image1=os.listdir(mask_image)

if len(tongue_image1)!=len(mask_image1):
    print('图片和mask数量不一致，请核查')

random.seed(2021)
random.shuffle(tongue_image1)
random.seed(2021)
random.shuffle(mask_image1)


train_tongue_image=tongue_image1[0:int(len(tongue_image1)*0.8)]
train_mask_image=mask_image1[0:int(len(mask_image1)*0.8)]
val_tongue_image=tongue_image1[int(len(tongue_image1)*0.8):]
val_mask_image=mask_image1[int(len(mask_image1)*0.8):]

# for i in train_mask_image:
#     path2=os.path.join(mask_image,i)
#     mask=Image.open(path2)
#     mask=np.array(mask)
#     print(mask.shape)
#     for j in range(mask.shape[0]):
#         for k in range(mask.shape[1]):
#             if mask[j][k]!=0:
#                 mask[j][k]=255
#     mask=Image.fromarray(mask)
#     save2=os.path.join(save,'train_masks')
#     if not os.path.exists(save2):
#         os.makedirs(save2)
#     save3=save2+'/'+i[:-4]+'.png'
#     mask.save(save3)
#
# for i in val_mask_image:
#     path2 = os.path.join(mask_image, i)
#     mask = Image.open(path2).convert('L')
#     mask = np.array(mask)
#     print(mask.shape)
#     for j in range(mask.shape[0]):
#         for k in range(mask.shape[1]):
#             if mask[j][k] != 0:
#                 print(mask[j][k])
#                 mask[j][k] = 255
#     # mask = Image.fromarray(mask)
#     # save2 = os.path.join(save, 'val_masks')
#     # if not os.path.exists(save2):
#     #     os.makedirs(save2)
#     # save3 = save2 + '/' + i[:-4] + '.png'
#     # mask.save(save3)
#
# for i in train_tongue_image:
#     path2 = os.path.join(tongue_image, i)
#     mask = Image.open(path2)
#     save2 = os.path.join(save, 'train_images')
#     if not os.path.exists(save2):
#         os.makedirs(save2)
#     save3 = save2 + '/' + i[:-4] + '.png'
#     mask.save(save3)
#
# for i in val_tongue_image:
#     path2 = os.path.join(tongue_image, i)
#     mask = Image.open(path2)
#     save2 = os.path.join(save, 'val_images')
#     if not os.path.exists(save2):
#         os.makedirs(save2)
#     save3 = save2 + '/' + i[:-4] + '.png'
#     mask.save(save3)
