import os
from PIL import Image
from shutil import copy

img_path=r'H:\舌诊图片数据备份（北京数据2021-09-07）\新建文件夹\IMG 2014-8-25至2014-11-29'
patient=os.listdir(img_path)

save_path=r'H:\舌诊图片数据备份（北京数据2021-09-07）\新建文件夹\1113tongue'
if not os.path.exists(save_path):
    os.makedirs(save_path)

for i in patient:
    path_1=os.path.join(img_path,i)
    patient1=os.listdir(path_1)
    if len(patient1)<2:
        break
    for j in range(len(patient1)):

        path_11=os.path.join(path_1,patient1[0])
        path_22=os.path.join(save_path,patient1[0])

        img=Image.open(path_11)
        img.save(path_22)
