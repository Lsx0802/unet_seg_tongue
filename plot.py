# 将分割图和原图合在一起
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob

from skimage import measure,data,color



# image1 原图
# image2 mask图
# image3 pred图
save_path=r'C:\Users\hello\PycharmProjects\UNet2D_tongue_2\save_image_test'
if not os.path.exists(save_path):
    os.makedirs(save_path)
original_path=r'C:\Users\hello\PycharmProjects\UNet2D_tongue_2\data'
predict_path=r'C:\Users\hello\PycharmProjects\UNet2D_tongue_2\output/pred/LSX'

test_image_path=os.path.join(original_path,'test_images')
test_masks_path=os.path.join(original_path,'test_masks')

image=os.listdir(test_image_path)
mask=os.listdir(test_masks_path)
predict=os.listdir(predict_path)



for i in range(len(image)):
    image1 = Image.open(os.path.join(test_image_path,image[i]))
    image2 = Image.open(os.path.join(test_masks_path,mask[i]))
    image3 = Image.open(os.path.join(predict_path,predict[i]))

    image3 = image3.resize((image1.size[0], image1.size[1]), Image.ANTIALIAS)

    plt.figure()

    f, ax = plt.subplots(2, 3)
    # 设置主标题
    f.suptitle('Tongue segmentation')
    # 设置子标题
    ax[0][0].set_title('Original')
    ax[0][1].set_title('Mask')
    ax[0][2].set_title('Predict')
    ax[1][0].set_title('Original+Mask')
    ax[1][1].set_title('Original+Predict')
    ax[1][2].set_title('Mask+Predict')

    plt.subplot(231)
    plt.imshow(image1)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(232)
    plt.imshow(image2)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(233)
    plt.imshow(image3)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(234)
    image12 = cv2.addWeighted(np.array(image1.convert('RGB')), 0.65, np.array(image2.convert('RGB')), 0.35, 0.9)
    plt.imshow(image12)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(235)
    image13 = cv2.addWeighted(np.array(image1.convert('RGB')), 0.65, np.array(image3.convert('RGB')), 0.35, 0.9)
    plt.imshow(image13)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(236)
    image23 = cv2.addWeighted(np.array(image2.convert('RGB')), 0.65, np.array(image3.convert('RGB')), 0.35, 0.9)
    plt.imshow(image23)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    plt.axis('off')
    plt.xticks([])
    plt.yticks([])

    # plt.show()
    plt.savefig(os.path.join(save_path,image[i]))