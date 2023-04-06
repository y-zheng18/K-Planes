import torchvision
from PIL import Image
import cv2
import numpy as np
import os

# img = Image.open('/Users/yangzheng/code/project/smoke/test_fluid_cam/train/r_130.png')
# pil2tensor = torchvision.transforms.ToTensor()
#
#
# img = pil2tensor(img)
# img = img.permute(1, 2, 0)
#
# img = img.numpy()
# print(img.shape)
# img = img[:, :, -1]
# img = img > 0
# img = img * 255
# print(img.shape)
# img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
# img = img.astype(np.uint8)
# cv2.imshow('img', img)
# cv2.waitKey(0)
save_dir = '/Users/yangzheng/code/project/smoke/fluid_debug/train_'
os.makedirs(save_dir, exist_ok=True)
for img_f in os.listdir('/Users/yangzheng/code/project/smoke/fluid_debug/train'):
    if img_f.endswith('.png'):
        img = cv2.imread(os.path.join('/Users/yangzheng/code/project/smoke/fluid_debug/train', img_f))
        mask = img.mean(axis=2)

        mask = mask.astype(np.uint8)
        mask = mask[:, :, np.newaxis]
        # cv2.imshow('mask', mask)
        # cv2.waitKey(0)
        img = np.concatenate([img, mask], axis=2)
        print(img.shape)
        # write png
        cv2.imwrite(os.path.join(save_dir, img_f), img)

save_dir = '/Users/yangzheng/code/project/smoke/fluid_debug/test_'
os.makedirs(save_dir, exist_ok=True)
for img_f in os.listdir('/Users/yangzheng/code/project/smoke/fluid_debug/test'):
    if img_f.endswith('.png'):
        img = cv2.imread(os.path.join('/Users/yangzheng/code/project/smoke/fluid_debug/test', img_f))
        mask = img.mean(axis=2)

        mask = mask.astype(np.uint8)
        mask = mask[:, :, np.newaxis]
        # cv2.imshow('mask', mask)
        # cv2.waitKey(0)
        img = np.concatenate([img, mask], axis=2)
        print(img.shape)
        # write png
        cv2.imwrite(os.path.join(save_dir, img_f), img)