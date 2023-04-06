import torchvision
from PIL import Image
import cv2
import numpy as np

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

img = cv2.imread('/Users/yangzheng/code/project/smoke/test_fluid_cam/train0/r_130.png')
mask = img.sum(axis=2) > 20

mask = mask * 255
# mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
mask = mask.astype(np.uint8)
mask = mask[:, :, np.newaxis]
# cv2.imshow('mask', mask)
# cv2.waitKey(0)
img = np.concatenate([img, mask], axis=2)
print(img.shape)
# write png
cv2.imwrite('/Users/yangzheng/code/project/smoke/fluid_debug/train/r_130_mask.png', img)