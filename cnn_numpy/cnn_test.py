# coding: utf8
from PIL import Image
import numpy as np
import cnn_module

img = Image.open('test.jpg')
# img = img.convert('L')
img_data = np.asarray(img)
# img_data = np.expand_dims(img_data, axis=2)
print img_data.shape

l1_filter = np.zeros((2, 3, 3, 3))
filter_kernel_1 = np.array([[[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]]])
filter_kernel_2 = np.array([[[1,   1,  1],
                             [0,   0,  0],
                             [-1, -1, -1]]])
l1_filter[0, :, :, 0] = filter_kernel_1
l1_filter[0, :, :, 1] = filter_kernel_1
l1_filter[0, :, :, 2] = filter_kernel_1

l1_filter[1, :, :, 0] = filter_kernel_2
l1_filter[1, :, :, 1] = filter_kernel_2
l1_filter[1, :, :, 2] = filter_kernel_2

print("\n**Working with conv layer 1**")
l1_feature_map = cnn_module.conv(img_data, l1_filter)
print l1_feature_map.shape
for index in range(l1_feature_map.shape[-1]):
    feature_map = Image.fromarray(l1_feature_map[:, :, index])
    feature_map = feature_map.convert('RGB')
    feature_map.save('conv_index_{0}.jpg'.format(index))

print("\n**ReLU**")
l1_feature_map_relu = cnn_module.relu(l1_feature_map)
print l1_feature_map_relu.shape
for index in range(l1_feature_map_relu.shape[-1]):
    feature_map = Image.fromarray(l1_feature_map_relu[:, :, index])
    feature_map = feature_map.convert('RGB')
    feature_map.save('relu_index_{0}.jpg'.format(index))

print("\n**Pooling**")
l1_feature_map_relu_pool = cnn_module.max_pooling(l1_feature_map_relu, 2, 2)
print l1_feature_map_relu_pool.shape
for index in range(l1_feature_map_relu_pool.shape[-1]):
    feature_map = Image.fromarray(l1_feature_map_relu_pool[:, :, index])
    feature_map = feature_map.convert('RGB')
    feature_map.save('max_pooling_index_{0}.jpg'.format(index))
print("**End of conv layer 1**\n")


