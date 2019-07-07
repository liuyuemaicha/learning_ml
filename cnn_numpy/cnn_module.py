# coding:utf8
import numpy as np
import sys


def conv_(img, conv_filter):
    filter_size = conv_filter.shape[1]
    result = np.zeros((img.shape))
    # Looping through the image to apply the convolution operation.
    # convert float data in to int
    for r in np.arange(filter_size / 2.0, img.shape[0] - filter_size / 2.0 + 1):
        for c in np.arange(filter_size / 2.0, img.shape[1] - filter_size / 2.0 + 1):
            """
            Getting the current region to get multiplied with the filter.
            How to loop through the image and get the region based on 
            the image and filer sizes is the most tricky part of convolution.
            """
            r, c = int(r), int(c)
            r_left = int(r - np.floor(filter_size / 2.0))
            r_right = int(r + np.ceil(filter_size / 2.0))
            c_left = int(c - np.floor(filter_size / 2.0))
            c_right = int(c + np.ceil(filter_size / 2.0))
            curr_region = img[r_left:r_right, c_left:c_right] # index must be int
            # Element-wise multipliplication between the current region and the filter.
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result)  # Summing the result of multiplication.
            result[r, c] = conv_sum  # Saving the summation in the convolution layer feature map.

    # Clipping the outliers of the result matrix.
    final_result = result[int(filter_size / 2.0):result.shape[0] - int(filter_size / 2.0),
                   int(filter_size / 2.0):result.shape[1] - int(filter_size / 2.0)]
    return final_result

#
# def conv_(img, conv_filter):
#     filter_size = conv_filter.shape[1]
#     result = np.zeros((img.shape))
#     # Looping through the image to apply the convolution operation.
#
#     for r in int(np.arange(filter_size / 2.0, img.shape[0] - filter_size / 2.0 + 1)):
#         for c in int(np.arange(filter_size / 2.0, img.shape[1] - filter_size / 2.0 + 1)):
#             """
#             Getting the current region to get multiplied with the filter.
#             How to loop through the image and get the region based on
#             the image and filer sizes is the most tricky part of convolution.
#             """
#             print r, c
#
#             curr_region = img[r - int(np.floor(filter_size / 2.0)):r + int(
#                 np.ceil(filter_size / 2.0)),
#                           c - int(np.floor(filter_size / 2.0)):c + int(
#                               np.ceil(filter_size / 2.0))]
#             # Element-wise multipliplication between the current region and the filter.
#             curr_result = curr_region * conv_filter
#             conv_sum = np.sum(curr_result)  # Summing the result of multiplication.
#             result[r, c] = conv_sum  # Saving the summation in the convolution layer feature map.
#
#     # Clipping the outliers of the result matrix.
#     final_result = result[int(filter_size / 2.0):result.shape[0] - int(filter_size / 2.0),
#                    int(filter_size / 2.0):result.shape[1] - int(filter_size / 2.0)]
#     return final_result

def conv(img, conv_filter):
    """
    :param img:  w,h, channel
    :param conv_filter: num, w, h, channel
    :return:
    """
    if len(img.shape) != 3 or len(conv_filter.shape) != 4:
        print("Error Shape Len: Need img 3, filter 4. input img: {0}, filter: {1}".format(img.shape, conv_filter.shape))
        sys.exit()
    if img.shape[-1] != conv_filter.shape[-1]:  # Check if number of image channels matches the filter depth.
        print("Error: Number of channels in both image and filter must match.")
        sys.exit()
    if conv_filter.shape[1] != conv_filter.shape[2]:  # Check if filter dimensions are equal.
        print('Error: Filter must be a square matrix. I.e. number of rows and columns must match.')
        sys.exit()
    if conv_filter.shape[1] % 2 == 0:  # Check if filter diemnsions are odd.
        print('Error: Filter must have an odd size. I.e. number of rows and columns must be odd.')
        sys.exit()

    # An empty feature map to hold the output of convolving the filter(s) with the image.
    feature_maps = np.zeros((img.shape[0]-conv_filter.shape[1]+1,
                                img.shape[1]-conv_filter.shape[1]+1,
                                conv_filter.shape[0]))

    # Convolving the image by the filter(s).
    for filter_num in range(conv_filter.shape[0]):
        print("Filter ", filter_num + 1)
        curr_filter = conv_filter[filter_num, :]  # getting a filter from the bank.
        """ 
        Checking if there are mutliple channels for the single filter.
        If so, then each channel will convolve the image.
        The result of all convolutions are summed to return a single feature map.
        """
        conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0])  # Array holding the sum of all feature maps.
        for ch_num in range(1, curr_filter.shape[-1]): # Convolving each channel with the image and summing the results.
            conv_map += conv_(img[:, :, ch_num], curr_filter[:, :, ch_num])
        feature_maps[:, :, filter_num] = conv_map  # Holding feature map with the current filter.
    return feature_maps  # Returning all feature maps.


def max_pooling(feature_map, size=2, stride=2):
    #Preparing the output of the pooling operation.
    w, h, num = feature_map.shape
    pool_out = np.zeros(((w-size+1)/stride+1, (h-size+1)/stride+1, num))
    for map_num in range(num):
        r2 = 0
        for r in np.arange(0, w-size+1, stride):
            c2 = 0
            for c in np.arange(0, h-size+1, stride):
                pool_out[r2, c2, map_num] = np.max([feature_map[r:r+size,  c:c+size, map_num]])
                c2 = c2 + 1
            r2 = r2 + 1
    return pool_out


def relu(feature_map):
    #Preparing the output of the ReLU activation function.
    return np.maximum(feature_map, 0)
    # relu_out = np.zeros(feature_map.shape)
    # for map_num in range(feature_map.shape[-1]):
    #     for r in np.arange(0,feature_map.shape[0]):
    #         for c in np.arange(0, feature_map.shape[1]):
    #             relu_out[r, c, map_num] = np.max([feature_map[r, c, map_num], 0])
    # return relu_out


if __name__ == '__main__':
    feature_map = np.random.randn(3, 4, 2)
    print feature_map
    # print relu(feature_map)
    print max_pooling(feature_map)
