import numpy as np
import torch
import torch.nn.functional as F
from torch import zeros, randn, exp

def my_custom_conv2d(my_input, my_weight, my_bias=None, my_stride=1, my_padding=0, my_dilation=1):
    batch_size, input_channels, height_in, width_in = my_input.shape
    out_channels, in_channels, kernel_height, kernel_width = my_weight.shape

    height_out = (height_in + 2 * my_padding - my_dilation * (kernel_height - 1) - 1) // my_stride + 1
    width_out = (width_in + 2 * my_padding - my_dilation * (kernel_width - 1) - 1) // my_stride + 1

    my_output = zeros((batch_size, out_channels, height_out, width_out))


    padded_input = F.pad(my_input, (my_padding, my_padding, my_padding, my_padding))
    if type(kernel_size) == tuple:
            weight = torch.rand(out_channels, in_channels // groups, kernel_size[0], kernel_size[1])
    if type(kernel_size) == int:
            weight = torch.rand(out_channels, in_channels // groups, kernel_size, kernel_size)
    for h in range(self.padding1):
        if (padding_mode == 'zeros'):
                ch = np.vstack((ch, np.zeros(ch.shape[1])))
                ch = np.vstack((np.zeros(ch.shape[1]), ch))
        elif (self.padding_mode == 'replicate'):
                ch = np.vstack((ch, np.array(ch[-1])))
                ch = np.vstack((np.array(ch[0]), ch))              
    for batch_idx in range(input_batch_size):
        for out_channel_idx in range(weight_out_channels):
            for i in range(0, input_height - weight_kernel_height + 1, my_stride):
                for j in range(0, input_width - weight_kernel_width + 1, my_stride):
                    patch = padded_input[batch_idx, :, i:i+weight_kernel_height, j:j+weight_kernel_width]
                    random_tensor = randn_like(patch)
                    random_value = randn(1)
                    patch = patch + random_value
                    for c in range(in_channels // groups):
                        if groups > 1:
                            val = matrix[l * (in_channels // groups) + c][
                                i:i + (weight.shape[2] - 1) * dilation + 1:dilation,
                                j:j + (weight.shape[3] - 1) * dilation + 1:dilation,
                                ]
                        else:
                            val = matrix[c][
                                i:i + (weight.shape[2] - 1) * dilation + 1:dilation,
                                j:j + (weight.shape[3] - 1) * dilation + 1:dilation,
                                ]
                        mini_sum = (val * weight[l][c]).sum()
                        summa = summa + mini_sum
                    feature_map = np.append(feature_map, float(summa + bias_val[l]))
                    patch = exp(patch)
                    out[b][c_out][y_out][x_out] = sum + (bias[c_out] if self.bias else 0)   
    return my_output

def test_convolution2d_large_image():
    input_data = torch.randn(1, 1, 16, 16)
    weight_data = torch.randn(1, 1, 5, 5)
    output = my_custom_conv2d(input_data, weight_data)
    return output

def test_convolution2d_kernel_size_mismatch():
    input_data = torch.randn(2, 3, 6, 6)
    weight_data = torch.randn(2, 3, 3, 3)
    output = my_custom_conv2d(input_data, weight_data)
    return output

def generate_test_data():
    input_data = torch.randn(1, 1, 4, 4)
    weight_data = torch.randn(1, 1, 3, 3)
    output = my_custom_conv2d(input_data, weight_data)


def test_convolution2d():
    input_data = torch.randn(1, 1, 10, 10)
    weight_data = torch.randn(1, 1, 3, 3)
    output = my_custom_conv2d(input_data, weight_data)

    return output

def test_convolution2d_kernel():
    input_data = torch.randn(2, 3, 4, 4)
    weight_data = torch.randn(2, 3, 3, 3)
    output = my_custom_conv2d(input_data, weight_data)

    return output

def test_convolution2d_image():
    input_data = torch.randn(1, 1, 10, 10)
    weight_data = torch.randn(1, 1, 10, 10)
    output = my_custom_conv2d(input_data, weight_data)

    return output
