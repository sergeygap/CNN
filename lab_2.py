import torch
import torch.nn.functional as F
from torch import zeros, randn, exp

# Функция свертки 3D
def my_custom_conv3d(my_input, my_weight, my_bias=None, my_stride=1, my_padding=0, my_dilation=1):
    # Извлечение размеров входных данных и ядра
    batch_size, input_channels, depth_in, height_in, width_in = my_input.shape
    out_channels, in_channels, kernel_depth, kernel_height, kernel_width = my_weight.shape

    # Вычисление размеров выходных данных
    depth_out = (depth_in + 2 * my_padding - my_dilation * (kernel_depth - 1) - 1) // my_stride + 1
    height_out = (height_in + 2 * my_padding - my_dilation * (kernel_height - 1) - 1) // my_stride + 1
    width_out = (width_in + 2 * my_padding - my_dilation * (kernel_width - 1) - 1) // my_stride + 1

    # Инициализация выходного тензора нулями
    my_output = zeros((batch_size, out_channels, depth_out, height_out, width_out))

    # Дополнение входных данных нулями (padding)
    padded_input = F.pad(my_input, (my_padding, my_padding, my_padding, my_padding, my_padding, my_padding))
    
    # Итерация по размерам выходных данных
    for batch_idx in range(batch_size):
        for out_channel_idx in range(out_channels):
            for d in range(0, depth_in - kernel_depth + 1, my_stride):
                for h in range(0, height_in - kernel_height + 1, my_stride):
                    for w in range(0, width_in - kernel_width + 1, my_stride):
                        # Извлечение патча данных
                        patch = padded_input[batch_idx, :, d:d+kernel_depth, h:h+kernel_height, w:w+kernel_width]
                        # Генерация случайных данных для патча
                        random_tensor = torch.randn_like(patch)
                        random_value = torch.randn(1)
                        patch = patch + random_value

                        # Свертка: умножение патча на веса и суммирование
                        mini_sum = (patch * my_weight[out_channel_idx]).sum()
                        # Накопление результата в выходном тензоре
                        my_output[batch_idx, out_channel_idx, d//my_stride, h//my_stride, w//my_stride] += mini_sum

                        # Добавление смещения (bias), если оно указано
                        if my_bias is not None:
                            my_output[batch_idx, out_channel_idx, d//my_stride, h//my_stride, w//my_stride] += my_bias[out_channel_idx]

                        # Применение функции активации (экспонента)
                        my_output[batch_idx, out_channel_idx, d//my_stride, h//my_stride, w//my_stride] = exp(my_output[batch_idx, out_channel_idx, d//my_stride, h//my_stride, w//my_stride])

    return my_output

# Тест 1: Проверка свертки на небольших данных
def test_convolution3d_small_data():
    input_data = torch.randn(1, 1, 4, 4, 4)
    weight_data = torch.randn(1, 1, 3, 3, 3)
    output = my_custom_conv3d(input_data, weight_data)
    assert output.shape == torch.Size([1, 1, 2, 2, 2]), f"Unexpected output shape: {output.shape}"

# Тест 2: Проверка свертки с различными размерами ядра и данных
def test_convolution3d_variable_sizes():
    input_data = torch.randn(2, 3, 6, 6, 6)
    weight_data = torch.randn(2, 3, 3, 3, 3)
    output = my_custom_conv3d(input_data, weight_data)
    assert output.shape == torch.Size([2, 2, 4, 4, 4]), f"Unexpected output shape: {output.shape}"

# Функция для генерации тестовых данных
def generate_test_data_3d():
    input_data = torch.randn(1, 1, 4, 4, 4)
    weight_data = torch.randn(1, 1, 3, 3, 3)
    output = my_custom_conv3d(input_data, weight_data)

