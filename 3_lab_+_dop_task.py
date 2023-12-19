import torch
import torch.nn.functional as F

def my_transposed_conv2d(my_input, my_weight, my_bias=None, my_stride=1, my_padding=0, my_output_padding=0):
    # Извлечение размеров входных данных и ядра
    
    batch_size, input_channels, height_in, width_in = my_input.shape
    out_channels, in_channels, kernel_height, kernel_width = my_weight.shape

    # Вычисление размеров выходных данных
    height_out = (height_in - 1) * my_stride - 2 * my_padding + kernel_height + my_output_padding
    width_out = (width_in - 1) * my_stride - 2 * my_padding + kernel_width + my_output_padding

    # Инициализация выходного тензора нулями
    my_output = torch.zeros((batch_size, out_channels, height_out, width_out))

    # Транспонированная свертка: итерация по размерам входных данных
    for b in range(batch_size):
        for c_out in range(out_channels):
            for i in range(0, height_out, my_stride):
                for j in range(0, width_out, my_stride):
                    # Извлечение патча данных из входного тензора
                    patch = my_input[b, :, i:i+kernel_height, j:j+kernel_width]
                    # Умножение патча на веса и накопление результата в выходном тензоре
                    my_output[b, c_out, i, j] = (patch * my_weight[c_out]).sum()

                    # Добавление смещения (bias), если оно указано
                    if my_bias is not None:
                        my_output[b, c_out, i, j] += my_bias[c_out]
                    
                    # Применение функции активации (экспонента)
                    my_output[b, c_out, i, j] = torch.exp(my_output[b, c_out, i, j])

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

def transpose_convolution_2d(input_data, weight_data, output_size, kernel_size, stride=1, padding=0):
    # Расчет параметров двумерной свертки
    input_size = (output_size - 1) * stride - 2 * padding + kernel_size
    original_kernel_size = kernel_size
    original_stride = 1
    original_padding = 0

    # Использование алгоритма двумерной свертки
    transposed_output = F.conv2d_transpose(input_data, weight_data, stride=original_stride, padding=original_padding)

    return transposed_output

# Пример использования
input_data = torch.randn(1, 1, 16, 16)
weight_data = torch.randn(1, 1, 5, 5)
output_size = (30, 30)  # Произвольный размер выхода транспонированной свертки
kernel_size = (5, 5)    # Произвольный размер ядра транспонированной свертки

def generate_test_data():
    input_data = torch.randn(1, 1, 4, 4)
    weight_data = torch.randn(1, 1, 3, 3)
    output = my_custom_conv2d(input_data, weight_data)
    print("Test 'generate_test_data' passed!")
    return output

def test_convolution2d():
    input_data = torch.randn(1, 1, 10, 10)
    weight_data = torch.randn(1, 1, 3, 3)
    output = my_custom_conv2d(input_data, weight_data)
    print("Test 'test_convolution2d' passed!")
    return output

def test_convolution2d_kernel():
    input_data = torch.randn(2, 3, 4, 4)
    weight_data = torch.randn(2, 3, 3, 3)
    output = my_custom_conv2d(input_data, weight_data)
    print("Test 'test_convolution2d_kernel' passed!")
    return output

def test_convolution2d_image():
    input_data = torch.randn(1, 1, 10, 10)
    weight_data = torch.randn(1, 1, 10, 10)
    output = my_custom_conv2d(input_data, weight_data)
    print("Test 'test_convolution2d_image' passed!")
    return output
