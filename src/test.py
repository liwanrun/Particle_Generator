import numpy as np
import matplotlib.pyplot as plt

# 设置图像的大小
n = 256  # 图像大小为 256x256
size = n

# 创建频域信号的幅度谱（控制轮廓的形状）
real = np.random.randn(size, size)  # 随机生成实部
imag = np.random.randn(size, size)  # 随机生成虚部

# 合成频域信号
freq_domain = real + 1j * imag

# 对频域信号应用逆 FFT，得到时域信号
time_domain = np.fft.ifft2(freq_domain)

# 获取时域信号的绝对值（对应不规则颗粒的形状）
particle_shape = np.abs(time_domain)

# 二值化操作：将低值部分设为 0（背景），高值部分设为 1（轮廓）
threshold = np.percentile(particle_shape, 90)  # 阈值为图像的 90% 最大值
binary_shape = particle_shape > threshold

# 显示生成的颗粒轮廓
plt.imshow(binary_shape, cmap='gray')
plt.title("Generated Particle Contour")
plt.colorbar()
plt.show()
