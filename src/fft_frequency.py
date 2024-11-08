## Extract particle contours from digital image
import numpy as np
import cv2


class ParticleImageFFT:
    '''Fourier analysis for particle image'''
    def __init__(self, path:str) -> None:
        self.path = path

    def sample_contour(self, num_points=128) -> np.array:
        # 读取图像并转换为灰度图像
        image = cv2.imread(self.path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用Canny边缘检测
        edges = cv2.Canny(gray, 100, 200)

        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 假设只处理第一个轮廓
        contour = contours[0]

        # 计算轮廓的几何中心
        M = cv2.moments(contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            center_x, center_y = 0, 0

        # 均匀采样128个点
        indices = np.linspace(0, len(contour) - 1, num_points, dtype=int)
        sampled_points = contour[indices].reshape(-1, 2)

        # 将几何中心移动到原点
        translated_points = sampled_points - np.array([[center_x, center_y]])
        return translated_points

    def fourier_analysis(self) -> np.array:
        pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from particle import PolygonParticle
    num_freqs = 128
    image_path = r'img/real_05.jpg'
    FFTer = ParticleImageFFT(image_path)
    contour = FFTer.sample_contour(num_freqs)
    complex_signal = contour[:, 0] + 1j*contour[:, 1]
    points = np.vstack((complex_signal.real, complex_signal.imag)).T
    particle = PolygonParticle(points)
    particle.moveTo(np.array([0.0, 0.0]))
    print(particle.calc_shape_indexes())

    complex_spectrum = np.fft.fft(complex_signal)
    random_phases = np.exp(1j * np.random.uniform(0.0, np.pi, num_freqs))
    random_contour = np.fft.ifft(num_freqs * complex_spectrum * random_phases)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), layout='constrained')
    # ax[0].plot(contour[:, 0], contour[:, 1])
    ax[0].set_aspect(1.0)
    ax[0].set_box_aspect(1.0)
    particle.render(ax[0], color='cyan', add_bbox=False)
    ax[1].bar(np.fft.fftfreq(128, 1.0/128), np.abs(complex_spectrum))
    ax[1].set_box_aspect(1.0)
    ax[2].plot(random_contour.real, random_contour.imag, 'k-', lw=0.5)
    ax[2].set_aspect(1.0)
    ax[2].set_box_aspect(1.0)
    fig.savefig('original_partical.svg', transparent=True)
    plt.show()
