import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

# 定义矩形边界
xmin, xmax = 0, 10
ymin, ymax = 0, 10

# 在矩形区域中生成随机点
original_points = np.random.rand(10, 2) * [xmax - xmin, ymax - ymin] + [xmin, ymin]

# 定义反射函数：将点反射到矩形边界外
def reflect_point(point, boundary):
    x, y = point
    xmin, xmax, ymin, ymax = boundary
    
    if x < xmin:
        x_new = xmin + (xmin - x)
    elif x > xmax:
        x_new = xmax - (x - xmax)
    else:
        x_new = x
        
    if y < ymin:
        y_new = ymin + (ymin - y)
    elif y > ymax:
        y_new = ymax - (y - ymax)
    else:
        y_new = y

    return np.array([x_new, y_new])

# 边界定义
boundary = (xmin, xmax, ymin, ymax)
border_points = []

# 生成对称点并追加到种子点列表
for point in original_points:
    reflected_point = reflect_point(point, boundary)
    border_points.append(reflected_point)

# 将 border_points 转换为 NumPy 数组
border_points = np.array(border_points)

# 将对称点添加到原始种子点中
extended_points = np.vstack([original_points, border_points])

# 执行 Voronoi 剖分
vor = Voronoi(extended_points)

# 可视化
xx = np.array([0.0, 10.0, 10.0, 0.0, 0.0])
yy = np.array([0.0, 0.0, 10.0, 10.0, 0.0])
fig, ax = plt.subplots()
ax.plot(xx, yy, 'r--')
voronoi_plot_2d(vor, ax=ax)
ax.plot(original_points[:, 0], original_points[:, 1], 'bo', label="Original Points")
ax.plot(border_points[:, 0], border_points[:, 1], 'ro', label="Reflected Points")
plt.xlim(xmin-5.0, xmax+5.0)
plt.ylim(ymin-5.0, ymax+5.0)
ax.set_aspect(1.0)
plt.legend()
plt.show()
