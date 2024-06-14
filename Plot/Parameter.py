import matplotlib.pyplot as plt

# 数据
labels = ['GCL-YOLO-S', 'YOLOv7-Tiny', 'YOLOv8-S', 'PicoDet-L', 'YOLOX-Tiny',
          'YOLOv5-S', 'Nanodet-Plus-M-1.5x', 'YOLOv5-N', 'YOLOv4-Tiny',
          'YOLOv3-Tiny', 'YOLOv5-Lite-G', 'DS3FN']

params = [1.64, 6.03, 11.10, 3.30, 5.04, 7.04, 2.44, 1.77, 5.89, 8.68, 5.39, 2.77]
map_50 = [39.6, 36.8, 39.2, 34.2, 31.3, 32.7, 30.4, 26.4, 19.5, 15.9, 27.3, 45.5]

colors = ['red', 'gray', 'yellow', 'purple', 'green', 'blue',
          'orange', 'pink', 'brown', 'gold', 'cyan', 'magenta']
marker = '*'

# 绘制散点图，所有散点均为星星形状，颜色根据原有颜色列表
for i in range(len(labels)):
    plt.scatter(params[i], map_50[i], s=100, c=colors[i], marker=marker)

# 添加标签
for i in range(len(labels)):
    plt.annotate(labels[i], xy=(params[i]+0.1, map_50[i]-0.1), fontsize=10)

# 设置坐标轴标签和标题
plt.xlabel('Params (M)')
plt.ylabel('mAP50 (%)')
plt.title('Scatter Plot with Star Markers')

# 保存图形到本地
plt.savefig("scatter_plot_star_markers.png", dpi=300)

# 显示图表
plt.show()