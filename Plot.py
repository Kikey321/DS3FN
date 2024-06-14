import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# 数据集路径及类别总数
data_folder = '/root/autodl-fs/mydata/DIOR/labels/train'
num_categories = 20
# # 定义马卡龙色系
# macaron_colors = ['#F0C373', '#5CA0A3', '#E69F00', '#D55E00', '#0072B2', '#CC79A7', '#009E73', '#F0E442', '#BF5B17', '#999999']
# 扩展马卡龙色系至20种颜色
macaron_colors = ['#F0C373', '#5CA0A3', '#E69F00', '#D55E00', '#0072B2', '#CC79A7', '#009E73', '#F0E442',
                  '#BF5B17', '#999999', '#800000', '#FF8C00', '#808000', '#008000', '#0000FF', '#800080',
                  '#FF0000', '#FFFF00', '#00FFFF', '#000080']


# 自定义类别名称列表（如果有的话，如果没有则保持类别编号作为名称）
# custom_category_names = ['Pedestrian', 'People', 'Bicycle', 'Car', 'Van', 'Truck', 'Tricycle', 'Awning-Tricycle', 'Bus', 'Motor']  # 示例列表
custom_category_names = ['Airplane', 'Airport', 'Baseballfield', 'Basketballcourt', 'Bridge', 'Chimney', 'Dam', 'Expressway-Service-area', 'Expressway-toll-station', 'Golffield',
                         'Groundtrackfield', 'Harbor', 'Overpass', 'Ship', 'Stadium', 'Storagetank', 'Tenniscourt', 'Trainstation', 'Vehicle', 'Windmill']  # 示例列表

# 初始化类别计数器
category_counts = [0] * num_categories

# 遍历文件夹中的所有txt文件
txt_files = glob.glob(os.path.join(data_folder, '*.txt'))
for txt_file in txt_files:
    with open(txt_file, 'r') as f:
        for line in f:
            # 解析YOLO格式的每行数据
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    category = int(parts[0])  # 获取类别ID
                    if 0 <= category < num_categories:
                        category_counts[category] += 1
                except ValueError:
                    pass

# 创建颜色列表，每个类别对应一种颜色
colors = macaron_colors

# 绘制直方图
# fig, ax = plt.subplots(figsize=(10, 6))
# fig, ax = plt.subplots(figsize=(12, 6))  # 调整figure大小以适应更多类别
fig, ax = plt.subplots(figsize=(25, 18))  # 调整figure大小以适应更多类别


bar_width = 0.5
bar_positions = np.arange(num_categories)

bars = ax.bar(bar_positions, category_counts, bar_width, color=colors, tick_label=[custom_category_names[i] if custom_category_names else f'Class {i+1}' for i in range(num_categories)])

# 设置x轴标签倾斜以便阅读
ax.tick_params(axis='x', labelrotation=60)

# 在每个柱子上方添加类别数量
for rect, count in zip(bars, category_counts):
    height = rect.get_height()
    ax.annotate(f'{count}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset.
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=10,
                color='dimgrey' if height > 10 else 'black')

ax.set_xlabel('Classes')
ax.set_ylabel('Number of Instances')
ax.set_title('Distribution of Object Classes in the DIOR Dataset')
ax.set_xlim([-0.5, num_categories - 0.5])
ax.set_ylim([0, max(category_counts) * 1.1])

# 可选：添加网格线
# ax.grid(axis='y', linestyle='--')


# 显示并保存图像
plt.savefig("DIOR.png", dpi=300)
plt.show()