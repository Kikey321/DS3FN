import matplotlib.pyplot as plt

# Sample data (replace with your actual data)
# lambda_vals = [1/10, 1/8, 1/6, 1/4, 1/2]
x = range(5)
YOLOv5 = [1.892, 2.09, 2.46, 2.92, 3.734]
YOLOX = [1.82, 1.89, 1.964, 2.121, 2.547]
YOLOv7 = [2.050, 2.18, 2.38, 2.75, 3.42]
YOLOv8 = [1.953, 2.003, 2.145, 2.543, 3.22]
DS3FN = [2.050, 2.18, 2.38, 2.75, 3.42]


plt.plot(x, YOLOv5, '*-', color='blue', label='YOLOv5', linestyle='dashed')
plt.plot(x, YOLOX, '^-', color='green', label='YOLOX', linestyle='dashed')
plt.plot(x, YOLOv7, 's-', color='orange', label='YOLOv7', linestyle='dashed')
plt.plot(x, YOLOv8, 'o-', color='red', label='YOLOv8', linestyle='dashed')
plt.plot(x, DS3FN, 's-', color='pink', label='DS3FN', linestyle='dashed')


plt.xticks([0, 1, 2, 3, 4], ["[1, 2]", "[2, 3]", "[3, 4]", "[4, 5]", "[5, 6]"], fontsize=10)
plt.ylim(1.4, 4.1)
plt.yticks([1.5, 2, 2.5, 3, 3.5, 4])
plt.xlabel('Number of tasks per client vehicle')
plt.ylabel('Average Task Delay (s)')

# Show grid
plt.grid(True)

# Adding a legend
plt.legend()

# Show the plot
plt.show()
plt.savefig("datasize", dpi=500, bbox_inches='tight')