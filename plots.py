import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

size = 3#15
_x, _y = np.meshgrid(np.arange(size), np.arange(size))
x, y = _x.ravel(), _y.ravel()

top = np.zeros([size, size])
values = [5, 2, 3, 2, 4, 7, 4, 5, 1]
#places = [2, 7, 12]
places = [0, 1, 2]
for i in range(len(places)):
    for j in range(len(places)):
        top[places[i], places[j]] = values[len(places)*i+j]
top = top.ravel()

bottom = np.zeros_like(x)

width = depth = 0.2 #1

#color = ['b', 'g', 'r', 'c', 'm', 'y']
color = None
fig = plt.figure(figsize=(20, 15))
ax = Axes3D(fig)
ax.bar3d(x, y, bottom, width, depth, top, color=color, shade=True)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
#fig.savefig('s_dil_conv.png')
fig.savefig('dil_conv.png')
plt.show()

