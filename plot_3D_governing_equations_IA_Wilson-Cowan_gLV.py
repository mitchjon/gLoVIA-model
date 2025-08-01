import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
import numpy as np

def Relu(x):
    return np.maximum(0, x)

X = np.arange(0.001, .999 , 0.001)
#X = np.arange(-.999, .999 , 0.001)
Y = np.arange(-10, 10.1, 0.1)
m = 0
M = 1
r=0

X, Y = np.meshgrid(X, Y)

# Wilson-Cowan:
# zfunc = lambda x,y: (x*(M-x)/M) * (r + y + np.log((M-x)/x))

# IA:
# zfunc = lambda x,y: (M - x) * Relu(y) + (m - x) * Relu(-y) + r*x

# gLV:
zfunc = lambda x,y: x*(y+r)


Z = zfunc(X,Y)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.set_facecolor((0, 0, 0, 0))  # transparent axes background
# fig.patch.set_alpha(0)          # transparent figure background

surf = ax.plot_surface(X, Y, Z, cmap=cm.seismic, vmin=Z.min(), vmax = -Z.min() )#Z.max())

# Add bold lines parallel to the x-axis at specific y values
y_locs = np.linspace(Y.min(), Y.max(), 5)
for y0 in y_locs:
    x_line = X[0, :]
    y_line = np.full_like(x_line, y0)
    z_line = zfunc(x_line,y0)
    ax.plot(x_line, y_line, z_line+.0001, color='darkred', linewidth=3, zorder=10)

# Uncomment this block to do y-parallel lines instead
x_locs = np.linspace(
X.min(), X.max(), 5)
for x0 in x_locs:
    y_line = Y[:, 0]
    x_line = np.full_like(y_line, x0)
    z_line = zfunc(x0,y_line)
    ax.plot(x_line, y_line, z_line+.0001, color='blue', linewidth=3, zorder=10)

#ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
ax.xaxis.set_rotate_label(False)
ax.set(xlabel="$a_i$", ylabel="net input")#, zlabel="$\dfrac{da_i}{dt}$")
ax.zaxis.set_rotate_label(False)
ax.set_zlabel("$\dfrac{da_i}{dt}$",rotation=0)
transparent = True
# plt.savefig(r"C:\Users\mitch\Documents\OSU\Research data and code\gLoVIA-paper\Wilson-Cowan_3d_ODE_eq_visualization.png", dpi=1000, transparent=transparent)
# plt.savefig(r"C:\Users\mitch\Documents\OSU\Research data and code\gLoVIA-paper\IA_3d_ODE_eq_visualization.png", dpi=1000, transparent=transparent)
plt.savefig(r"C:\Users\mitch\Documents\OSU\Research data and code\gLoVIA-paper\gLV_3d_ODE_eq_visualization.png", dpi=1000, transparent=transparent)

# plt.show()