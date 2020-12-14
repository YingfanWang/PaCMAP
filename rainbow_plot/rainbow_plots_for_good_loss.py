import matplotlib.pyplot as plt
import numpy as np
import umap
from scipy import integrate
from pacmap import PaCMAP
from experiments.run_experiments import data_prep


cmap_fig = plt.cm.get_cmap("Spectral")
cmap = plt.cm.get_cmap("RdYlGn_r")
cmap_ = plt.cm.get_cmap("gist_yarg")


# If you would like discrete ladders, use ladder_map
# Otherwise, just leave it, see examples below
def ladder_map(grids, ladder_range):
    l_map = np.zeros(grids.shape)
    for thres in ladder_range:
        l_map += (grids > thres).astype(np.float32)
    l_map /= len(ladder_range)
    return l_map

# parameter "a" and "b" use default values as below
def attr(x):
    return -pow(x, 0.79)/(1 + pow(x, 2))

def repul(x):
    return 0.895 * x/(1 + pow(x, 2))/(0.001 + pow(x, 2))

def integ_attr(b):
    res = np.zeros(b.shape)
    for i in range(b.shape[1]):
        res[0][i] = integrate.quad(attr, 0, b[0][i], points=[0])[0]
    return res

def integ_repul(b):
    res = np.zeros(b.shape)
    for i in range(b.shape[0]):
        res[i][0] = integrate.quad(repul, 0, b[i][0], points=[0])[0]
    return res

# For t-SNE we choose a neighbor and further point to visualize forces on them (using COIL20 dataset, 300 iterations)
def t_attr(x):
    qij = 1.0 / (x ** 2 + 1.0) / 11500
    qij = np.maximum(qij, 1e-12)
    force = - (8.58 * 1e-5 - qij) * x / (1.0 + x ** 2)
    return force

def t_repul(x):
    qij = 1.0 / (x ** 2 + 1.0) / 11500
    qij = np.maximum(qij, 1e-12)
    force = - 10 * (1.19 * 1e-8 - qij) * x / (1.0 + x ** 2)
    return force

def t_integ_attr(b):
    res = np.zeros(b.shape[0])
    for i in range(b.shape[0]):
        res[i] = integrate.quad(t_attr, 0, b[i], points=[0])[0]
    return res

def t_integ_repul(b):
    res = np.zeros(b.shape[0])
    for i in range(b.shape[0]):
        res[i] = integrate.quad(t_repul, 0, b[i], points=[0])[0]
    return res

def t_integ_attr_(b):
    res = np.zeros(b.shape)
    for i in range(b.shape[1]):
        res[0][i] = integrate.quad(t_attr, 0, b[0][i], points=[0])[0]
    return res

def t_integ_repul_(b):
    res = np.zeros(b.shape)
    for i in range(b.shape[0]):
        res[i][0] = integrate.quad(t_repul, 0, b[i][0], points=[0])[0]
    return res


plt.figure(figsize=(28, 15))



plt.axes([0.047, 0.52, 0.2, 0.44])
x = np.linspace(0.0001, 100, num=7000)# d_ij
y = np.linspace(0.0001, 100, num=7000)# d_ik
xx, yy = np.meshgrid(x, y, sparse=True)
tsne_loss = -t_integ_attr_(xx) - t_integ_repul_(yy)
tsne_U = t_attr(xx) + 0 * yy
tsne_V = t_repul(yy) + 0 * xx
plt.streamplot(xx, yy, tsne_U, tsne_V, density=(2.4, 1.0), linewidth=0.8, arrowsize=2.5, maxlength=1.)
im = plt.imshow(tsne_loss, origin='lower', extent=(0.0001, 100, 0.0001, 100), cmap=cmap)
cb = plt.colorbar(im)
cb.ax.tick_params(labelsize=23)
plt.title('Loss (t-SNE)', fontsize=38)
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)
plt.xlabel(r'$d_{ij}$', fontsize=38)
plt.ylabel(r'$d_{ik}$', fontsize=38)

plt.axes([0.047, 0.03, 0.2, 0.44])
tsne_grad_inten = np.sqrt(tsne_U ** 2 + tsne_V ** 2)
tsne_grad_inten = np.array(tsne_grad_inten)
for i in range(tsne_grad_inten.shape[0]):
    for j in range(tsne_grad_inten.shape[1]):
        if tsne_grad_inten[i, j] > 0.00005:
            tsne_grad_inten[i, j] = 0.00005
plt.streamplot(xx, yy, tsne_U, tsne_V, density=(2.4, 1.0), linewidth=0.8, arrowsize=2.5, maxlength=1.)
im = plt.imshow(tsne_grad_inten, origin='lower', extent=(0.0001, 100, 0.0001, 100), cmap=cmap_)
cb = plt.colorbar(im)
cb.ax.tick_params(labelsize=23)
plt.title('Gradient magnitude', fontsize=34)
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)
plt.xlabel(r'$d_{ij}$', fontsize=38)
plt.ylabel(r'$d_{ik}$', fontsize=38)



plt.axes([0.293, 0.52, 0.2, 0.44])
x = np.linspace(0.0001, 25, num=7000) # d_ij
y = np.linspace(0.0001, 25, num=7000) # d_ik
xx, yy = np.meshgrid(x, y, sparse=True)
u_loss = -integ_attr(xx) -integ_repul(yy)
u_U = attr(xx) + 0*yy
u_V = repul(yy) + 0*xx
plt.streamplot(xx, yy, u_U, u_V, density=(2.4, 1.0), linewidth=0.8, arrowsize=2.5, maxlength=1.)
im = plt.imshow(u_loss, origin='lower', extent=(0.0001, 25, 0.0001, 25), cmap=cmap)
cb = plt.colorbar(im)
cb.ax.tick_params(labelsize=23)
plt.title('Loss (UMAP)', fontsize=38)
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)
plt.xlabel(r'$d_{ij}$', fontsize=38)
plt.ylabel(r'$d_{ik}$', fontsize=38)

plt.axes([0.293, 0.03, 0.2, 0.44])
u_grad_inten = np.sqrt(u_U ** 2 + u_V ** 2)
for i in range(u_grad_inten.shape[0]):
    for j in range(u_grad_inten.shape[1]):
        if u_grad_inten[i, j] > 1:
            u_grad_inten[i, j] = 1
plt.streamplot(xx, yy, u_U, u_V, density=(2.4, 1.0), linewidth=0.8, arrowsize=2.5, maxlength=1.)
im = plt.imshow(u_grad_inten, origin='lower', extent=(0.0001, 25, 0.0001, 25), cmap=cmap_)
cb = plt.colorbar(im)
cb.ax.tick_params(labelsize=23)
plt.title('Gradient magnitude', fontsize=34)
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)
plt.xlabel(r'$d_{ij}$', fontsize=38)
plt.ylabel(r'$d_{ik}$', fontsize=38)


plt.axes([0.543, 0.52, 0.2, 0.44])
x = np.linspace(0.0001, 200, num=7000) # d_ij
y = np.linspace(0.0001, 200, num=7000) # d_ik
xx, yy = np.meshgrid(x, y, sparse=True)
t_loss = (1.0 + xx**2)/(2.0 + xx**2 + yy**2)
t_U = (2*xx + 2 * xx * yy**2)/(2 + xx**2 + yy**2)**2
t_V = (-2*yy*(1 + xx**2))/(2 + xx**2 + yy**2)**2
plt.streamplot(xx, yy, -t_U, -t_V, density=(2.4, 1.0), linewidth=0.8, arrowsize=2.4, maxlength=1.)
im = plt.imshow(t_loss, origin='lower', extent=(0.0001, 200, 0.0001, 200), cmap=cmap)
cb = plt.colorbar(im)
cb.ax.tick_params(labelsize=23)
plt.title('Loss (TriMAP)', fontsize=38)
plt.xticks([50, 100, 150, 200], fontsize=23)
plt.yticks([50, 100, 150, 200], fontsize=23)
plt.xlabel(r'$d_{ij}$', fontsize=38)
plt.ylabel(r'$d_{ik}$', fontsize=38)

plt.axes([0.543, 0.03, 0.2, 0.44])
t_grad_inten = np.sqrt(t_U ** 2 + t_V ** 2)
for i in range(t_grad_inten.shape[0]):
    for j in range(t_grad_inten.shape[1]):
        if t_grad_inten[i, j] > 0.012:
            t_grad_inten[i,j] = 0.012
plt.streamplot(xx, yy, -t_U, -t_V, density=(2.4, 1.0), linewidth=0.8, arrowsize=2.5, maxlength=1.)
im = plt.imshow(t_grad_inten, origin='lower', extent=(0.0001, 200, 0.0001, 200), cmap=cmap_)
cb = plt.colorbar(im)
cb.ax.tick_params(labelsize=23)
plt.title('Gradient magnitude', fontsize=34)
plt.xticks([50, 100, 150, 200],fontsize=23)
plt.yticks([50, 100, 150, 200], fontsize=23)
plt.xlabel(r'$d_{ij}$', fontsize=38)
plt.ylabel(r'$d_{ik}$', fontsize=38)


plt.axes([0.795, 0.52, 0.2, 0.44])
x = np.linspace(0.0001, 50, num=7000) # d_ij
y = np.linspace(0.0001, 50, num=7000) # d_ik
xx, yy = np.meshgrid(x, y, sparse=True)
p_loss = 1.5 * (xx**2 + 1)/(11.0 + xx**2) + 3.0/(2.0 + yy**2)
p_U = -1.5 * (20*xx)/(11.0 + xx**2)**2 + (0 * yy)
p_V = 3 * (2*yy)/(2 + yy**2)**2 + (0 * xx)
plt.streamplot(xx, yy, p_U, p_V, density=(2.4, 1.0), linewidth=0.8, arrowsize=2.4, maxlength=1.)
im = plt.imshow(p_loss, origin='lower', extent=(0.01, 50, 0.01, 50), cmap=cmap)
cb = plt.colorbar(im)
cb.ax.tick_params(labelsize=23)
plt.title('Loss (PaCMAP)', fontsize=38)
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)
plt.xlabel(r'$d_{ij}$', fontsize=38)
plt.ylabel(r'$d_{ik}$', fontsize=38)

plt.axes([0.795, 0.03, 0.2, 0.44])
p_grad_inten = np.sqrt(p_U ** 2 + p_V ** 2)
plt.streamplot(xx, yy, p_U, p_V, density=(2.4, 1.0), linewidth=0.8, arrowsize=2.5, maxlength=1.)
for i in range(p_grad_inten.shape[0]):
    for j in range(p_grad_inten.shape[1]):
        if p_grad_inten[i, j] > 0.5:
            p_grad_inten[i,j] = 0.5
im = plt.imshow(p_grad_inten, origin='lower', extent=(0.0001, 50, 0.0001, 50), cmap=cmap_)
cb = plt.colorbar(im)
cb.ax.tick_params(labelsize=23)
plt.title('Gradient magnitude', fontsize=34)
plt.xticks(fontsize=23)
plt.yticks(fontsize=23)
plt.xlabel(r'$d_{ij}$', fontsize=38)
plt.ylabel(r'$d_{ik}$', fontsize=38)


plt.savefig('rainbow_good_loss')
