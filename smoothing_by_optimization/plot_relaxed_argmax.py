import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from scipy.special import softmax as softargmax


def projection_simplex(v, z=1):
  n_features = v.shape[0]
  u = np.sort(v)[::-1]
  cssv = np.cumsum(u) - z
  ind = np.arange(n_features) + 1
  cond = u - cssv / ind > 0
  rho = ind[cond][-1]
  theta = cssv[cond][-1] / float(rho)
  w = np.maximum(v - theta, 0)
  return w


sparse_argmax = projection_simplex


def compute_data(ix=1):
  T1 = np.arange(-2.5, 2.5, 0.005)
  T2 = np.arange(-2.5, 2.5, 0.005)
  X, Y = np.meshgrid(T1, T2)

  keys = ('argmax', 'softargmax', 'sparseargmax')
  data = {key: np.zeros_like(X) for key in keys}

  for i in range(len(T1)):
    for j in range(len(T2)):
      z = np.array([X[i, j], Y[i, j], 0])
      data['argmax'][i, j] = 1 if np.argmax(z) == ix else 0
      data['softargmax'][i, j] = softargmax(z)[ix]
      data['sparseargmax'][i, j] = sparse_argmax(z)[ix]

  return data


def plot3d(ax, Z, title=None):
  T1 = np.arange(-2.5, 2.5, 0.005)
  T2 = np.arange(-2.5, 2.5, 0.005)
  X, Y = np.meshgrid(T1, T2)

  ax.plot_surface(X, Y, Z,
                  linewidth=0,
                  antialiased=False,
                  alpha=1,
                  color='C0', edgecolor='gray',
                  rstride=1, cstride=1)

  if title:
    ax.set_title(title)

  ax.set_xlabel("$u_1$")
  ax.set_ylabel("$u_2$")


def main(ix=1):
  data = compute_data(ix=ix)

  for key, Z in data.items():
    plt.set_cmap('tab20b')
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plot3d(ax, Z, title=key)
    plt.savefig("3dplot_{}.png".format(key), dpi=1200)
    plt.close()


if __name__ == '__main__':
  main()
