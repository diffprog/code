import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams


rcParams.update({
  'font.size': 14,
  'lines.linewidth': 2
})


def get_compa_ops(type):
  if type == 'hard':

    def greater(u1, u2):
      return np.heaviside(u1 - u2, 1)

    def equal(u1, u2):
      return np.heaviside(u1 - u2, 1)*np.heaviside(u2 - u1, 1)

  elif type == 'soft_logistic':

    def greater(u1, u2):
      return  1/(1+np.exp(-(u1 - u2)))

    def equal(u1, u2):
      return 4/(2 + np.exp(u2-u1) + np.exp(u1-u2))

  else:
    raise NotImplementedError(f'{type} relaxation_type not implemented.')
  
  return greater, equal


def plot_compa_ops(type, fig_title=None):
  ops = get_compa_ops(type)
  fig, axs = plt.subplots(1, 2, figsize=(2*4.6, 4))
  for ax, op, title in zip(axs, ops, ['$u_1$ greater than $u_2$', '$u_1$ equal to $u_2$']):
    palette = sns.color_palette('rocket', as_cmap=True)
    x = np.linspace(0, 4, 100)
    y = np.linspace(0, 4, 100)
    X, Y = np.meshgrid(x, y)
    Z = op(X, Y)

    # heatmap
    im = ax.pcolormesh(X, Y, Z, cmap=palette, linewidth=0, rasterized=True, vmin=0, vmax=1)

    ax.locator_params(axis='y', nbins=3)
    ax.locator_params(axis='x', nbins=3)
    if type == 'hard':
      ax.set_xlabel('$u_1$')
      ax.set_ylabel("$u_2$")
    else:
      ax.set_xlabel('$u_1$')
      ax.set_ylabel("$u_2$")
    ax.set_title(title, fontsize=18)
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', label='Value')
    cbar.ax.locator_params(nbins=3)
  fig.tight_layout()
  if fig_title is not None:
    fig.suptitle(fig_title, y=1.02)
  # fig.savefig('greater_equal_ops_' + type + '.pdf', format='pdf', bbox_inches='tight')


plot_compa_ops('hard')

plt.show()
