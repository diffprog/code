import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams


rcParams.update({
  'font.size': 14,
  'lines.linewidth': 2
})


def get_relaxation_logical_op(relaxation_type):
  if relaxation_type == 'probabilistic':

    def and_op(pi1, pi2):
      return pi1 * pi2

    def or_op(pi1, pi2):
      return pi1 + pi2 - pi1 * pi2

  elif relaxation_type == 'extremum':

    def and_op(pi1, pi2):
      return np.minimum(pi1, pi2)

    def or_op(pi1, pi2):
      return np.maximum(pi1, pi2)

  elif relaxation_type == 'lukasiewicz':

    def and_op(pi1, pi2):
      return np.maximum(pi1 + pi2 - 1, 0)

    def or_op(pi1, pi2):
      return np.minimum(pi1 + pi2, 1)

  else:
    raise NotImplementedError(f'{relaxation_type} relaxation_type not implemented.')
  
  return and_op, or_op


def plot_logic(relaxation_type, fig_title=None):
  ops = get_relaxation_logical_op(relaxation_type)
  fig, axs = plt.subplots(1, 2, figsize=(2*4.6, 4))
  for ax, op, title in zip(axs, ops, ['And operator', 'Or operator']):
    palette = sns.color_palette('rocket', as_cmap=True)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = op(X, Y)

    im = ax.pcolormesh(X, Y, Z, cmap=palette)
    ax.contour(X, Y, Z, 10, interpolation='none', linestyles='dotted', alpha=0.5, cmap='Greys_r')

    ax.locator_params(axis='y', nbins=3)
    ax.locator_params(axis='x', nbins=3)
    ax.set_xlabel('$\pi$')
    ax.set_ylabel("$\pi'$")
    ax.set_title(title, fontsize=18)
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', label='Value')
    cbar.ax.locator_params(nbins=3)
  fig.tight_layout()
  if fig_title is not None:
    fig.suptitle(fig_title, y=1.02)
  fig.savefig('and_or_ops_' + relaxation_type + '.pdf', format='pdf', bbox_inches='tight')


plot_logic('probabilistic')
plot_logic('extremum', 'Extremum T-Norm')
plot_logic('lukasiewicz', 'Łukasiewicz T-Norm')

plt.show()
