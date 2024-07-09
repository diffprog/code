import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

import seaborn as sns

palette = sns.color_palette("tab10", 3)
rcParams.update({
    'font.size': 16,
    'lines.linewidth': 3,
    'mathtext.fontset': 'dejavusans',
})

n = 4

rng = np.random.default_rng(3)
keys = rng.uniform(0, 1, n)
values = rng.uniform(0, 1, n)




def kernel(x, y, sigma=1.):
  return np.exp(-np.abs(x-y)**2/(2*sigma**2))

def kernel_smoother(k, keys, values, sigma=1.):
  num = np.sum([kernel(k, ki, sigma)*vi for ki, vi in zip(keys, values)])
  den = np.sum([kernel(k, ki, sigma) for ki in keys])
  return num/den

fig, ax = plt.subplots(1)

ks = np.linspace(0, 1, 100)
ax.scatter(keys, values, color=palette[0], marker='x', s=70, zorder=2, label='Key-value pairs')
ax.plot(ks, [kernel_smoother(k, keys, values, 0.07) for k in ks], color=palette[1], zorder=1, label='Kernel estimator')
ax.set_xlabel('Keys')
ax.set_ylabel('Values')
ax.set_ylim(bottom=0.)
ax.set_xlim(left=0., right=1.)
ax.locator_params(axis='y', nbins=3)
ax.locator_params(axis='x', nbins=3)
ax.legend(loc='lower right')
fig.tight_layout()
# fig.savefig('dict_kernel_estim.pdf', format='pdf', bbox_inches='tight')
plt.show()