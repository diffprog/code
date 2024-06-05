import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams

rcParams.update({
  'font.size': 14,
  'lines.linewidth': 3
})

fig, axs = plt.subplots(1, 3, figsize=(3*4, 4))

n = 100
xs = np.linspace(-2, 2, n)

hv_ys = np.heaviside(xs, 1)
mid_point = int(n/2)
hv_ys[mid_point] = np.nan
axs[0].plot(xs[mid_point], 1, 'o', color='tab:blue')
axs[0].plot(xs, hv_ys, color='tab:blue')
axs[0].set_title('Discontinuous at 0')
axs[0].locator_params(axis='y', nbins=3)

def sparse_sigmoid(x):
  return (np.maximum(x+1, 0) + np.minimum(-x+1, 0))/2
sp_ys = sparse_sigmoid(xs)

axs[1].plot(xs, sp_ys, color='tab:green')
axs[1].set_title('Continuous\nnon-differentiable at 1 and -1')
axs[1].locator_params(axis='y', nbins=3)

def logistic(x):
  return 1/(1+np.exp(-x))

lg_ys = logistic(xs)
axs[2].plot(xs, lg_ys, color='tab:orange')
axs[2].set_title('Differentiable everywhere')
axs[2].set_ylim(0, 1)
axs[2].locator_params(axis='y', nbins=3)

fig.tight_layout(pad=2.0)
plt.show()

