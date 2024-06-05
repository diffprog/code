from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib.collections as mcoll
import numpy as np
import seaborn as sns

rcParams.update({
    'font.size': 14,
    'lines.linewidth': 4,
    'mathtext.fontset': 'dejavusans',
})


def colorline(
    x,
    y,
    z=None,
    cmap=plt.get_cmap('copper'),
    norm=plt.Normalize(0.0, 1.0),
    linewidth=3,
    alpha=1.0,
    ax=None,
):
  """Plot a colored line.

  http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
  http://matplotlib.org/examples/pylab_examples/multicolored_line.html

  Plot a colored line with coordinates x and y Optionally specify colors in the 
  array z.
  Optionally specify a colormap, a norm function and a line width
  """

  # Default colors equally spaced on [0,1]:
  if z is None:
    z = np.linspace(0.0, 1.0, len(x))

  # Special case if a single number:
  if not hasattr(
      z, '__iter__'
  ):  # to check for numerical input -- this is a hack
    z = np.array([z])

  z = np.asarray(z)

  points = np.array([x, y]).T.reshape(-1, 1, 2)
  segments = np.concatenate([points[:-1], points[1:]], axis=1)
  lc = mcoll.LineCollection(
      segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha
  )

  if ax is None:
    ax = plt.gca()
  ax.add_collection(lc)

  return lc


fig, axs = plt.subplots(1, 2, figsize=(4 * 2, 4))

num_tangents = 8
palette = sns.color_palette('plasma', num_tangents)


def fun(x):
  return x * np.log(x) + (1 - x) * np.log(1 - x)


def conj_fun(u):
  return np.log(1 + np.exp(u))


###############
# Envelope plot
ax = axs[0]
xs = np.linspace(0.0, 1.0, 100)
conj_domain = (-2.0, 1.0)

# Plot function
ys = fun(xs)
ax.plot(xs, ys, color='k')

# Plot tangents
for i, u in enumerate(np.linspace(*conj_domain, num_tangents)[1:-1]):
  ys = u * xs - conj_fun(u)
  # Plot tangent
  ax.plot(xs, ys, '--', color=palette[i], linewidth=2)
  # Plot intercept
  ax.plot(0.0, -conj_fun(u), 'o', color=palette[i])
  if i == num_tangents - 3:
    # Add legend to intercept
    ax.annotate(
        '$-f^*(v)$',
        (0, -conj_fun(u)),
        xytext=(-0.3, -conj_fun(u) - 0.12),
        color=palette[i],
        fontsize=16,
    )
    # Add legend to slope (indicating what v corresponds to)
    ax.plot(
        xs[:20],
        np.zeros_like(xs[:20]) - conj_fun(u),
        '--',
        linewidth=1,
        color=palette[i],
    )
    ax.vlines(
        xs[19],
        -conj_fun(u),
        u * xs[19] - conj_fun(u),
        linestyles='--',
        linewidth=1,
        color=palette[i],
    )
    ax.annotate(
        '$v$',
        (xs[19], u * xs[8] - conj_fun(u)),
        xytext=(xs[19] + 0.03, u * xs[8] - conj_fun(u)),
        color=palette[i],
    )

# Approriate zoom in y axis
ax.set_ylim(-1.2, 0.0)

# Set y_axis to the left
ax.spines['left'].set_position(('data', 0))

# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Show ticks in the left and lower axes only
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# Less ticks
ax.locator_params(axis='y', nbins=3)
ax.locator_params(axis='x', nbins=3)

# Labels
ax.set_xlabel('$u$', fontsize=16)
ax.set_ylabel('$f\,(u)$', fontsize=16)

#########################
# Conjugate function plot
ax = axs[1]

# Plot conjugate function
us = np.linspace(*conj_domain, 100)
vs = conj_fun(us)
ax.plot(us, vs, color='k')

# Higlight points corresponding to the tangents computed previously
for i, u in enumerate(np.linspace(*conj_domain, num_tangents)[1:-1]):
  # Vertical lines instead of the tangents ince now we are in conjugate domain
  ax.vlines(
      u, 0, conj_fun(u), color=palette[i], linestyles='dashed', linewidth=2
  )
  # Intercepts are highlighted by points along the function now
  ax.plot(u, conj_fun(u), 'o', color=palette[i])
# Make x-axis as a gradient color line to illustrate continuous possibilities of
# tangents
colorline(
    us, np.zeros_like(us), cmap=plt.get_cmap('plasma'), ax=ax, linewidth=8
)
# Appropriate zooms
ax.set_ylim(bottom=-0.01)
ax.set_xlim(us[0], us[-1])

# Add labels
ax.set_xlabel('$v$', fontsize=16)
ax.set_ylabel('$f^*(v)$', fontsize=16)

# Less ticks
ax.locator_params(axis='y', nbins=3)

fig.tight_layout()
plt.show()
