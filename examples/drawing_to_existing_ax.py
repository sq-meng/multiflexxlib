# Draws into existing matplotlib axes object, for when you absolutely need to control everything.
import multiflexxlib as mfl
import matplotlib.pyplot as plt
df = mfl.read_and_bin()
f, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)

ax = axes[0, 0]
mfl.draw_voronoi_patch(ax, df.data.loc[2])  # Reference a row in DataFrame
ax.set_aspect(df.ub_matrix.figure_aspect)  # Set display aspect to be equal in absolute reciprocal length.

ax = axes[0, 1]
mfl.draw_scatter(ax, df.data.loc[2])
ax.set_aspect(df.ub_matrix.figure_aspect)

ax = axes[1, 0]
mfl.draw_voronoi_patch(ax, df.data.loc[2], mesh=True)
ax.set_aspect(df.ub_matrix.figure_aspect)

ax = axes[1, 1]
mfl.draw_interpolated_patch(ax, df.data.loc[2])
ax.set_aspect(df.ub_matrix.figure_aspect)
f.tight_layout()
plt.show()