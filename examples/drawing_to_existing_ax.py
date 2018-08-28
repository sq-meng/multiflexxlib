# Draws into existing matplotlib axes object, for when you absolutely need to control everything.
import multiflexxlib as mfl
import matplotlib.pyplot as plt
df = mfl.read_and_bin()
f, ax = plt.subplots(1)

mfl.draw_voronoi_patch(ax, df.data.loc[2])  # Reference a row in DataFrame
ax.set_aspect(df.ub_matrix.figure_aspect)  # Set display aspect to be equal in absolute reciprocal length.
plt.show()