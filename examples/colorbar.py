# Adds a colorbar and removes buttons, for making figures for publications.
import multiflexxlib as mfl
import matplotlib.pyplot as plt

print('Double-click on a plot to open new window.')
df = mfl.read_and_bin()
print(df)
p = df.plot(controls=False)
p.add_colorbar()
plt.show()
