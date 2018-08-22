import multiflexxlib as mfl
import matplotlib.pyplot as plt

print('Double-click on a plot to open new window.')
df = mfl.read_and_bin()
print(df)
p = df.plot()
p.auto_lim()
plt.show()