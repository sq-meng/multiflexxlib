import multiflexxlib as mfl
import matplotlib.pyplot as plt

df = mfl.read_and_bin()
print(df)
p = df.plot()
p.set_plim(pmin=0, pmax=99.8)
plt.show()
