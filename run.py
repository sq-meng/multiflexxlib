import multiflexxlib as mfl
import matplotlib.pyplot as plt

df = mfl.read_and_bin()
print(df)
p = df.plot()
plt.show()
