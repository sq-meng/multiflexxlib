import multiflexxlib as mfl
from matplotlib import pyplot as plt
if __name__ == "__main__":
    df = mfl.read_and_bin()
    print(df)
    p = df.plot()
    print('close this console window after you are finished.')
    plt.show()
