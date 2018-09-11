import multiflexxlib as mfl

df = mfl.read_and_bin()
df.to_csv()  # Writes to data folder.

c = df.cut_voronoi([1, 0, 0], [1.5, 0, 0], subset=[3])
c.to_csv()  # Will ask you where to save data. Only cut objects containing exactly one scan can be exported.
