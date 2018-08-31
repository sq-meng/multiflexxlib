import multiflexxlib as mfl
import os
import pkg_resources
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # not zip safe, but acceptable for tests

files = [os.path.join(THIS_DIR, os.pardir, 'sampledata/MnF2/058777'),
         os.path.join(THIS_DIR, os.pardir, 'sampledata/MnF2/058778')]


df = mfl.read_and_bin(files)
p = df.plot([2, 3])
p.set_lognorm()
p.set_norm(None)
p.auto_lim()

df.cut_voronoi([-1, 0, -0.5], [-1, 0, -1], subset=[2, 3])
