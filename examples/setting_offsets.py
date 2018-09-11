# Apply custom UB-matrix and angle offsets without overriding raw data files, to save you headache 3 years later.
# By the way, using accidental Bragg reflection spurions is a great way to pinpoint offset values.
import multiflexxlib as mfl

# Creates a custom UB-matrix for when the one in data files is wrong.
u = mfl.UBMatrix(latparam=[4.87, 4.87, 3.3, 90, 90, 90], hkl1=[1, 0, 0], hkl2=[0, 0, 1])

a3_offset_dict = {58773: 2.0, '058774': 2.3}  # Accepts integer or string for file names.
a4_offset = 1.2  # Just a number if the offset is consistent but non-zero.
df = mfl.read_and_bin(ub_matrix=u, a3_offset=a3_offset_dict, a4_offset=a4_offset)

# Now you can do your plotting.
