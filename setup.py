from setuptools import setup

requires = ['pyclipper>=1.1.0',
            'numpy>=1.14',
            'pandas>=0.22.0',
            'matplotlib>=2.1.0']

setup(name='multiflexxlib',
      version='0.1.0.dev2',
      description='Tools library for CAMEA neutron detector MultiFLEXX',
      url='http://github.com/yumemi5k/multiflexxlib',
      author='Siqin Meng',
      author_email='mengsq04@gmail.com',
      license='',
      packages=['multiflexxlib'],
      zip_safe=False,
      include_package_data=True)