# -*- coding: utf-8 -*-

"""
Setup file for the e13Tools package.

"""


# %% IMPORTS
# Built-in imports
from codecs import open

# Package imports
from setuptools import find_packages, setup


# %% SETUP DEFINITION
# Get the long description from the README file
with open('README.rst', 'r') as f:
    long_description = f.read()

# Get the requirements list
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

# Read the __version__.py file
with open('e13tools/__version__.py', 'r') as f:
    vfile = f.read()

# Obtain version from read-in __version__.py file
vstr = "__version__ = "
version = vfile.partition(vstr)[2].partition('\n')[0].replace("'", '').strip()

# Setup function declaration
setup(name="e13tools",
      version=version,
      author="Ellert van der Velden",
      author_email='ellert_vandervelden@outlook.com',
      description=("Provides a collection of functions that were created by "
                   "1313e."),
      long_description=long_description,
      url='https://www.github.com/1313e/e13Tools',
      license='BSD-3',
      platforms=["Windows", "Linux", "Unix"],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: BSD License',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: Unix',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Software Development :: Libraries :: Python Modules',
          'Topic :: Utilities',
          ],
      keywords='e13tools utilities latin hypercube sampling math tools',
      python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, <4',
      packages=find_packages(),
      package_dir={'e13tools': "e13tools"},
      include_package_data=True,
      install_requires=requirements,
      zip_safe=False,
      )
