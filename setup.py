# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from codecs import open

# Get the long description from the README file
with open('README.rst', 'r') as f:
    long_description = f.read()

# Get the version
with open('e13tools/VERSION', 'r', encoding='utf-8') as version_file:
    __version__ = version_file.read().strip()

setup(name="e13tools",
      version=__version__,
      author="Ellert van der Velden",
      author_email='ellert_vandervelden@outlook.com',
      description=("Provides a collection of functions that were created by "
                   "1313e."),
      long_description=long_description,
      url='https://www.github.com/1313e/e13Tools',
      license='BSD-3',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          ],
      keywords='e13tools utilities sampling',
      python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, <4',
      packages=find_packages(exclude=['tests']),
      package_dir={'e13tools': "e13tools"},
      include_package_data=True,
      install_requires=['numpy>=1.6', 'matplotlib>=1.4.3,<2', 'astropy>=1.3'],
      zip_safe=False,
      entry_points={
          'console_scripts': [
              'e13tools=e13tools:main',
              ],
          },
      )
