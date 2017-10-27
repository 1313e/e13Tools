# -*- coding: utf-8 -*-

import e13tools
from setuptools import setup, find_packages

setup(name="e13tools",
      version=e13tools.__version__,
      author="Ellert van der Velden",
      author_email='ellert_vandervelden@outlook.com',
      description=("Provides a collection of functions that were created by "
                   "1313e."),
      url='https://www.github.com/1313e/e13Tools',
      license='BSD-3',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          ],
      keywords='e13tools utilities sampling',
      python_requires='>=2.6, !=3.0.*, !=3.1.*, !=3.2.*, <4',
      packages=find_packages(),
      package_dir={'e13tools': "e13tools"},
      install_requires=['numpy>=1.6', 'matplotlib>=1.4.3', 'astropy>=1.3'],
      zip_safe=False,
      entry_points={
          'console_scripts': [
              'e13tools=e13tools:main'
          ],
      }
      )
