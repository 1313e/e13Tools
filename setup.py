# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(name="e13tools",
      version='0.1.0a4',
      author="Ellert van der Velden",
      author_email='ellert_vandervelden@outlook.com',
      description=("Provides a collection of functions that were created by "
                   "1313e."),
      url='https://www.github.com/1313e/e13tools',
      license='BSD-3',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          ],
      keywords='e13tools utilities',
      python_requires='>=3.3',
      packages=find_packages(),
      package_dir={'e13tools': "e13tools"},
      install_requires=['numpy', 'matplotlib', 'astropy'],
      zip_safe=False,
      entry_points={
          'console_scripts': [
              'e13tools=e13tools:main'
          ],
      }
      )
