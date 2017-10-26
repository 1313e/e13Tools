# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

version = {}
exec(open('e13tools/version.py').read(), version)

setup(name="e13Tools",
      version=version['__version__'],
      author="Ellert van der Velden",
      author_email='ellert_vandervelden@outlook.com',
      description=("Provides a collection of functions that were created by "
                   "1313e."),
      url='https://www.github.com/1313e/e13Tools',
      license='BSD-3',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          ],
      keywords='e13tools utilities sampling',
      python_requires='>=3.3',
      packages=find_packages(),
      package_dir={'e13tools': "e13tools"},
      install_requires=['numpy>=1.6', 'matplotlib>=1.4.3', 'astropy'],
      zip_safe=False,
      entry_points={
          'console_scripts': [
              'e13tools=e13tools:main'
          ],
      }
      )
