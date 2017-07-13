# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

exec(open('e13tools/version.py').read())

setup(name="e13tools",
      version=__version__,
      author="1313e",
      description=("Provides a collection of functions that were created by "
                   "1313e."),
      packages=find_packages(),
      package_dir={'e13tools': "e13tools"},
      zip_safe=False
      )
