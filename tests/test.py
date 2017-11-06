# -*- coding: utf-8 -*-
"""
Testing program for e13Tools.

"""
import doctest
from pkgutil import walk_packages
import e13tools

# Obtain list with all modules found in the package
mod_list = [name for _, name, _ in walk_packages(e13tools.__path__,
                                                 e13tools.__name__+'.')]

tests = doctest.DocTestSuite()

# Make list of modules containing scripts that involve randomness
skip_cases = ['e13tools.sampling.lhcs']

# Make a unittest for every doctest found in the package
for name in mod_list:
    if name in skip_cases:
        tests.addTests(doctest.DocTestSuite(name, optionflags=doctest.SKIP))
    else:
        tests.addTests(doctest.DocTestSuite(name))

# Perform the unittest debug
tests.debug()
