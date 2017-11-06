# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:56:20 2017

@author: 1313e
"""

import doctest
import pkgutil
import e13tools

mod_list = [name for _, name, _ in pkgutil.walk_packages(e13tools.__path__, e13tools.__name__+'.')]
tests = doctest.DocTestSuite()

skip_cases = ['e13tools.sampling.lhcs']

for name in mod_list:
    if name in skip_cases:
        tests.addTests(doctest.DocTestSuite(name, optionflags=doctest.SKIP))
    else:
        tests.addTests(doctest.DocTestSuite(name))
tests.debug()
