# -*- coding: utf-8 -*-
"""
Testing script for travis

"""

import pytest

def tests():
  # Check if all modules can be imported
  import e13tools as e13
  import e13tools.pyplot as e13plt
  import e13tools.sampling as e13spl
  
if(__name__ == '__main__'):
  tests()
