#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright © 2014 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2

# I am used to Python 3, this enables some future features here in Python 2.
from __future__ import division, absolute_import, print_function, unicode_literals

import unittest

import numpy as np

import main

class TestFolding(unittest.TestCase):
    def test_simple(self):
        x = np.array([1, 0, 0, 0])
        y = main.fold_data(x)
        expected = np.array([1, 0, 0])
        self.assertTrue(np.array_equal(y, expected))

    def test_case2(self):
        x = np.array([3, 2, 0, 2])
        y = main.fold_data(x)
        expected = np.array([3, 2, 0])
        self.assertTrue(np.array_equal(y, expected))

    def test_case3(self):
        x = np.array([3., 2., 1., 2., 5., 3.])
        y = main.fold_data(x)
        expected = np.array([3., 2.5, 3., 2.])
        self.assertTrue(np.array_equal(y, expected))

if __name__ == '__main__':
    unittest.main()
