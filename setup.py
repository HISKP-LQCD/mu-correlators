#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright © 2014 Martin Ueding <dev@martin-ueding.de>

from setuptools import setup, find_packages

setup(
    author="Martin Ueding",
    author_email="dev@martin-ueding.de",
    description="Finds file formats that might become unreadable.",
    license="GPL2",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: GNU General Public License v2",
        "Programming Language :: Python",
    ],
    name="mu-correlators",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'mu-correlators=main:main',
        ],
    },
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
    ],
    url="https://github.com/martin-ueding/mu-correlators",
    #download_url="http://martin-ueding.de/download/PROJECT/",
    version="0.1",
)
