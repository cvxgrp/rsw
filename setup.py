from setuptools import setup, find_packages

setup(
    name="rsw",
    version="0.1",
    author="Shane Barratt, Guillermo Angeris, Stephen Boyd",
    packages=find_packages(),
    install_requires=[
        "numpy >= 1.15",
        "cvxpy >= 1.0",
        "scipy >= 1.1",
        "pandas >= 1.0",
        "qdldl >= 0.1.0"],
    url="http://github.com/cvxgrp/rsw/",
    license="Apache License, Version 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
