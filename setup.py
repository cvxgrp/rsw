from setuptools import setup, find_packages

setup(
    name="rsw",
    version="0.1",
    author="Shane Barratt, Guillermo Angeris, Stephen Boyd",
    packages=find_packages(),
    setup_requires=[
        "numpy>=1.15",
    ],
    install_requires=[
        "cvxpy>=1.0",
        "scipy>=1.4",
        "pandas",
        "qdldl>=0.1.1",
        "matplotlib"
    ],
    url="http://github.com/cvxgrp/rsw/",
    license="Apache License, Version 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
