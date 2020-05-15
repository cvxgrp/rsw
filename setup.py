from setuptools import setup, find_packages

setup(
    name="rsw",
    version="0.1",
    author="Shane Barratt, Guillermo Angeris, Stephen Boyd",
    packages=find_packages(),
    setup_requires=[
        "numpy",
    ],
    install_requires=[
        "cvxpy",
        "scipy",
        "pandas",
        "qdldl",
        "matplotlib"
    ],
    url="http://github.com/cvxgrp/rsw/",
    license="Apache License, Version 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
