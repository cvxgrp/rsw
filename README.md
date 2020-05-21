# rsw

Optimal representative sample weighting (`rsw`) in Python.
This package implements the methods described in the paper [Optimal Representative Sample Weighting](https://stanford.edu/~boyd/papers/optimal_representative_sampling.html).
At a high level, the package takes in a dataset assigns to each data point a nonnegative weight, so as to make
weighted sample averages equal or close to some desired averages.
For more details behind the math, we highly recommend checking out the paper.

## Installation

We *highly recommend* upgrading your version of pip before installing `rsw`:
```bash
$ pip install --upgrade pip
```

Clone the repository, then run:
```bash
$ python setup.py install
```

## API
`rsw` exposes one method, with the signature
```
def rsw(df, funs, losses, regularizer, lam=1, **kwargs):
    """Optimal representative sample weighting.

    Arguments:
        - df: Pandas dataframe
        - funs: functions to apply to each row of df.
        - losses: list of losses, each one of rsw.EqualityLoss, rsw.InequalityLoss, rsw.LeastSquaresLoss,
            or rsw.KLLoss()
        - regularizer: One of rsw.ZeroRegularizer, rsw.EntropyRegularizer,
            or rsw.KLRegularizer, rsw.BooleanRegularizer
        - lam (optional): Regularization hyper-parameter (default=1).
        - kwargs (optional): additional arguments to be sent to solver. For example: verbose=False,
            maxiter=5000, rho=50, eps_rel=1e-5, eps_abs=1e-5.

    Returns:
        - w: Final sample weights.
        - out: Final induced expected values as a list of numpy arrays.
        - sol: Dictionary of final ADMM variables. Can be ignored.
    """
```

## Running the examples

There are two examples, one on simulated data and one on the [CDC BRFSS dataset](https://stanford.edu/~boyd/papers/optimal_representative_sampling.html).

### Simulated
To run the simulated example, after installing `rsw`, navigate to the `examples` folder and run:
```
$ python simulated.py
```

### CDC BRFSS
To run the CDC BRFSS example, first download the data:
```
$ cd examples/data
$ wget https://www.cdc.gov/brfss/annual_data/2018/files/LLCP2018XPT.zip
$ unzip LLCP2018XPT.zip
```

In the examples folder, to run all the examples in the paper, execute the following command:
```
$ python brfss.py
```

## Citing
If you use rsw in your research, please consider citing us by using the following bibtex:
```
@misc{barratt2020optimal,
  title={Optimal Representative Sample Weighting},
  author={Barratt, Shane and Angeris, Guillermo and Boyd, Stephen},
  month={May},
  year={2020},
  howpublished={\texttt{https://stanford.edu/~boyd/papers/optimal_representative_sampling.html}}
}
```

## License

This repository carries a permissive Apache 2.0 license.
