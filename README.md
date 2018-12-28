# BPTF Tutorial
Bayesian Poisson Tensor Factorization

This project includes the detailed derivations and a Python code for the tensor factorization with Poisson observations. The code is written using Numpy, Scipy and Tensorly libraries.

## Doc:
You can find the detailed derivations of the BPTF model in bptf.pdf

## Sources:

- NMF: Poisson NMF implementation
- Parafac: Parafac solver with Alternating Least Squares
- BPTF: Bayesian Poisson Tensor Factorization implementation

## How to run tests:
You can run the tests from the main directory using nosetests.

- nosetests --nocapture test/test_nmf.py
- nosetests --nocapture test/test_parafac.py
- nosetests --nocapture test/test_bptf.py

