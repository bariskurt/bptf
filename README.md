# BPTF for Dummies
Bayesian Poisson Tensor Factorization

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

