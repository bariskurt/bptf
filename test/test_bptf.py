from src import bptf

# How to run:
# 1) Go to the parent directory
# 2) Run: nosetests --nocapture test/test_bptf.py


def test_bptf():
    print('Testing BPTF on synthetic data...')
    b = bptf.BPTF(shape=(3, 4, 2), n_components=2, a=10, b=10)
    b_est = bptf.BPTF(shape=b.shape, n_components=b.n_components)
    elbo = b_est.fit(b.X)
    for i in range(len(elbo)):
        if i == 0:
            print(elbo[0])
        else:
            print('{} ; {}'.format(elbo[i], elbo[i]-elbo[i-1]))
    print('Num. iterations: {}'.format(len(elbo) - 1))
    print('done.')
