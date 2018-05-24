import nmf

# How to run:
# 1) Go to the parent directory
# 2) Run: nosetests --nocapture test/test_nmf.py


def test_nmf():
    print('Testing NMF on synthetic data...')
    alpha = 1
    beta = 10
    n_components = 2
    shape = (8, 6)

    n = nmf.NMF(shape=shape, n_components=n_components, alpha=alpha, beta=beta)

    n_est = nmf.NMF(shape=n.shape, n_components=n.n_components, alpha=alpha, beta=beta)
    elbo = n_est.fit(n.X, max_iter=2000)
    for i in range(len(elbo)):
        if i == 0:
            print(elbo[0])
        else:
            print('{} ; {}'.format(elbo[i], elbo[i]-elbo[i-1]))
    print('Num. iterations: {}'.format(len(elbo) - 1))
    print('done.')
