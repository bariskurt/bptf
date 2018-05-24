import parafac

# How to run:
# 1) Go to the parent directory
# 2) Run: nosetests --nocapture test/test_parafac.py


def test_parafac_synth():
    print('Testing PARAFAC on synthetic data...')
    p = parafac.Parafac(shape=(3, 4, 2), n_components=2)
    p_est = parafac.Parafac(shape=p.shape, n_components=p.n_components)
    dist = p_est.fit(p.X)
    for i in range(len(dist)):
        if i == 0:
            print(dist[0])
        else:
            print('{} ; {}'.format(dist[i], dist[i]-dist[i-1]))
    print('Num. iterations: {}'.format(len(dist) - 1))
    print('\nX - X_est:')
    print(p.X - p_est.X)
    print('done.')

