## Author: Devin D. Whitten
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.utils import resample

def fit(distribution, n_components, iterations=30):
    ## Bootstrap Gaussian Mixture Modeling
    distro_size = len(distribution)
    distribution = np.array(distribution)

    GM_STATS = {'means' : [], 'cov': [], 'weights': [],
                'AIC': [], 'BIC': []}

    for i in range(iterations):
        data = resample(distribution,
                        replace=True,
                        n_samples= distro_size)

        work_data = data.reshape((-1,1))

        GM = GaussianMixture(n_components)

        GM.fit(work_data)

        MEANS = GM.means_.flatten()

        SORT = np.argsort(MEANS)

        ### main stats
        GM_STATS['means'].append(GM.means_.flatten()[SORT])
        GM_STATS['cov'].append(GM.covariances_.flatten()[SORT])
        GM_STATS['weights'].append(GM.weights_[SORT])

        ### Information Criterion
        GM_STATS['AIC'].append(GM.aic(distribution.reshape(-1, 1)))
        GM_STATS['BIC'].append(GM.bic(distribution.reshape(-1, 1)))

    ### now just average and ship
    return {"means" : np.median(np.array(GM_STATS['means']), axis=0),
            "std" : np.sqrt(np.median(np.array(GM_STATS['cov']), axis=0)),
            'weights' : np.median(np.array(GM_STATS['weights']), axis=0),
            'aic' : np.median(GM_STATS['AIC']), 'bic' : np.median(GM_STATS['BIC'])}


def GMM(distro, n_components = 2):
    GM = GaussianMixture(n_components)

    GM.fit(np.array(distro).reshape((-1, 1)))

    return GM



def KM_BOOTSTRAP(data, n_components, iterations = 30):
    KM_STATS = {'means' : [], 'cov': []}

    for i in range(iterations):
        data = resample(distribution,
                        replace=True,
                        n_samples= distro_size)



        KM = Kmeans(n_clusters = n_components)

        KM.fit(data)

        CENTERS = KM.cluster_centers_

        SORT = CENTERS[np.argsort(CENTERS[:, 0])]



    ### now just average and ship
    return {"means" : np.median(np.array(GM_STATS['means']), axis=0),
            "std" : np.sqrt(np.median(np.array(GM_STATS['cov']), axis=0)),
            'weights' : np.median(np.array(GM_STATS['weights']), axis=0),
            'aic' : np.median(GM_STATS['AIC']), 'bic' : np.median(GM_STATS['BIC'])}
