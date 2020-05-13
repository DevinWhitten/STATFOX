
import numpy as np
import pandas as pd
import pickle as pkl
import parmap
import emcee, corner
from scipy.stats import gaussian_kde
from matplotlib.transforms import Affine2D
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import scipy
from scipy.interpolate import LinearNDInterpolator as LinearND
from scipy.stats import skewnorm, norm
from scipy import integrate
from scipy.optimize import fminbound
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic_2d
from statsmodels.nonparametric.kde import KDEUnivariate
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)


############################################################
def MAD(array):
    return np.median(np.abs(array - np.median(array)))

def S_MAD(array):
    return MAD(array)/0.6745


def print_columns(FRAME):
    for column in FRAME:
        print(column)

    return


def resample(input_array, frac = 0.8):
    ### Basic bootstrap iteration
    ## sample with replacement

    size = int(frac * len(input_array))
    reindex = np.random.choice(np.random.randint(0, size, size),
                               size = size, replace=True)

    return np.array(input_array)[reindex]



def resample_frame(input_frame, frac = 0.8):

    size = int(frac * len(input_frame))
    reindex = np.random.choice(np.random.randint(0, size, size),
                                   size = size, replace=True)

    return input_frame.iloc[reindex]


def gen_kde_pdf(distribution, bounds=None, kde_width = None):
    ## boundary correction for KDE

    if bounds == None:
        print("\t setting bounds to max value")
        var_min, var_max = min(distribution), max(distribution)

    else:
        distribution = distribution[np.where((distribution > bounds[0]) & (distribution < bounds[1]))]
        var_min, var_max = bounds[0], bounds[1]

    lower = var_min - abs(distribution - var_min)
    upper = var_max + abs(distribution - var_max)
    merge = np.concatenate([lower, upper, distribution])

    if kde_width == None:
        print("... setting kde_width")
        kde_width = S_MAD(distribution)/2.

    KDE_MERGE = KDEUnivariate(merge)
    KDE_MERGE.fit(bw = kde_width)

    SCALE = np.divide(1., integrate.quad(KDE_MERGE.evaluate, var_min, var_max)[0])

    return lambda X: SCALE * KDE_MERGE.evaluate(X)







def uniform_kde_sample(frame, variable, bounds, p_scale=0.7, cut=True):
    ### updated uniform sample function to
    ### homogenize the distribution of the training variable.


    print("... uniform_kde_sample")

    if variable == 'TEFF':
        kde_width = 100
    else:
        kde_width = 0.15

    ### Basics
    var_min, var_max = min(frame[variable]), max(frame[variable])

    distro = np.array(frame[variable])

    ### Handle boundary solution

    lower = var_min - abs(distro - var_min)
    upper = var_max + abs(distro - var_max)
    merge = np.concatenate([lower, upper, distro])

    ### KDE

    KDE_MERGE = KDEUnivariate(merge)
    KDE_MERGE.fit(bw = kde_width)

    #### interp KDE_MERGE for computation speed
    span = np.linspace(var_min, var_max, 100)
    KDE_FUN = interp1d(span, KDE_MERGE.evaluate(span))

    ### Rescale
    full_c = len(distro) / integrate.quad(KDE_MERGE.evaluate, var_min, var_max)[0]
    #### This rescales the original distribution KDE function


    ### respan, because I don't want to be penalized for low counts outide variable range
    respan = np.linspace(bounds[0], bounds[1], 100)

    scale = np.percentile(KDE_MERGE.evaluate(respan)*full_c, p_scale * 100.)

    ### Accept-Reject sampling
    sample = np.random.uniform(0, 1, len(distro)) * KDE_FUN(distro) * full_c
    boo_array = sample < scale

    selection = frame.iloc[boo_array].copy()
    shuffle   = selection.iloc[np.random.permutation(len(selection))].copy()

    return shuffle

######
def determine_scale(catalog, parameter, pdf_fun, params, bounds = [-3.0, -0.5]):

    #### This func attempts to find the optimal scale value for an input pdf function
    #### such that the number of stars is maximized
    #### Catalog   : Master Catalog with arbitrary function
    #### parameter : the variable with which to consider
    #### pdf_fun   : input_function pdf
    #### params    : parameters of the input function, might delete

    param_span = np.linspace(*bounds, 100)

    ### Get the PDF of the master function
    KDE= KDEUnivariate(catalog[parameter])
    KDE.fit(bw=np.std(catalog[parameter])/3.)
    KDE_FUN = interp1d(param_span, KDE.evaluate(param_span))


    #####

    N = len(catalog[catalog[parameter].between(*bounds)])

    int_points = param_span[np.where((param_span > bounds[0]) & (param_span < bounds[1]))]

    ## Here's the pdf scale for the parent
    parent_scale = N / integrate.quad(KDE_FUN, bounds[0], bounds[1],
                              points = int_points,
                              limit=200)[0]



    ### This is the optimization section
    x0 = np.median(parent_scale*KDE_FUN(param_span) / pdf_fun(param_span, *params))
    print(x0)

    err_fun = lambda scale : (abs(parent_scale*KDE_FUN(param_span) - scale*pdf_fun(param_span, *params)) *  \
                             (1 +np.heaviside(scale*pdf_fun(param_span, *params) -  parent_scale*KDE_FUN(param_span), 0))).sum()
    #err_fun = lambda scale : (abs(N*KDE_FUN(param_span) - scale*pdf_fun(param_span, *params))).sum()

    result = minimize( err_fun, x0,
                        method='SLSQP', bounds = [(0.0, None)])


    return result, lambda x: parent_scale * KDE_FUN(x)


######

def sample_pdf(catalog, parameter, pdf_fun, params, bounds):

    ## Catalog: pd.DataFrame() input catalog with arbitrary distribution function
    ## input_fun: desired distribution of sample
    ## scale:   scale of sample

    param_span = np.linspace(min(catalog[parameter]), max(catalog[parameter]), 100)

    print("... determine master KDE")

    KDE = KDEUnivariate(catalog[parameter])
    KDE.fit(bw=np.std(catalog[parameter])/3)

    KDE_FUN = interp1d(param_span, KDE.evaluate(param_span))

    ## need to rescale within the bounds.

    NORM = np.divide(1.,
                     integrate.quad(KDE.evaluate, bounds[0], bounds[1],
                                    points = param_span[np.where((param_span > bounds[0]) & (param_span < bounds[1]))],
                                    limit=200)[0])

    ##########################################

    N = len(catalog[catalog[parameter].between(*bounds)])

    ############################################

    ### we need the scale from the other function

    result, kde_fun = determine_scale(catalog, parameter, pdf_fun, params, bounds = bounds)



    sample = np.random.uniform(0.0, 1.0, len(catalog)) * len(catalog) * NORM * KDE_FUN(catalog[parameter])

    boo_array = sample < result['x'] * pdf_fun(catalog[parameter], *params)

    return catalog[boo_array & (catalog[parameter].between(bounds[0], bounds[1], inclusive=True))].copy()


    #########################

def mask_stat(X, Y, Z, xrange, yrange, mincount=5):
    xedges = np.linspace(*xrange, 30)
    yedges = np.linspace(*yrange, 30)

    H, xedges, yedges = np.histogram2d(X, Y, bins = (xedges, yedges))

    BIN_STAT          = binned_statistic_2d(X, Y, Z,
                               statistic='median', bins = (xedges, yedges))

    masked_BIN_STAT = np.ma.masked_where(H.T < mincount, BIN_STAT[0].T)
    X, Y = np.meshgrid(xedges, yedges)

    return X, Y, masked_BIN_STAT



def mask_stat_between(X, Y, Z, xrange, yrange, mincount=[0, 5]):
    xedges = np.linspace(*xrange, 30)
    yedges = np.linspace(*yrange, 30)

    H, xedges, yedges = np.histogram2d(X, Y, bins = (xedges, yedges))

    BIN_STAT          = binned_statistic_2d(X, Y, Z,
                               statistic='median', bins = (xedges, yedges))
    print("SANITY 2")

    masked_BIN_STAT = np.ma.masked_where( (H.T < mincount[0]) | (H.T > mincount[1]) , BIN_STAT[0].T)
    #masked_BIN_STAT = np.ma.masked_where(H.T > mincount[1], masked_BIN_STAT)

    X, Y = np.meshgrid(xedges, yedges)


    X_array = np.unique(X)
    Y_array = np.unique(Y)

    X_array = [np.mean([X_array[i], X_array[i+1]]) for i in range(len(X_array) - 1)]
    Y_array = [np.mean([Y_array[i], Y_array[i+1]]) for i in range(len(Y_array) - 1)]

    X, Y = np.meshgrid(X_array, Y_array)
    INDEX = np.invert(masked_BIN_STAT.mask.flatten())

    return X.flatten()[INDEX], Y.flatten()[INDEX], masked_BIN_STAT.flatten()[INDEX]


################################################################################
### MCMC
################################################################################


def alpha_prior(alpha):

    if alpha < -7:
        return -np.inf
    if alpha > 0:
        return -np.inf

    return 0.0

def mu_prior(mu):


    return 5*np.log(norm.pdf(mu, 0.0, 0.3))
    return 0.





def LL_FUNCTION(theta, array):

    alpha = theta[0]
    mu    = theta[1]
    sigma = theta[2]
    N = len(array)
    try:
        constant = GLOBAL_INTERP([alpha, mu, sigma])[0]
    except:
        return -np.inf

    #LL = N*np.log(constant) + (erf((array - mu)/(np.sqrt(2) * sigma)*alpha) - 0.5*np.square((array - mu)/sigma) - np.log(sigma)).sum()
    LL =  N*np.log(constant) + np.log(skewnorm.pdf(array, alpha, mu, sigma)).sum()

    if not np.isfinite(LL):
        return -np.inf


    return LL + alpha_prior(alpha) + mu_prior(mu)


def pool_function(input, arg_bounds):
    alpha, mu, sigma = input[0], input[1], input[2]


    return np.divide(1., integrate.quad(skewnorm.pdf, *arg_bounds, args=(alpha, mu, sigma,))[0])





def generate_interp(dim, zbounds = [-3.5, -0.75], bound_dict = {'alpha' : [-5, -1.5],
                                                                'mu'    : [-1.5, 0.75],
                                                                'sigma' : [0.25, 1.0]}):



    alpha_array = np.linspace(*bound_dict['alpha'], dim)
    mu_array    = np.linspace(*bound_dict['mu'], dim)
    sigma_array = np.linspace(*bound_dict['sigma'], dim)


    print("... meshing")
    alpha_grid, mu_grid, sigma_grid = np.meshgrid(alpha_array,  mu_array, sigma_array, indexing='xy')

    print('... integrating')


    #p = multiprocessing.Pool(4)
    print("... parallelizing")
    result = parmap.map(pool_function, zip(alpha_grid.flatten(), mu_grid.flatten(), sigma_grid.flatten()), zbounds, pm_processes=4)

    output_frame = pd.DataFrame({'alpha' : np.array(alpha_grid.flatten()),
                                'mu'    : np.array(mu_grid.flatten()),
                                'sigma' : np.array(sigma_grid.flatten()),
                                'result': np.array(result).flatten()})

    print("... interpolating")
    INTERP = LinearND(points = output_frame[['alpha', 'mu', 'sigma']].values,
                      values = output_frame['result'])


    return INTERP


def gen_grid_interp(dim, zbounds = [-2.5, -0.25], bound_dict = {'alpha' : [-5, -1.5],
                                                                'mu'    : [-1.5, 0.75],
                                                                'sigma' : [0.25, 1.0]}):
    ### Modified to accomodate input function
    span_list = {}
    for key in bound_dict.keys():
        print(key)
        span_list[key] = np.arange(*bound_dict[key], 0.1)
        print(span_list[key])


    MESH = np.meshgrid(*[span_list[key] for key in span_list.keys()], indexing='ij')


    pool_output = parmap.map(pool_function,
                        zip(*[GRID_ELE.flatten() for GRID_ELE in MESH]),
                        zbounds, pm_processes=4)

    print([len(span_list[key]) for key in span_list.keys()])

    grid_interp = RegularGridInterpolator([GRID_ELE.flatten() for GRID_ELE in MESH],
                            np.array(pool_output).flatten().reshape([len(span_list[key]) for key in span_list.keys()])
                            )


    return grid_interp





def kde_param(distribution, x0):
    ### kde_param tries to ensure correct handling of multimodal distributions

    ### compute kernal density estimation
    KDE = KDEUnivariate(distribution)

    KDE.fit(bw=MAD(distribution)/2.0)

    result = scipy.optimize.minimize(lambda x: -1*KDE.evaluate(x),
    x0 = x0, method='Powell')  ## Powell has been working pretty well.

    return {'result' : float(result['x']), 'kde' : KDE}


def get_params(sampler, burnin = 100):
    print('fuxck')
    ndim = sampler.chain.shape[2]
    print(ndim)
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

    #print(samples)
    value2 = np.median(samples, axis=0)
    kde_output = [kde_param(samples[:, i], value2[i]) for i in range(ndim)]

    #value2 = np.median(samples, axis=0)
    #std =    np.std(samples, axis=0)


    return [ele['result'] for ele in kde_output]



def custom_corner(sampler, burnin):
    ndim = sampler.chain.shape[2]
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))





    return



def plot_GAMMA_grid(LEFT_SAMPLER, RIGHT_SAMPLER, burnin=100, lim =3,
                    axis_dict = {'lamb' : [2., 3], 'mu' : [3, 6]}, kernel=None):

    ndim = LEFT_SAMPLER.shape[1]

    print('here')

    fig = plt.figure(figsize = (5, 4.5)) #, ax = plt.subplots(2, 2)

    ax = np.array([[plt.axes([0.15, 0.71, 0.55, 0.08]), plt.axes([0.85, 0.75, 0.15, 0.20])],
                    [plt.axes([0.15, 0.1, 0.55, 0.6]), plt.axes([0.71, 0.1, 0.08, 0.6])]])

    lim_map = {0: 'lamb', 1:'mu'}
    BINS = (20, 20)


    BANDWIDTH = 0.15
    #alpha mu
    #COUNTS, xedges, yedges = np.histogram2d(SAMPLER[:, 0], SAMPLER[:, 1],
    #                                        bins=BINS)

    #MASK = np.ma.masked_where(COUNTS.T < lim, COUNTS.T)

    #alpha_dict = {"median": np.median(SAMPLER[:, 0]),
    #             "scale":  np.std(SAMPLER[:, 0])}

    #beta_dict = {"median": np.median(SAMPLER[:, 1]),
    #             "scale":  np.std(SAMPLER[:, 1])}

    cmap=plt.cm.copper

    #########################################################

    print("... kerneling")

    lim = 10000
    ax[1, 0].scatter(LEFT_SAMPLER[:lim, 1], LEFT_SAMPLER[:lim, 0],
                        c = gaussian_kde(LEFT_SAMPLER[:lim, :].T, bw_method=BANDWIDTH).evaluate(LEFT_SAMPLER[:lim, :].T))

    ax[1, 0].scatter(RIGHT_SAMPLER[:lim, 1], RIGHT_SAMPLER[:lim, 0],
                        c = gaussian_kde(RIGHT_SAMPLER[:lim, :].T, bw_method=BANDWIDTH).evaluate(RIGHT_SAMPLER[:lim, :].T))



    #

    #span = np.linspace(alpha_dict['median'] - alpha_dict['scale'],
    #                   alpha_dict['median'] + alpha_dict['scale'], 80)

    #ax[1,0].axvspan(alpha_dict['median'] - alpha_dict['scale'],
    #                alpha_dict['median'] + alpha_dict['scale'], alpha=0.3)

    #ax[1,0].axhspan(beta_dict['median'] - beta_dict['scale'],
    #                beta_dict['median'] + beta_dict['scale'], alpha=0.3)

    #top_pdf = top_kernel.evaluate(span)

    #ax[0,0].fill_between(span, np.zeros(len(span)), top_pdf, alpha=0.30, color = 'grey')

    #span = np.linspace(alpha_dict['median'] - 5*alpha_dict['scale'],
    #                   alpha_dict['median'] + 5*alpha_dict['scale'], 80)
    #top_pdf = top_kernel.evaluate(span)

    #ax[0,0].plot(span, top_pdf, alpha=0.30, color = cmap(0.3))


    #right_kernel = gaussian_kde(SAMPLER[:, 1], bw_method=BANDWIDTH)

    #span = np.linspace(beta_dict['median'] - beta_dict['scale'],
    #                   beta_dict['median'] + beta_dict['scale'], 80)

    #ax[1,1].fill_betweenx(span, np.zeros(len(span)), right_kernel.evaluate(span), alpha=0.3, color = cmap(0.3))

    #span = np.linspace(beta_dict['median'] - 5*beta_dict['scale'],
    #               beta_dict['median'] + 5*beta_dict['scale'], 80)

    #ax[1,1].plot(right_kernel.evaluate(span), span, alpha=0.3, color = cmap(0.3))



    #kernel = gaussian_kde(SAMPLER.T, bw_method = BANDWIDTH)

    #DENSITY = kernel.evaluate(SAMPLER.T)

    #ax[1,0].scatter(SAMPLER[:, 0], SAMPLER[:, 1], s=3, alpha=0.2, c=DENSITY, cmap=cmap, zorder=3)


    plt.setp(ax[1,1].get_yticklabels(), visible=False)
    plt.setp(ax[0,0].get_yticklabels(), visible=False)
    plt.setp(ax[1,1].get_xticklabels(), visible=False)
    plt.setp(ax[0,0].get_xticklabels(), visible=False)

    #ax[1,0].text(2.68, 4.8, r'$\alpha = $' + "%.2F" % alpha_dict['median'] + r"$\pm$" + "%.2F" % alpha_dict['scale'])



    ax[0,1].set_visible(False)

    [label.tick_params(which ='both', direction='in') for label in ax.flatten()]
    [label.xaxis.set_minor_locator(AutoMinorLocator()) for label in ax.flatten()]

    [label.yaxis.set_minor_locator(AutoMinorLocator()) for label in ax[1, :]]

    ax[1,0].tick_params(which='both', top=True, right=True)
    #[label.set_xlim([2.67, 2.73]) for label in ax[:, 0]]
    #[label.set_ylim([4.71, 4.82]) for label in ax[1, :]]


    ax[0,0].set_ylim([0, 50])
    ax[1,1].set_xlim([0, 25])

    ax[1,1].set_xticks([])

    ax[1,0].set_xlabel(r'$\alpha$')
    ax[1,0].set_ylabel(r'$\beta$')

    #plt.savefig('results/BETA_PLOT.pdf', format='pdf')

    return fig






def plot_SND_grid(SAMPLER, burnin=100, lim =2, axis_dict = {'alpha' : [-3, -1],
                                                            'mu' : [-0.75, -0.25],
                                                            'sigma' : [0.25, 0.75]}):

    ndim = SAMPLER.chain.shape[2]
    SAMPLER = SAMPLER.chain[:, burnin:, :].reshape((-1, ndim))

    print('here')

    fig, ax = plt.subplots(3, 3)



    BINS = (10, 10)


    ax[0,0].hist(SAMPLER[:, 0])
    ax[1,1].hist(SAMPLER[:, 1])
    ax[2,2].hist(SAMPLER[:, 2])

    ############################################################################
    #alpha mu
    COUNTS, xedges, yedges = np.histogram2d(SAMPLER[:, 0], SAMPLER[:, 1],
                                            bins=BINS, range = [axis_dict['alpha'], axis_dict['mu']])

    MASK = np.ma.masked_where(COUNTS.T < lim, COUNTS.T)

    ax[1,0].pcolormesh(xedges, yedges, np.log(MASK))
    ax[1,0].scatter(SAMPLER[:, 0], SAMPLER[:, 1], s=1)

    ############################################################################
    ## alpha sigma
    COUNTS, xedges, yedges = np.histogram2d(SAMPLER[:, 0], SAMPLER[:, 2],
                                            bins=BINS, range = [axis_dict['alpha'], axis_dict['sigma']])
    MASK = np.ma.masked_where(COUNTS.T < lim, COUNTS.T)

    ax[2,0].pcolormesh(xedges, yedges, np.log(MASK))
    ax[2,0].scatter(SAMPLER[:, 0], SAMPLER[:, 2], s=1)


    ############################################################################
    ## alpha sigma
    COUNTS, xedges, yedges = np.histogram2d(SAMPLER[:, 1], SAMPLER[:, 2],
                                            bins=BINS, range = [axis_dict['mu'], axis_dict['sigma']])
    MASK = np.ma.masked_where(COUNTS.T < lim, COUNTS.T)

    ax[2,1].pcolormesh(xedges, yedges, np.log(MASK))
    ax[2,1].scatter(SAMPLER[:, 1], SAMPLER[:, 2], s=1)


    return





def plot_corner(sampler, burnin, bound_dict = {'alpha' : [-6, -0.5],
                                               'mu'    : [-1.0, 0.75],
                                               'sigma' : [0.5, 1.25]}):


    ndim = sampler.chain.shape[2]
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    fig = corner.corner(samples, labels= [r"$\alpha$", r"$\mu$", r"$\sigma$"],
                        color='black', hist2d_kwargs = {'plot_density':True}, hist_kwargs={'normed':True})

    alpha_array = np.linspace(*bound_dict['alpha'], 25)
    mu_array    = np.linspace(*bound_dict['mu'], 25)
    sigma_array = np.linspace(*bound_dict['sigma'], 25)


    #fig.suptitle(suptitle, fontsize=15)
    MEDIAN = np.median(samples, axis=0)
    std =    np.std(samples, axis=0)

    value2    = [kde_param(row, x0 = x0)['result'] for row, x0 in zip(samples.T, MEDIAN)]
    kde_array = [kde_param(row, x0 = x0)['kde'] for row, x0 in zip(samples.T, MEDIAN)]


    axes = np.array(fig.axes).reshape((ndim, ndim))

    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(value2[xi], color="r")
            ax.axhline(value2[yi], color="r")
            ax.plot(value2[xi], value2[yi], "sr")


    for i in range(ndim):
        span = np.linspace(min(samples.T[i]), max(samples.T[i]), 50)
        axes[i,i].axvline(value2[i], color='r', alpha=0.75)
        axes[i,i].plot(span, kde_array[i].evaluate(span))


    axes[2,0].set_xlabel(r"$\alpha$")
    axes[2,1].set_xlabel(r"$\mu$")
    axes[2,2].set_xlabel(r"$\sigma$")

    axes[1, 0].set_ylabel("$\mu$")
    axes[2, 0].set_ylabel("$\sigma$")


    ## ALPHA SET
    [label.set_xlim([-6.5, -1.]) for label in axes[:, 0]]
    #[label.set_xlim([0.25, 0.75]) for label in axes[:, 0]]

    ## SIGMA SET
    [label.set_ylim([0.5, 0.9]) for label in axes[2, :-1]]
    #axes[2, 2].set_xlim([0.5, 0.9])

    #[label.set_xlim([-5, 5]) for label in axes[:, 0]]
    return fig, {"mean": value2, "std":std}



######## SPATIAL CONVOLUTION #############
######## SPATIAL CONVOLUTION #############
######## SPATIAL CONVOLUTION #############

def gaussian_weight(x, y, MU, sigma):
    #print(MU.shape)
    return sigma * np.exp((-0.5/sigma**2) * (np.power(x - MU[:, 0], 2) + np.power(y - MU[:, 1], 2)))


def weight_value(X, y, MU, sigma = 0.2):
    ## X is the input coordinate of interest
    ## y is the VALUE to be computed
    ## MU is the distance to each value
    ## sigma

    ## first get the weight array
    weight_array = gaussian_weight(X[0], X[1], MU, sigma)

    return np.dot(y, weight_array)/(weight_array.sum())


def spatial_2D_convolve(X_COLS, Y_COL, SIGMA = 0.25):

    return [weight_value(X_ELE, y = Y_COL, MU = X_COLS) for X_ELE in X_COLS]

######## SPATIAL CONVOLUTION #############


### BIN RESAMPLING
def local_norm(array, normx, bandwidth):
    ### array: input array for normalization
    ### normx: input x value for local normalization
    ### bandwidth: width of Gaussian convolution kernel.

    KDE_ = gaussian_kde(array, bw_method=bandwidth)

    return 1./KDE_.evaluate(normx)


def peak_norm(array, bandwidth):
    ### local_norm(), but at the KDE peak

    KDE_ = gaussian_kde(array, bw_method=bandwidth)

    res = minimize(lambda x : KDE_.evaluate(x), x0 = np.median(array))

    print(res)
    print("..............")

    return 1./KDE_.evaluate(res['x'])




def bin_stats(array, error, bins = None, xrange=[-4.0, 0.0], iterations = 100, normed=True):

    ## resamples in input 1d array
    bin_array = []

    for i in range(iterations):

        sample = resample(np.random.normal(array, np.ones(len(array)) * error))
        sample = sample[np.isfinite(sample)]
        sample = sample[(sample > xrange[0]) & (sample < xrange[1])]

        if bin != None:
            bin_count, edges = np.histogram(sample, bins=bins, density=True)
            bin_array.append(bin_count)


    # average
    MEAN = np.median(np.array(bin_array), axis=0)

    #calculate variance
    VAR = np.var(np.array(bin_array), axis=0)

    if normed:
        return {"MEAN" : MEAN/sum(MEAN), 'STD' : np.sqrt(VAR + MEAN)/sum(MEAN)}

    print("Not")
    return {"MEAN" : MEAN, 'STD' : np.sqrt(VAR)}
