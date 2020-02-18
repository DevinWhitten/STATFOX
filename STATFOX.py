
import numpy as np

from scipy import integrate
from scipy.optimize import fminbound
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import interp1d
from statsmodels.nonparametric.kde import KDEUnivariate


############################################################
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
