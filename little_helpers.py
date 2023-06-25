"""
    Contents:

        * util
        * softmax
        * mean_l
        * gini (coefficient)
        * gauss
        * pdf_fit_error
        * gauss_generator
        * standardize
        * popcode_params
        * dev_from_uniform
        * puzzle_connectivity
        * show_and_save
        * plot_run_time
        * datetime
        * printT
        * print_progress
        * entropy
        * plot bin_avg

"""

import time
import numpy as np
import matplotlib.pyplot as mpl
from scipy.optimize import curve_fit
import math
from sklearn.metrics import adjusted_mutual_info_score

eps = 0.0000000001


def name():
    return 'little_helpers'

# %% util


def r(x):
    return np.arange(x)


def enum(x):
    return enumerate(x)


def pos(x):
    #print(type(x), x)
    if isinstance(x, (float, int)):
        return 0 if x < 0 else x
    else:
        x[x < 0] = 0
        return x

# %%

def normalized_input(input, max_scaling=1/5, plot=False):
    # max_scaling: maximum scaling factor for underrepresented inputs

    cmax = np.max(np.mean(input, 0))
    idcmax = np.argmax(np.mean(input, 0))  # column with most input

    max_scaling *= cmax

    rmax = np.max(input[:, idcmax])  # representative maximum input strength
    idrmax = np.argmax(input[:, idcmax])

    # normalize column means
    mean_cols = np.mean(input, 0)
    mean_cols[mean_cols < max_scaling] = max_scaling

    input_normalized = input / mean_cols

    if(plot):
        mpl.title("Normalized mean")
        mpl.plot(input_normalized.T)
        mpl.plot(np.mean(input_normalized, 0))
        mpl.show()

    # normalize maximum value
    input_normalized = input_normalized / input_normalized[idrmax, idcmax] * rmax

    if(plot):
        mpl.title("Normalized Maximum")
        mpl.plot(input_normalized.T)
        mpl.plot(np.mean(input_normalized, 0))
        mpl.show()
    
    return input_normalized

def softmax(x, beta=1):
    """
        Computes softmax values of x.
    """
    e_x = np.exp(beta*x - np.max(beta*x))
    return e_x / e_x.sum()


def mean_l(l, axis=0):
    """
        Computes mean of 2D np array of lists (of differing lengths).

    """

    if l.ndim == 1:
        l = l[:, None]

    a = np.zeros(l.shape[axis])
    for i in range(l.shape[axis]):
        k = []
        for j in range(l.shape[1-axis]):
            for m in range(len(l[i, j]) if axis == 0 else len(l[j, i])):
                k.append(l[i, j][m] if axis == 0 else l[j, i][m])
        a[i] = np.mean(k)

    return a


def max_l(list_array):
    """
        Computes max of 1D np array of lists (of differing lengths).

    """
    max_array = np.zeros(list_array.shape[0])
    for i, l in enumerate(list_array):
        max_array[i] = np.max(l)

    return max_array


def gini(x):
    """
        Computes gini coefficient on a body of numbers as a measure of sparseness.

    """
    x = x.flatten()
    x = np.sort(x)
    n = len(x)
    if not n:
        #print("[WARNING] Empty input array!")
        return n
    if x[0] < 0:
        print("[WARNING] Negative values in gini coefficient calculation!")

    s = np.sum(x*(np.arange(n)+1))

    if np.sum(x) != 0:
        g = s*2 / (n*np.sum(x)) - 1
    else:
        g = 0

    return g


# %%
def gauss(x, p):
    """
        Gaussian Kernel

        * p: parameters of Gauss

    """
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


def fit_gauss(x, *p):
    """
        Gaussian Kernel

    """
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


def pdf_fit_error(x, func=[gauss, fit_gauss], p0=None, rel=False, plot=False):
    """
        Computes difference between normalized sample pdf and fitted pdf**

        * x: sample distribution
        * func: functions for fitting and generating values of fitted pdf
        * p0: starting values for fitting procedure
        * rel: instead of absolute difference compute difference relative to fit
        * plot: whether to plot sample pdf and fitted pdf

        ** probability density function
    """
    x = x / np.sum(x)
    if p0 is None:
        p0 = [1, 0, 1]
    try:
        fit_params = curve_fit(func[1], np.arange(len(x)), x, p0)[0]
    except Exception as exc:
        fit_params = p0
        print(exc)
    fit = func[0](np.arange(len(x)), fit_params)
    if rel:
        error = np.round(np.sum(np.abs(x - fit)/fit), 3)
    else:
        error = np.round(np.sum(np.abs(x - fit)), 3)
    if plot:
        mpl.plot(fit)
        mpl.plot(x)
        mpl.title(str(np.round(fit_params, 1))+" Error: "+str(error))
    return error


# %%
def gauss_generator(A, mu, sigma, res):
    """
        Generates Gaussian population code

    """
    gaussian = np.round([A * np.exp(-(x-mu)**2/(2*(sigma+eps)**2))
                        for x in range(res)], 5)
    return gaussian


def gauss_generator_ring(A, mu, sigma, res):
    """
        Generates Gaussian population code ring attractor style

    """

    mu %= res

    # the parameter for normalizing sigma was derived randomly
    return np.round([A * np.exp(-(np.sin((x-mu)/(res)*np.pi))**2/(2*(sigma/((res)/10*np.pi))**2)) for x in range(res)], 3)


def dog_generator(A, mu, sigma, res):
    """
        Generates Difference of Gaussians population code

    """

    if isinstance(mu, (int, float)):
        mu = (mu, mu)

    return np.round([A[0] * np.exp(-(x-mu[0])**2/(2*sigma[0]**2)) - A[1] * np.exp(-(x-mu[1])**2/(2*sigma[1]**2)) for x in range(res)], 3)


def lin_generator(A, start, inv=0):
    """
        Generates a linear push-pull population code

    """

    return np.round([A*(inv*(res-1)+(1-2*inv)*x-start)/(res-1-start) for x in range(res)], 3)

# import numpy as np
# import matplotlib.pyplot as mpl
# import matplotlib

# matplotlib.rcParams['figure.figsize'] = [10, 6]
# eps = 0.0000000001

# res = 19

# for k in np.arange(9):
#     mpl.subplot(331+k)
#     rf = []
#     A = []
#     S = []
#     for i in np.arange(150):

#         a = 1
#         while a>1 or a<=0.1:
#             a = np.random.lognormal(np.log(0.3), 1)
#         s = -1
#         while s>res-2 or s<(res-1)/2:
#             s = res-np.random.lognormal(np.log(res*0.1), res/6)

#         for inv in (0,1):
#             rf.append(lin_generator(a, s, inv))
#             A.append(a)
#             if inv:
#                 S.append(res-1-s)
#             else:
#                 S.append(s)
#             rf[i*2+inv][rf[i*2+inv]<0] = 0
#             mpl.plot(rf[i*2+inv])

# mpl.show()

# rf_sorted = np.zeros((len(rf), res))
# idx = np.argsort(S)
# for i, r in enumerate(rf):
#     rf_sorted[i] = rf[idx[i]]

# response = []
# for i, x in enumerate(9+np.arange(-5, 6)*0.4):
#     inp = gauss_generator(1, x, 0.5, res)
#     # mpl.plot(inp)
#     # mpl.show()
#     response.append(np.dot(rf_sorted, inp)[len(rf)//2-10:len(rf)//2+10])
#     mpl.plot(response[i])
# mpl.show()

# r = np.array(response)
# C = np.cov(r.T)

# mpl.imshow(C)

# w1 = np.zeros(C.shape[0])+0.1
# w2 = np.zeros(C.shape[0])+0.1
# w3 = np.zeros(C.shape[0])+0.1
# post1 = 0
# post2 = 0
# post3 = 0
# for i in np.arange(10000):
#     pre = r[i%r.shape[0]]-np.mean(r, 0)
#     post1 = np.dot(w1, pre)
#     post2 = np.dot(w2, pre-w1*post1)
#     post3 = np.dot(w2, pre-w1*post1-w2*post2)

#     w1 += pre * post1 - post1**2*w1
#     w2 += pre * post2 - post2**2*w2
#     w3 += pre * post3 - post3**2*w3

#     w1[w1<0] = 0
#     w2[w2<0] = 0
#     w3[w3<0] = 0

# mpl.plot(w1)
# mpl.plot(w2)
# mpl.plot(w3)


# mpl.hist(S)
# mpl.show()

# mpl.plot(lin_generator(a, s))
# mpl.plot(rf[i])

def long_generator(A, mu, sigma, res):
    """
        broad stimulus generator (A = l)

        (see Fig. 2: https://doi.org/10.1016/j.neuron.2014.12.026)
    """

    return np.round([(1/(1+np.exp(-(x-res/2+A/2)/sigma))) * (1-1/(1+np.exp(-(x-res/2-A/2)/sigma))) for x in range(res)], 2)


# %%
def standardize(x):
    """
        standardizes a numpy array

    """

    return (x - np.mean(x)) / np.std(x)


# %%
def popcode_params(popcode, scale=None, min_zero=False):
    """
        computes mean and SD of a 1D distribution (neuron population)

        * popcode: ~pdf over scale
        * scale: scale of pdf

    """

    if sum(popcode) > 0:

        if scale is None:
            scale = np.arange(len(popcode))

        # ensure only positive values
        if min_zero:
            popcode = popcode - np.min(popcode)
        else:
            popcode = popcode - min(0, np.min(popcode))
        # normalize area below pdf
        y = popcode/sum(popcode)

        mean = sum(scale*y)
        sigma = np.sqrt(sum(y*(scale-mean)**2))

        return mean, sigma

    else:

        return 0, 0


def popcode_params2(popcode, scale, axis=None):
    """
        computes mean and SD of a 1D distribution (neuron population). Does not check for negative values!!!

        * popcode: ~pdf over scale
        * scale: scale of pdf

    """

    # normalize area below pdf
    y = popcode/(np.sum(popcode, axis)[:, None] + eps)

    mean = np.sum(scale[0]*y, axis)
    sigma = np.sqrt(np.sum(y*(scale-mean[:, None])**2, axis))

    return mean, sigma


def popcode_params3(popcode, scale=None):
    """
        Fits a Gaussian to a 1D distribution and returns the parameters of the best fit

        * popcode: ~pdf over scale
        * scale: scale of pdf

    """

    popcode[np.isnan(popcode)] = 0

    res = len(popcode)
    scale = res if scale is None else scale

    min_mu, min_s = popcode_params(popcode)
    A = np.max(popcode)
    min_loss = ml = np.sum(
        np.abs(popcode - gauss_generator(A, min_mu, min_s, res)))

    for j in np.arange(5):
        # try different sigma
        min_loss_old = min_loss
        r_s = np.linspace(min_s*(1-1/(2+j**2)), min_s/(1-1/(2+j**2)), 100)
        g = gauss_generator(A, min_mu, r_s, res)
        loss = np.sum(np.abs(popcode[:, None] - g), 0)
        if min(loss) < min_loss:
            min_loss = min(loss)
            min_s = r_s[np.argmin(loss)]

        # try different mu
        # min_loss_old = min_loss
        r_mu = np.linspace(min_mu-2*min_s*(1/(2+j**2)),
                           min_mu+2*min_s*(1/(2+j**2)), 100)
        g = gauss_generator(A, r_mu, min_s, res)
        loss = np.sum(np.abs(popcode[:, None] - g), 0)
        if min(loss) < min_loss:
            min_loss = min(loss)
            min_mu = r_mu[np.argmin(loss)]
        if min_loss_old - min_loss < 0.001:
            break

    if False:  # visualizing
        print(j)
        print("Simple loss:", np.round(ml, 4))
        print("Advanced loss:", np.round(min_loss, 4))

        mpl.plot(popcode)
        mpl.plot(gauss_generator(A, min_mu, min_s, res))
        mu, s = popcode_params(popcode)
        mpl.plot(gauss_generator(A, mu, s, res))
        mpl.show()

    return min_mu*scale/res, min_s*scale/res


# %%
def dev_from_uniform(x, bin_width=None, limits=None, vis=False):
    """
        computes deviation from uniform distribution

    """

    x = x.flatten()

    if limits is None:
        l = np.min(x)
        u = np.max(x)
    else:
        l = limits[0]
        u = limits[1]

    if bin_width is None:
        bin_width = 1

    ticks = [l, ]
    while (ticks[-1]+bin_width) <= u:
        ticks += (ticks[-1]+bin_width,)

    bins = len(ticks)
    expected = len(x)/bins
    dev = 0

    for bi in ticks:
        x1 = x[x >= bi]
        x2 = x1[x1 < bi+bin_width]
        dev += np.abs(len(x2) - expected)

    if vis:
        mpl.hist(x, bins=int(bins))
        mpl.title("Deviation from expected: "+str(np.round(dev/len(x), 2)))
        mpl.axhline(expected, color='black')
        mpl.show()

    return (dev/len(x))


# %%
def puzzle_connectivity(x, w=None, TD=True, short=False):
    """
        puzzles a connectivity matrix from hashed connectivity lists

            * x: .npz archive containing saved connectivity of a projection
            * w: matrix from a weight monitor (if 'None' will be taken from x)
            * TD: whether the output should be two-dimensional receptive fields
            * short: only process first and last temporal slice from weight monitor
    """

    if w is None:
        w = x['w']
        matrix = np.zeros(
            (int(np.max(max_l(x['pre_ranks']))+1), int(np.max(x['post_ranks'])+1)))

        # dendrite (list) i of projection w
        for i, d in enumerate(w):
            # synapse j in dendrite i
            for j, s in enumerate(d):
                matrix[x['pre_ranks'][i][j], i] = s
        if TD:
            matrix = np.reshape(matrix, (int(np.sqrt(matrix.shape[0])), int(np.sqrt(
                matrix.shape[0])), int(np.sqrt(matrix.shape[1])), int(np.sqrt(matrix.shape[1]))))

    else:
        matrix = np.zeros((2 if short else w.shape[0], int(
            np.max(max_l(x['pre_ranks']))+1), int(np.max(x['post_ranks'])+1)))

        # timestep t
        for t in (0, -1) if short else np.arange(w.shape[0]):
            # dendrite (list) i of projection w
            for i, d in enumerate(w[t]):
                # synapse j in dendrite i
                for j, s in enumerate(d):
                    matrix[t, x['pre_ranks'][i][j], i] = s
        if TD:
            matrix = np.reshape(matrix, (matrix.shape[0], int(np.sqrt(matrix.shape[1])), int(
                np.sqrt(matrix.shape[1])), int(np.sqrt(matrix.shape[2])), int(np.sqrt(matrix.shape[2]))))

    return matrix


# %%

def show_and_save(plot_id, dpi=300, sdir=''):
    """
        plots a mpl plot and saves it to disk

    """
    if len(sdir):
        if sdir[-1] != '/':
            sdir += '/'
    mpl.savefig(sdir + 'images/' + str(plot_id) + '.png', dpi=dpi)
    mpl.show()

    return plot_id+1


# %%
def plot_runtime(rdir='/scratch/forch/BodySchema/data', file='run_times.csv', param1=0, param2=-1):
    """
        plots simulation times of network runs

    """

    data = np.loadtxt(open(rdir+'/'+file, 'rb'), delimiter=',')

    mpl.plot(data[:, param1], data[:, param2])


# %%
def f(string, digits=2):
    """
        fills a string with leading digits to a desired length

    """

    diff = (digits - len(string))
    if diff < 0:
        diff = 0

    return '0' * diff + string


def datetime(start=0, end=19, F=False):
    """
        returns a string with the datetime, format: YYYYMMDDHHMMSS

    """

    dtstring = ''

    dtstring += str(time.localtime().tm_year)+("." if F else "")
    dtstring += f(str(time.localtime().tm_mon))+("." if F else "")
    dtstring += f(str(time.localtime().tm_mday))+("," if F else "")
    dtstring += f(str(time.localtime().tm_hour))+(":" if F else "")
    dtstring += f(str(time.localtime().tm_min))+(":" if F else "")
    dtstring += f(str(time.localtime().tm_sec))

    return dtstring[start:end]


def printT(*args):
    a = str(args[0])
    while a[0] == '\n':
        print('')
        a = a[1:]
    print("["+datetime(11, F=True)+"]", *((a,)+args[1:]))


# %%
def print_progress(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


# %%
def entropy(p):

    p[p <= 0] = 0.000001

    return -np.sum(p/np.sum(p) * np.log(p/np.sum(p)))


def entropy_from_raw(labels, base=None):
    """
    Computes entropy of label distribution.

    Taken from: https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
    """

    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute entropy
    base = math.e if base is None else base
    for i in probs:
        ent -= i * math.log(i, base)

    return ent


def mutual_information(a, b=None, scale=[0, 1], adj=False):
    """
        Computes Mutual Information of digitized empirical variables

    """

    eps = 0.00000001

    if len(a.shape) == 1:
        if b is not None:
            a = np.vstack((a, b))
        else:
            b = a

    Im = np.zeros((a.shape[0], a.shape[0]))
    if adj:
        for i in np.arange(a.shape[0]):
            for j in np.arange(a.shape[0]):
                X = a[i]
                Y = a[j]
                Im[i, j] = adjusted_mutual_info_score(X, Y)
    else:
        for i in np.arange(a.shape[0]):
            for j in np.arange(a.shape[0]):
                X = a[i]
                Y = a[j]
                XY = a[i]*len(scale) + a[j]
                for x in scale:
                    for y in scale:
                        pX = len(X[X == x])/len(X)
                        pY = len(Y[Y == y])/len(Y)
                        pXY = len(XY[XY == x*10+y])/len(XY)
                        Im[i, j] += pXY * np.log(pXY/(pX*pY+eps)+eps)
                H = entropy_from_raw(X) + entropy_from_raw(Y)
                Im[i, j] = (Im[i, j]) / (H/2+eps)
    Im[Im < 0] = 0

    return Im


def dim(M):
    """
        Computes Dimensionality of a Matrix (https://doi.org/10.1016/j.neuron.2017.01.030)
    """

    eigvalM, eigvecM = np.linalg.eig(M)
    dimM = float(np.sum(eigvalM)**2/np.sum(eigvalM**2))

    return dimM

# %%


def bin_avg(a, n_bins):
    """
        Binned Averages of Array where first Dimension represents Time
    """
    a = np.reshape(a, (a.shape[0]//n_bins, n_bins,) + a.shape[1:])
    a = np.mean(a, 1)
    return a


def bin_var(a, n_bins):
    """
        Binned Averages of Array where first Dimension represents Time
    """
    a = np.reshape(a, (a.shape[0]//n_bins, n_bins,) + a.shape[1:])
    a = np.var(a, 1)
    return a
