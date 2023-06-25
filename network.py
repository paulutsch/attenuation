""" unsupervised Hebbian learning of coordinate transformation / multisensory integration task """
""" every modality has its own input with variable gain per trial, except for eye position with fixed gain"""


from dh_kinematic.data import create_input_train, create_input_test
import matplotlib.cm as cm
import os
import numpy as np
import matplotlib.pyplot as mpl
from time import time
from copy import deepcopy
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from Whiten import whiten, whiten2
from projections import FB_inh, FF
from generate_coords import (
    gen_coords3,
    gen_coords_test,
    random_select,
    range_select,
    G_select,
)
from little_helpers import (
    printT,
    datetime,
    popcode_params3,
    popcode_params2,
    gini,
    gauss_generator,
)
from model_definitions import (
    Neurons_inh,
    X_Neurons,
    cov,
    cov_elbw,
    cov_sep,
    cov_sep_elbw,
    oja,
    antihebb,
)
from ANNarchy import (
    Neuron,
    Synapse,
    Population,
    Projection,
    Monitor,
    Network,
    TimedArray,
    setup,
)

setup(num_threads=1)  # larger networks may profit from more threads


# -------------------------- #
# ---  model parameters  --- #
# -------------------------- #

##  network parameters  ##
res_I = [19, 19, 19, 19]
res_I_inh = res_I  # //4+1
res_X = 50
res_X_inh = res_X // 4 + 1

lat_exc_I = True
self_connect_I = True
lat_exc_X = False
self_connect_X = False

learn_sep = True


##  general training parameters  ##
n_trials_train = 300000
T_stim = 100
input_multiplier = 2.5


##  input parameters  ##
# G_train = [0.5, 5]
whiten_inputs = 0
dog_input = 0  # difference of gaussians

FWHM = np.array([(res_I[i] / 8) for i in range(4)])  # target full width at half maximum
sigma_I = np.array([(FWHM[i] / 2.355) for i in range(4)])  # target distribution sigma
# normalization factor ensuring mean input sum of 1
mu_S = np.array([1 / (FWHM[i] / sigma_I[i]) for i in range(4)])


## evaluation parameters ##
n_trials_eval = -1  # -1: number of trials will be computed depending on setup

G_eval = [0, 0.5, 1, 2.75, 5]
pos_eval = np.arange(0, 19)


##  initial connectivity parameters  ##
mu_I = [0.363, 0.0736, 0.0813, 0.331]  # see https://doi.org/10.1101/696088
sigma_I = [x / (2 * np.pi) for x in [0.860, 0.630, 0.569, 0.803]]

mu_ff = [
    0.0,
]
sigma_ff = (
    [
        1,
    ]
    if (whiten_inputs or dog_input)
    else [
        0.3,
    ]
)

mu_X = [0.0, 0.2, 0.2, 0.003]
sigma_X = [0.1, 0.1, 0.2, 0.1]


##  monitor parameters  ##
sample_period_N = 10
sample_period_W = 10000


## left arm joint angles: 0 shoulder yaw, 1 shoulder pitch, 2 shoulder roll, 3 elbow ##
theta_static = np.radians(np.array([90, 90, 0, -90]))
static_side = "left"
eyes_angle = 0


# -------------------------------- #
# ---   custom neuron models   --- #
# -------------------------------- #

Input_Neurons = Neuron(
    name="Input_Neurons",
    parameters="""
        tau = 20 : population
        k = 0.3 : population
        n = 2 : population
    """,
    equations="""
        I_exc = sum(FF) + sum(LAT)
        I_inh = sum(INH)

        tau * dmp/dt + mp = if modulo(t-1, """
    + str(T_stim)
    + """) < tau: 0 else: I_exc - I_inh
        r = if modulo(t-1, """
    + str(T_stim)
    + """) < tau: 0 else: k * mp :"""
    + ("min = 0.0," if not (whiten_inputs or dog_input) else "")
    + """ max = 10.0
    """,
)

# ------------------------------------ #
# ---   customize synapse models   --- #
# ------------------------------------ #

# FF targets
ff_t = ["FF_shrl", "FF_elbw", "FF_angl", "FF_tctl"]

if learn_sep:
    # create unique plasticity rule per projection
    cov_rule = []
    for t in ff_t:
        if t == "FF_elbw":
            H = deepcopy(cov_sep_elbw)
        else:
            H = deepcopy(cov_sep)
        pos = H.equations.find("sep")
        while pos >= 0:
            H.equations = H.equations[:pos] + t + H.equations[pos + 3 :]
            pos = H.equations.find("sep")
        cov_rule.append(H)
else:
    cov_rule = [
        cov,
    ] * 4
    cov_rule[1] = cov_elbw


# -------------------------- #
# ---    model inputs    --- #
# -------------------------- #


# create training data
(
    data_train,
    input_train_shrl,
    input_train_elbw,
    input_train_angl,
    input_train_tctl,
    X_train,
) = create_input_train(
    n_trials_train,
    res_I,
    theta_static,
    static_side,
    eye_angle_degrees=eyes_angle,
    sigma_I=sigma_I,
    ratio_within_forearm_region=1.0,
    plot=False,
    data_path="./dh_kinematic/data",
    shuffle=True,
    create_new_data=False,
    filter=True,
    norm=False,
    plotNorm=False,
)
input_train_shrl *= input_multiplier
input_train_elbw *= input_multiplier
input_train_angl *= input_multiplier
input_train_tctl *= input_multiplier
n_trials_train = input_train_shrl.shape[0]

# create test data
(
    input_test_shrl,
    input_test_elbw,
    input_test_angl,
    input_test_tctl,
    X_test,
) = create_input_test(
    theta_static,
    static_side,
    sigma_I=sigma_I,
    eye_angle_degrees=eyes_angle,
    res=res_I,
    data_path="./dh_kinematic/data",
    create_new_data=False,
    only_within_forearm_region=True,
    filter=True,
    norm=False,
    plotNorm=False,
)
n_trials_eval = input_test_shrl.shape[0]

n_trials = n_trials_train + n_trials_eval

# input = np.vstack((input_train, input_test))
input_shrl = np.vstack((input_train_shrl, input_test_shrl))
input_elbw = np.vstack((input_train_elbw, input_test_elbw))
input_angl = np.vstack((input_train_angl, input_test_angl))
input_tctl = np.vstack((input_train_tctl, input_test_tctl))
if whiten_inputs:
    if whiten_inputs == 1:
        input_shrl = whiten(input_shrl)
        input_elbw = whiten(input_elbw)
        input_angl = whiten(input_angl)
        input_tctl = whiten(input_tctl)
    else:
        input_shrl = whiten2(input_shrl)
        input_elbw = whiten2(input_elbw)
        input_angl = whiten2(input_angl)
        input_tctl = whiten2(input_tctl)
    # input = input / np.max(input) * np.max(G)
    input_shrl = input_shrl / np.max(input_shrl)
    input_elbw = input_elbw / np.max(input_elbw)
    input_angl = input_angl / np.max(input_angl)
    input_tctl = input_tctl / np.max(input_tctl)

    # input_train = input[:n_trials_train]
    # input_test = input[n_trials_train:]

# sensory  modalities
S_shrl = TimedArray(input_shrl, schedule=T_stim, name="S_shrl")
S_elbw = TimedArray(input_elbw, schedule=T_stim, name="S_elbw")
S_angl = TimedArray(input_angl, schedule=T_stim, name="S_angl")
S_tctl = TimedArray(input_tctl, schedule=T_stim, name="S_tctl")


# ---------------------------- #
# ---  neural populations  --- #
# ---------------------------- #

# input populations
I_shrl_exc = Population(res_I[0], Input_Neurons, "I_shrl_exc")
I_elbw_exc = Population(res_I[1], Input_Neurons, "I_elbw_exc")
I_angl_exc = Population(res_I[2], Input_Neurons, "I_angl_exc")
I_tctl_exc = Population(res_I[3], Input_Neurons, "I_tctl_exc")

I_shrl_inh = Population(res_I_inh[0], Input_Neurons, "I_shrl_inh")
I_elbw_inh = Population(res_I_inh[1], Input_Neurons, "I_elbw_inh")
I_angl_inh = Population(res_I_inh[2], Input_Neurons, "I_angl_inh")
I_tctl_inh = Population(res_I_inh[3], Input_Neurons, "I_tctl_inh")

# fusion populations
X_exc = Population(res_X, X_Neurons, "X_exc")
X_inh = Population(res_X_inh, Neurons_inh, "X_inh")


# ----------------------------#
# ---  neural projections  ---#
# ----------------------------#

# sensory input projections
P_S_shrl = Projection(pre=S_shrl, post=I_shrl_exc, target="FF").connect_one_to_one(
    mu_S[0]
)
P_S_elbw = Projection(pre=S_elbw, post=I_elbw_exc, target="FF").connect_one_to_one(
    mu_S[1]
)
P_S_angl = Projection(pre=S_angl, post=I_angl_exc, target="FF").connect_one_to_one(
    mu_S[2]
)
P_S_tctl = Projection(pre=S_tctl, post=I_tctl_exc, target="FF").connect_one_to_one(
    mu_S[3]
)

if not (whiten_inputs or dog_input):
    # recurrent input projections
    P_I_ei_shrl, P_I_ii_shrl, P_I_ie_shrl, P_I_ee_shrl = FB_inh(
        I_shrl_exc, I_shrl_inh, mu_I, sigma_I, lat_exc_I, self_connect_I, None, None
    )
    P_I_ei_elbw, P_I_ii_elbw, P_I_ie_elbw, P_I_ee_elbw = FB_inh(
        I_elbw_exc, I_elbw_inh, mu_I, sigma_I, lat_exc_I, self_connect_I, None, None
    )
    P_I_ei_angl, P_I_ii_angl, P_I_ie_angl, P_I_ee_angl = FB_inh(
        I_angl_exc, I_angl_inh, mu_I, sigma_I, lat_exc_I, self_connect_I, None, None
    )
    P_I_ei_tctl, P_I_ii_tctl, P_I_ie_tctl, P_I_ee_tctl = FB_inh(
        I_tctl_exc, I_tctl_inh, mu_I, sigma_I, lat_exc_I, self_connect_I, None, None
    )

# feedforward projections I-->X
P_FF_shrl = FF(
    I_shrl_exc, X_exc, None, "random", mu_ff, sigma_ff, "FF_shrl", syn_exc=cov_rule[0]
)
P_FF_elbw = FF(
    I_elbw_exc, X_exc, None, "random", mu_ff, sigma_ff, "FF_elbw", syn_exc=cov_rule[1]
)
P_FF_angl = FF(
    I_angl_exc, X_exc, None, "random", mu_ff, sigma_ff, "FF_angl", syn_exc=cov_rule[2]
)
P_FF_tctl = FF(
    I_tctl_exc, X_exc, None, "random", mu_ff, sigma_ff, "FF_tctl", syn_exc=cov_rule[3]
)

# recurrent fusion projections
P_X_ei, P_X_ii, P_X_ie, P_X_ee, (m1, m2, m3) = FB_inh(
    X_exc, X_inh, mu_X, sigma_X, lat_exc_X, self_connect_X, oja, antihebb, random=True
)


# -------------------------- #
# ---      monitors      --- #
# -------------------------- #

## neuron variables ##

M_I_shrl_exc = Monitor((I_shrl_exc), "r", period=sample_period_N)
M_I_elbw_exc = Monitor((I_elbw_exc), "r", period=sample_period_N)
M_I_angl_exc = Monitor((I_angl_exc), "r", period=sample_period_N)
M_I_tctl_exc = Monitor((I_tctl_exc), "r", period=sample_period_N)

M_I_shrl_inh = Monitor((I_shrl_inh), "r", period=sample_period_N)
M_I_elbw_inh = Monitor((I_elbw_inh), "r", period=sample_period_N)
M_I_angl_inh = Monitor((I_angl_inh), "r", period=sample_period_N)
M_I_tctl_inh = Monitor((I_tctl_inh), "r", period=sample_period_N)

M_X_exc = Monitor((X_exc), "r", period=sample_period_N)
M_X_inh = Monitor((X_inh), "r", period=sample_period_N)

## synapse variables ##
M_P_FF_shrl = Monitor(P_FF_shrl, "w", period=sample_period_W)
M_P_FF_elbw = Monitor(P_FF_elbw, "w", period=sample_period_W)
M_P_FF_angl = Monitor(P_FF_angl, "w", period=sample_period_W)
M_P_FF_tctl = Monitor(P_FF_tctl, "w", period=sample_period_W)

M_P_X_ei = Monitor(P_X_ei, "w", period=sample_period_W)
M_P_X_ie = Monitor(P_X_ie, "w", period=sample_period_W)

# --------------------------- #
# ---   network compile   --- #
# --------------------------- #

net = Network(everything=True)
idx = datetime() + str(np.random.randint(10000))
net.compile(directory="net_compile/annarchy_" + idx)
os.popen("cp network.py net_compile/annarchy_" + idx + "/network.py")

print()
total_neurons = 0
for pop in net.populations:
    total_neurons = total_neurons + pop.size
    print(
        "The number of neurons of",
        pop.name,
        "is",
        str(pop.size),
        "with geometry",
        str(pop.geometry),
    )
print("The total number of neurons is", str(total_neurons))


# Synapses
print()
total_connections = 0
for proj in net.projections:
    total_connections = total_connections + proj.nb_synapses
    print(
        "The number of connections from",
        proj.pre.name,
        "to",
        proj.post.name,
        "type",
        proj.target,
        "is",
        str(proj.nb_synapses),
    )
print("The total number of connections is", str(total_connections))


# ---------------------------- #
# ---   network simulate   --- #
# ---------------------------- #

printT("Starting Simulation")
time_start = time()

## train ##
net.simulate(n_trials_train * T_stim)

## test ##
net.disable_learning()
net.simulate(n_trials_eval * T_stim)

time_end = time()
printT("Simulation took", np.round(time_end - time_start, 1), "seconds")
printT(
    "Simulated",
    net.get_current_step(),
    "steps ("
    + str(int(net.get_current_step() / (time_end - time_start) / 1000))
    + "k/s)",
)


# ---------------------------- #
# ---       get data       --- #
# ---------------------------- #

## firing rates of I_i over time: shape (N*T_stim, res_I) ##
r_shrl = net.get(M_I_shrl_exc).get("r")
r_elbw = net.get(M_I_elbw_exc).get("r")
r_angl = net.get(M_I_angl_exc).get("r")
r_tctl = net.get(M_I_tctl_exc).get("r")

## firing rates of X over time: shape (N*T_stim, res_X) ##
r_X = net.get(M_X_exc).get("r")

## weights of I_i -> X over time: shape (N*T_stim, res_I) ##
w_shrl = net.get(M_P_FF_shrl).get("w")
w_elbw = net.get(M_P_FF_elbw).get("w")
w_angl = net.get(M_P_FF_angl).get("w")
w_tctl = net.get(M_P_FF_tctl).get("w")


# ------------------------------ #
# ---   preprocessing data   --- #
# ------------------------------ #

# receptive field: weights of I_i -> X after simulation: shape (res_X, res_I)
RF_shrl = w_shrl[-1]
RF_elbw = w_elbw[-1]
RF_angl = w_angl[-1]
RF_tctl = w_tctl[-1]

# index of maximum weight I_i -> X per X_j: shape (res_X, 1)
argmax_shrl = np.argmax(RF_shrl, 1)
argmax_elbw = np.argmax(RF_elbw, 1)
argmax_angl = np.argmax(RF_angl, 1)
argmax_tctl = np.argmax(RF_tctl, 1)

# value of maximum weight I_i -> X per X_j: shape (res_X, 1)
max_shrl = np.max(RF_shrl, 1)
max_elbw = np.max(RF_elbw, 1)
max_angl = np.max(RF_angl, 1)
max_tctl = np.max(RF_tctl, 1)

# sum of weights I_i -> X per X_j: shape (res_X, 1)
sum_shrl = np.sum(RF_shrl, 1)
sum_elbw = np.sum(RF_elbw, 1)
sum_angl = np.sum(RF_angl, 1)
sum_tctl = np.sum(RF_tctl, 1)

# parameters from gaussian fits
pos_shrl = np.zeros(res_X)
pos_elbw = np.zeros(res_X)
pos_angl = np.zeros(res_X)
pos_tctl = np.zeros(res_X)

width_shrl = np.zeros(res_X)
width_elbw = np.zeros(res_X)
width_angl = np.zeros(res_X)
width_tctl = np.zeros(res_X)

for i in range(res_X):
    pos_shrl[i], width_shrl[i] = popcode_params3(RF_shrl[i])
    pos_elbw[i], width_elbw[i] = popcode_params3(RF_elbw[i])
    pos_angl[i], width_angl[i] = popcode_params3(RF_angl[i])
    pos_tctl[i], width_tctl[i] = popcode_params3(RF_tctl[i])


# construct RFs from activity
# # r_X_ contains every 10th firing rate of hidden layer after training
r_X_test = r_X[9 + (sample_period_N * n_trials_train) :: sample_period_N]
r_X_test_mean = np.mean(r_X_test, 0).T


def RFr(X, n_bins=20):
    # X should be standardized, containing only values between 0 and 1
    x_prev = -0.1

    RFr = np.zeros((res_X, n_bins))

    for i in range(n_bins):
        x = (i + 1) / n_bins

        # save all firing rates of output neuron given input in range x_prev < X <= x
        idx = np.logical_and((x_prev < X), (X <= x))  # (n_trials_test, )

        # mean firing rates of each neuron in hidden layer with respect to inputs in range (filtered using idx)
        RFr[:, i] = np.mean(r_X_test[idx], 0)
        x_prev = x

    return RFr


# RFs for different modalities, each shape (res_X, 20), 20 bins for input range
# X_test contains (N, n_trials_test) values between 0 and 1
RFr_shrl = RFr(X_test[0])
RFr_elbw = RFr(X_test[1])
RFr_angl = RFr(X_test[2])
RFr_tctl = RFr(X_test[3])


# at which input stimulus is max firing rate X_j?
posr_shrl = np.zeros(res_X)
posr_elbw = np.zeros(res_X)
posr_angl = np.zeros(res_X)
posr_tctl = np.zeros(res_X)

# how strongly can preferred input stimulus vary while activity of X_j stays high?
widthr_shrl = np.zeros(res_X)
widthr_elbw = np.zeros(res_X)
widthr_angl = np.zeros(res_X)
widthr_tctl = np.zeros(res_X)
for i in range(res_X):
    posr_shrl[i], widthr_shrl[i] = popcode_params3(RFr_shrl[i])
    posr_elbw[i], widthr_elbw[i] = popcode_params3(RFr_elbw[i])
    posr_angl[i], widthr_angl[i] = popcode_params3(RFr_angl[i])
    posr_tctl[i], widthr_tctl[i] = popcode_params3(RFr_tctl[i])


# ----------------------------- #
# ---       plot data       --- #
# ----------------------------- #

##  inputs ##

mpl.subplot(131)
mpl.title("Mean Input Strength Train")
mpl.plot(np.mean(input_train_shrl, 0).T)
mpl.plot(np.mean(input_train_elbw, 0).T)
mpl.plot(np.mean(input_train_angl, 0).T)
mpl.plot(np.mean(input_train_tctl, 0).T)
mpl.ylim(0)
mpl.subplot(132)
mpl.title("Mean Input Strength Test")
mpl.plot(np.mean(input_test_shrl, 0).T)
mpl.plot(np.mean(input_test_elbw, 0).T)
mpl.plot(np.mean(input_test_angl, 0).T)
mpl.plot(np.mean(input_test_tctl, 0).T)
mpl.ylim(0)
mpl.subplot(133)
mpl.title("Mean Hidden Layer Strength Test")
# mpl.scatter(x_res_X, r_X_mean.T)
# mpl.bar(np.arange(1, res_X + 1), np.mean(r_X, 0).T)
counts, bins = np.histogram(r_X_test_mean, 10)
mpl.stairs(counts, bins)
# mpl.hist(np.arange(1, res_X + 1))
mpl.show()


# inputs_train = np.hstack(
#     (input_train[:, 0], input_train[:, 1], input_train[:, 2], input_train[:, 3])
# )
inputs_train = np.hstack((r_shrl, r_elbw, r_angl, r_tctl))
CI = np.cov(inputs_train.T)

mpl.title("Input Covariance")
mpl.imshow(CI)
mpl.xticks(mpl.yticks()[0][1:-1])
mpl.tight_layout()
mpl.show()


# r_X_ = r_X[9+(T_stim*N_trials_train)//10::10]

# mpl.title("Hidden Layer Sparseness")
# mpl.plot(np.unique(np.sum(G, 0)), sparseness)
# mpl.xlabel("Sum of input gains")
# mpl.show()


# sorts the rows of a similarity matrix in descending order of their total similarity to all other rows, returning the new order as an array of indices.
def argsort_sim_mat(mat):
    # index of the row in 'mat' which has the highest sum, indicating it is most similar to all other rows in the matrix
    idx = [np.argmax(np.sum(mat, axis=1))]

    for _ in range(1, len(mat)):
        # create a copy of the row in 'mat' at the index of the last element in 'idx'
        sm_i = mat[idx[-1]].copy()

        # set the similarity of all previously selected elements to -1 in sm_i, so they won't be selected again
        sm_i[idx] = -1

        # find the index of the next element that is most similar to all remaining elements
        idx.append(np.argmax(sm_i))
    return np.array(idx)


r_X_test = r_X[9 + (T_stim * n_trials_train) // 10 :: 10]
argmax_r_X_test = np.argpartition(r_X_test, 4)


CX = np.cov(r_X_test.T)

X_tctl_ACTIVE = r_X_test[np.mean(input_test_tctl, axis=1) > 0]
X_tctl_INACTIVE = r_X_test[np.mean(input_test_tctl, axis=1) == 0]
CX_tctl_ACTIVE = np.cov(X_tctl_ACTIVE.T)
CX_tctl_INACTIVE = np.cov(X_tctl_INACTIVE.T)

idx_sorted_CX = argsort_sim_mat(CX)
CX_sorted = CX[idx_sorted_CX, :][:, idx_sorted_CX]

idx_sorted_CX_tctl_ACTIVE = argsort_sim_mat(CX_tctl_ACTIVE)
idx_sorted_CX_tctl_INACTIVE = argsort_sim_mat(CX_tctl_INACTIVE)
CX_sorted_tctl_ACTIVE = CX_tctl_ACTIVE[idx_sorted_CX_tctl_ACTIVE, :][
    :, idx_sorted_CX_tctl_ACTIVE
]
CX_sorted_tctl_INACTIVE = CX_tctl_INACTIVE[idx_sorted_CX_tctl_INACTIVE, :][
    :, idx_sorted_CX_tctl_INACTIVE
]

mpl.subplot(121)
mpl.title("Hidden Layer Covariance")
mpl.imshow(CX)

mpl.subplot(122)
mpl.title("Hidden Layer Covariance sorted")
mpl.imshow(CX_sorted)

mpl.show()


mpl.subplot(121)
mpl.title("Hidden Layer Covariance tctl ACTIVE")
mpl.imshow(CX_tctl_ACTIVE)

mpl.subplot(122)
mpl.title("Hidden Layer Covariance tctl ACTIVE sorted")
mpl.imshow(CX_sorted_tctl_ACTIVE)

mpl.show()


mpl.subplot(121)
mpl.title("Hidden Layer Covariance tctl INACTIVE")
mpl.imshow(CX_tctl_INACTIVE)

mpl.subplot(122)
mpl.title("Hidden Layer Covariance tctl INACTIVE sorted")
mpl.imshow(CX_sorted_tctl_INACTIVE)

mpl.show()


idx_outliers_r_X_test = np.argpartition(r_X_test_mean, 2)[0:2]

# shape (2, res_I)
outliers_RF_shrl = RF_shrl[idx_outliers_r_X_test]
outliers_RF_elbw = RF_elbw[idx_outliers_r_X_test]
outliers_RF_angl = RF_angl[idx_outliers_r_X_test]
outliers_RF_tctl = RF_tctl[idx_outliers_r_X_test]

# shape (2, n_bins), default n_bins=20 for range of input
outliers_RFr_shrl_tctl_ACTIVE = RFr_shrl[idx_outliers_r_X_test]
# [np.mean(input_test[:, 3], axis=1) > 0]
outliers_RFr_elbw_tctl_ACTIVE = RFr_elbw[idx_outliers_r_X_test]
outliers_RFr_angl_tctl_ACTIVE = RFr_angl[idx_outliers_r_X_test]
outliers_RFr_tctl_tctl_ACTIVE = RFr_tctl[idx_outliers_r_X_test]

mpl.subplot(141)
mpl.title("HL Outlier 1 RFs shrl")
# mpl.plot(outliers_RF_shrl[0])
mpl.bar(np.arange(1, res_I[0] + 1), outliers_RF_shrl[0])

mpl.subplot(142)
mpl.title("HL Outlier 1 RFs elbw")
# mpl.plot(outliers_RF_elbw[0])
mpl.bar(np.arange(1, res_I[1] + 1), outliers_RF_elbw[0])

mpl.subplot(143)
mpl.title("HL Outlier 1 RFs angl")
# mpl.plot(outliers_RF_angl[0])
mpl.bar(np.arange(1, res_I[2] + 1), outliers_RF_angl[0])

mpl.subplot(144)
mpl.title("HL Outlier 1 RFs tctl")
# mpl.plot(outliers_RF_tctl[0])
mpl.bar(np.arange(1, res_I[3] + 1), outliers_RF_tctl[0])

mpl.show()

mpl.subplot(141)
mpl.title("HL Outlier 2 RFs shrl")
# mpl.plot(outliers_RF_shrl[1])
mpl.bar(np.arange(1, res_I[0] + 1), outliers_RF_shrl[1])

mpl.subplot(142)
mpl.title("HL Outlier 2 RFs elbw")
# mpl.plot(outliers_RF_elbw[1])
mpl.bar(np.arange(1, res_I[1] + 1), outliers_RF_elbw[1])

mpl.subplot(143)
mpl.title("HL Outlier 2 RFs angl")
# mpl.plot(outliers_RF_angl[1])
mpl.bar(np.arange(1, res_I[2] + 1), outliers_RF_angl[1])

mpl.subplot(144)
mpl.title("HL Outlier 2 RFs tctl")
# mpl.plot(outliers_RF_tctl[1])
mpl.bar(np.arange(1, res_I[3] + 1), outliers_RF_tctl[1])

mpl.show()


eigv, eigV = np.linalg.eig(CX)
eigv_I, eigV_I = np.linalg.eig(CI)

# see https://doi.org/10.1016/j.neuron.2017.01.030
dim = np.sum(eigv) ** 2 / np.sum(eigv**2)
dim_I = np.sum(eigv_I) ** 2 / np.sum(eigv_I**2)

mpl.title("Sorted Eigenvalues (dim = " + str(dim)[:4] + ")")
mpl.plot(np.arange(1, res_X + 1), np.sort(eigv)[::-1])
mpl.xticks(np.arange(1, res_X, res_X // 5))
mpl.xlabel("Eigenvector")
mpl.ylabel("Eigenvalue")
mpl.show()

##  convergence of learning  ##

mpl.subplot(221)
mpl.title("Sum of incoming weights (-->X)")
mpl.plot(np.sum(w_shrl, 2))
mpl.ylabel("FF_shrl pathway")

mpl.subplot(222)
mpl.plot(np.sum(w_elbw, 2))
mpl.ylabel("FF_elbw pathway")

mpl.subplot(223)
mpl.plot(np.sum(w_angl, 2))
mpl.ylabel("FF_angl pathway")

mpl.subplot(224)
mpl.plot(np.sum(w_tctl, 2))
mpl.ylabel("FF_tctl pathway")

mpl.tight_layout()
mpl.show()


mpl.subplot(221)
mpl.title("Sum of outgoing weights (I-->)")
mpl.plot(np.sum(w_shrl, 1))
mpl.ylabel("FF_shrl pathway")

mpl.subplot(222)
mpl.plot(np.sum(w_elbw, 1))
mpl.ylabel("FF_elbw pathway")

mpl.subplot(223)
mpl.plot(np.sum(w_angl, 1))
mpl.ylabel("FF_angl pathway")

mpl.subplot(224)
mpl.plot(np.sum(w_tctl, 1))
mpl.ylabel("FF_tctl pathway")

mpl.tight_layout()
mpl.show()

## receptive fields ##

mpl.subplot(411)
mpl.title("Mean Receptive Fields from Weights")
mpl.plot(np.mean(RF_shrl, 0))
mpl.ylabel("FF_shrl pathway")
mpl.ylim(0)

mpl.subplot(412)
mpl.plot(np.mean(RF_elbw, 0))
mpl.ylabel("FF_elbw pathway")
mpl.ylim(0)

mpl.subplot(413)
mpl.plot(np.mean(RF_angl, 0))
mpl.ylabel("FF_angl pathway")
mpl.ylim(0)

mpl.subplot(414)
mpl.plot(np.mean(RF_tctl, 0))
mpl.ylabel("FF_tctl pathway")
mpl.tight_layout()

mpl.show()


mpl.subplot(411)
mpl.title("Sample Receptive Fields from Weights")
mpl.plot(RF_shrl[:5].T)
mpl.ylabel("FF_shrl pathway")

mpl.subplot(412)
mpl.plot(RF_elbw[:5].T)
mpl.ylabel("FF_elbw pathway")

mpl.subplot(413)
mpl.plot(RF_angl[:5].T)
mpl.ylabel("FF_angl pathway")

mpl.subplot(414)
mpl.plot(RF_tctl[:5].T)
mpl.ylabel("FF_tctl pathway")

mpl.tight_layout()

mpl.show()

mpl.subplot(411)
mpl.title("Sample Receptive Fields from Responses")
mpl.plot(RFr_shrl[:4].T)
mpl.ylabel("FF_shrl pathway")

mpl.subplot(412)
mpl.plot(RFr_elbw[:4].T)
mpl.ylabel("FF_elbw pathway")

mpl.subplot(413)
mpl.plot(RFr_angl[:4].T)
mpl.ylabel("FF_angl pathway")

mpl.subplot(414)
mpl.plot(RFr_tctl[:4].T)
mpl.ylabel("FF_tctl pathway")

mpl.tight_layout()

mpl.show()


mpl.title("RF Centers from Weights")
mpl.scatter(
    argmax_shrl + np.random.normal(0, 0.1, argmax_shrl.shape),
    argmax_elbw + np.random.normal(0, 0.1, argmax_elbw.shape),
    alpha=1,
    c=argmax_tctl,
)
mpl.xlabel("FF_shrl pathway")
mpl.ylabel("FF_elbw pathway")
mpl.colorbar().set_label("FF_tctl")

mpl.show()

mpl.title("RF Centers from Weights")
mpl.scatter(
    argmax_angl + np.random.normal(0, 0.1, argmax_angl.shape),
    argmax_elbw + np.random.normal(0, 0.1, argmax_elbw.shape),
    alpha=1,
    c=argmax_tctl,
)
mpl.xlabel("FF_angl pathway")
mpl.ylabel("FF_elbw pathway")
mpl.colorbar().set_label("FF_tctl")

mpl.show()


mpl.title("RF Centers from Responses")
mpl.scatter(posr_shrl, posr_elbw, alpha=1, c=posr_tctl)
mpl.xlabel("FF_shrl pathway")
mpl.ylabel("FF_elbw pathway")
mpl.colorbar().set_label("FF_tctl")

mpl.show()

mpl.title("RF Centers from Responses")
mpl.scatter(posr_angl, posr_elbw, alpha=1, c=posr_tctl)
mpl.xlabel("FF_angl pathway")
mpl.ylabel("FF_elbw pathway")
mpl.colorbar().set_label("FF_tctl")

mpl.show()


mpl.title("RF Strength from Weights")
mpl.scatter(max_shrl, max_elbw, c=max_tctl)
mpl.ylabel("FF_shrl")
mpl.xlabel("FF_elbw")
mpl.colorbar().set_label("FF_tctl")

mpl.show()

mpl.title("RF Strength from Weights")
mpl.scatter(max_angl, max_elbw, c=max_tctl)
mpl.ylabel("FF_angl")
mpl.xlabel("FF_elbw")
mpl.colorbar().set_label("FF_tctl")

mpl.show()


mpl.title("RF Strength from Responses")
mpl.scatter(np.max(RFr_shrl, 1), np.max(RFr_elbw, 1), c=np.max(RFr_tctl, 1))
mpl.ylabel("FF_shrl")
mpl.xlabel("FF_elbw")
mpl.colorbar().set_label("FF_tctl")

mpl.show()

mpl.title("RF Strength from Responses")
mpl.scatter(np.max(RFr_angl, 1), np.max(RFr_elbw, 1), c=np.max(RFr_tctl, 1))
mpl.ylabel("FF_angl")
mpl.xlabel("FF_elbw")
mpl.colorbar().set_label("FF_tctl")

mpl.show()
