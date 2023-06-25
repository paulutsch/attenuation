"""
    Contents:

        * Input_Neurons
        * Input_Neurons_quad_inh
        * Gain_Neurons
        * Inhibition_Neurons
        * Inhibition_Neurons_2D

        * HebbTeichmann
        * HebbBrunel
        * Antihebb
        * AntihebbSimple

"""

from ANNarchy import Neuron, Synapse, add_function

# ----------------------------#
# ---   custom functions   ---#
# ----------------------------#

add_function("cutoff(x, threshold) = if x<threshold: 0 else: x")
# a = 1.2 b = 2
add_function("logistic(x, a, b) = a * (2 / (1. + exp(-b*x)) - 1)")
add_function("loglinear(x) = x + (1 / (1. + exp(-10.*x)) - 0.5) / (x+1)^2")

# add_function("sign(x) = if x < 0: -1 else: 1")

# ---------------------------#
# ---    neuron models    ---#
# ---------------------------#

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

        tau * dmp/dt + mp = I_exc - I_inh
        r = k * mp : 'min = 0.0, max = 10.0
    """,
)

Neurons_inh = Neuron(
    name="Neurons_inh",
    parameters="""
        tau = 10 : population
        k = 0.3 : population
        n = 2 : population
    """,
    equations="""
        I_exc = sum(FF) + sum(LAT)
        I_inh = sum(INH)

        tau * dmp/dt + mp = I_exc - I_inh
        r = k * mp : min = 0.0, max = 10.0
    """,
)

X_Neurons = Neuron(
    name="X_Neurons",
    parameters="""
        tau = 20 : population
        k = 0.3 : population
        n = 2 : population
    """,
    equations="""
        FF_shrl = sum(FF_shrl)
        FF_elbw = sum(FF_elbw)
        FF_angl = sum(FF_angl)
        FF_tctl = sum(FF_tctl)
        I_exc = FF_shrl + FF_elbw + FF_angl + FF_tctl + sum(LAT)
        I_inh = sum(INH)

        tau * dmp/dt + mp = I_exc - I_inh
        r = k * mp : min = 0.0, max = 10.0
    """,
)

# ---------------------------#
# ---   synapse modells   ---#
# ---------------------------#

cov = Synapse(
    name="cov",
    parameters="""
        tau_W = 1500 : projection
        tau_alpha = 1000 : projection
        alpha_minus = 0.0005 : projection
    """,
    equations="""
        tau_alpha * dalpha/dt = - alpha_minus + pos(post.I_exc - 0.6)^2 : min = 0.01, init = 0.1
        tau_W * dw/dt = (pre.r - mean(pre.r)) * post.r - alpha * post.r^2 * w : min = 0.0
    """,
)

cov_elbw = Synapse(
    name="cov_elbw",
    parameters="""
        tau_W = 750 : projection
        tau_alpha = 1000 : projection
        alpha_minus = 0.0005 : projection
    """,
    equations="""
        tau_alpha * dalpha/dt = - alpha_minus + pos(post.I_exc - 0.6)^2 : min = 0.01, init = 0.1
        tau_W * dw/dt = (pre.r - mean(pre.r)) * post.r - alpha * post.r^2 * w : min = 0.0
    """,
)

cov_sep = Synapse(
    name="cov_sep",
    parameters="""
        tau_W = 3000 : projection
        tau_alpha = 1000 : projection
        alpha_minus = 0.0005 : projection
    """,
    equations="""
        tau_alpha * dalpha/dt = - alpha_minus + pos(post.sep - 0.2)^2 : min = 0.01, init = 0.1
        tau_W * dw/dt = (pre.r - mean(pre.r)) * post.r - alpha * post.sep^2 * w : min = 0.0
    """,
)

cov_sep_elbw = Synapse(
    name="cov_sep_elbw",
    parameters="""
        tau_W = 1500 : projection
        tau_alpha = 1000 : projection
        alpha_minus = 0.0005 : projection
    """,
    equations="""
        tau_alpha * dalpha/dt = - alpha_minus + pos(post.sep - 0.2)^2 : min = 0.01, init = 0.1
        tau_W * dw/dt = (pre.r - mean(pre.r)) * post.r - alpha * post.sep^2 * w : min = 0.0
    """,
)

oja = Synapse(
    name="oja",
    parameters="""
        tau_W = 3000 : projection
    """,
    equations="""
        tau_W * dw/dt = pre.r * post.r - 0.1 * post.r^2 * w : min = 0.0
    """,
)

antihebb = Synapse(
    name="antihebb",
    parameters="""
        tau_W = 2000 : projection
        alpha = 0.05
        th_singleside = 0.01 : projection
    """,
    equations="""
        tau_W * dw/dt = pre.r * post.r - alpha * (post.r + th_singleside) * w : min = 0.0
    """,
)

vogels = Synapse(
    parameters="""
		tau_W = 2000 : projection
        alpha = 0.05  : projection
	""",
    equations="""
    	tau_W * dw/dt = pre.r * post.r - pre.r * alpha : min = 0.0
	""",
)


miehl_exc = Synapse(
    parameters="""
		tau_W = 2000 : projection
        alpha = 1  : projection
	""",
    equations="""
    	tau_W * dw/dt = pre.r * post.r * (post.r - alpha) : min = 0.0
	""",
)

miehl_inh = Synapse(
    parameters="""
		tau_W = 2000 : projection
        alpha = 0.05  : projection
	""",
    equations="""
    	tau_W * dw/dt = pre.r * post.r * (post.r - alpha) : min = 0.0
	""",
)
