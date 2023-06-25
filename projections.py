"""

    * FB_inh
    * FF (can be used as FB)
    * FF_FB

"""

import numpy as np
import matplotlib.pyplot as mpl

from ANNarchy import Projection, Uniform

from model_definitions import cov as HebbTeichmann

from connection_patterns import (connect_gaussian1d,
                                  connect_gaussian2d,
                                  connect_gaussian1dTo2d_v,
                                  connect_gaussian1dTo2d_h,
                                  connect_gaussian1dTo2d_diag,
                                  connect_gaussian2dTo1d_v,
                                  connect_gaussian2dTo1d_h,
                                  connect_gaussian2dTo1d_diag)

from little_helpers import gauss_generator_ring

from operator import xor
from scipy.sparse import lil_matrix

def FB_inh(exc, inh, amp, sigma, x=False, self_connect=False, syn_exc=None, syn_inh=None, syn_rec_inh=None, syn_rec_exc=None, random=False, sparse=False, ring=False):
    """

        Creates a recurrent inhibitory loop (with self-inhibition)
        using a Gaussian kernel - both populations must have the same dimensionality

        * exc: excitatory population
        * inh: inhibitory population(s)
        * amp: amplitude of the Gaussian kernel (1x4 or 4x1 values)
        * sigma: sigma of the Gaussian kernel (1x4 or 4x1 values)
        * x (exc): activates recurrent excitation in excitatory population
        * self_connect: allow lateral self connections
        * syn_exc/inh: excitatory and inhibitory synapse types
        * random: random connections between Xh and Xhi (to stop edge effects)
        * sparse(ness): of random connections
        * ring: assume circular population layout

    """

    if isinstance(amp, (float, int)):
        amp = [amp,]*7 if isinstance(inh, (tuple, list)) else 4
    if isinstance(sigma, (float,int)):
        sigma = [sigma,]*7 if isinstance(inh, (tuple, list)) else 4

    mc = None
    if isinstance(inh, (tuple, list)):

        mc = inh[1]
        inh = inh[0]

        syn_inh_mc = syn_inh[1]
        syn_inh = syn_inh[0]

        Exc_MC = Projection(
            pre=exc, post=mc, target='LAT',
            synapse=syn_exc
        )
        MC_Exc = Projection(
            pre=mc, post=exc, target='INH',
            synapse=syn_inh_mc
        )
        MC_Inh = Projection(
            pre=mc, post=inh, target='INH',
            synapse=syn_inh_mc
        )

    Exc_Inh = Projection(
        pre=exc, post=inh, target='LAT',
        synapse=syn_exc
    )
    Inh_Exc = Projection(
        pre=inh, post=exc, target='INH',
        synapse=syn_inh
    )
    Inh_Inh = Projection(
        pre=inh, post=inh, target='INH',
        synapse=syn_rec_inh
    )

    if random:

        matrix1 = np.array(np.random.uniform(amp[0]-sigma[0], amp[0]+sigma[0], (inh.size, exc.size)), dtype='object')
        matrix1[matrix1<0] = 0
        matrix2 = np.array(np.random.uniform(amp[2]-sigma[2], amp[2]+sigma[2], (exc.size, inh.size)), dtype='object')
        matrix2[matrix2<0] = 0
        matrix3 = np.array(np.random.uniform(amp[1]-sigma[1], amp[1]+sigma[1], (inh.size, inh.size)), dtype='object')
        matrix3[matrix3<0] = 0
        if self_connect == False:
            matrix3[np.eye(inh.size)==1] = None
        if mc is not None:
            matrix4 = np.array(np.random.uniform(amp[4]-sigma[4], amp[4]+sigma[4], (mc.size, exc.size)), dtype='object')
            matrix4[matrix4<0] = 0
            matrix5 = np.array(np.random.uniform(amp[5]-sigma[5], amp[5]+sigma[5], (exc.size, mc.size)), dtype='object')
            matrix5[matrix5<0] = 0
            matrix6 = np.array(np.random.uniform(amp[6]-sigma[6], amp[6]+sigma[6], (inh.size, mc.size)), dtype='object')
            matrix6[matrix6<0] = 0


        if sparse:
            for i in np.arange(inh.size):
                if mc is None:
                    knockouts = np.random.choice(np.arange(exc.size), int(exc.size*(sparse)), replace=False)
                    matrix1[i][knockouts] = None
                    # knockouts = np.random.choice(np.arange(exc.size), int(exc.size*(sparse)), replace=False)
                    # matrix2[knockouts,i] = None
                else:
                    knockouts = np.random.choice(np.arange(exc.size), int(exc.size*(sparse)), replace=False)
                    matrix4[i][knockouts] = None

        Exc_Inh.connect_from_matrix(matrix1)
        Inh_Exc.connect_from_matrix(matrix2)
        Inh_Inh.connect_from_matrix(matrix3)
        if mc is not None:
            Exc_MC.connect_from_matrix(matrix4)
            MC_Exc.connect_from_matrix(matrix5)
            MC_Inh.connect_from_matrix(matrix6)

    elif ring:
        matrix1 = np.zeros((inh.size, exc.size), dtype='object')
        for i, mu in enumerate(np.linspace(0, exc.size-1, inh.size+1)[:-1]):
            matrix1[i] = gauss_generator_ring(amp[0], mu, sigma[0]*exc.size, exc.size)
        matrix1[matrix1<0.001] = None

        matrix2 = np.zeros((exc.size, inh.size), dtype='object')
        for i, mu in enumerate(np.linspace(0, inh.size-1, exc.size+1)[:-1]):
            matrix2[i] = gauss_generator_ring(amp[2], mu, sigma[2]*inh.size, inh.size)
        matrix2[matrix2<0.001] = None

        matrix3 = np.zeros((inh.size, inh.size), dtype='object')
        for i, mu in enumerate(np.linspace(0, inh.size-1, inh.size+1)[:-1]):
            matrix3[i] = gauss_generator_ring(amp[1], mu, sigma[1]*inh.size, inh.size)
        matrix3[matrix3<0.001] = None

        Exc_Inh.connect_from_matrix(matrix1)
        Inh_Exc.connect_from_matrix(matrix2)
        Inh_Inh.connect_from_matrix(matrix3)

    else:
        Exc_Inh.connect_gaussian(amp=amp[0], sigma=sigma[0])
        Inh_Exc.connect_gaussian(amp=amp[2], sigma=sigma[2])
        Inh_Inh.connect_gaussian(amp=amp[1], sigma=sigma[1], allow_self_connections=self_connect)

    if x:
        Exc_Exc = Projection(
            pre=exc, post=exc, target='LAT',
            synapse=syn_rec_exc
        )
        if random:
            Exc_Exc.connect_all_to_all(weights=Uniform(amp[3]-sigma[3], amp[3]+sigma[3]), allow_self_connections=self_connect)
        elif ring:
            matrix4 = np.zeros((exc.size, exc.size), dtype='object')
            for i, mu in enumerate(np.linspace(0, exc.size-1, exc.size+1)[:-1]):
                matrix4[i] = gauss_generator_ring(amp[3], mu, sigma[3]*exc.size, exc.size)
            matrix4[matrix4<0.001] = None
            Exc_Exc.connect_from_matrix(matrix4)
        else:
            Exc_Exc.connect_gaussian(amp=amp[3], sigma=sigma[3], allow_self_connections=self_connect)

    else:
        Exc_Exc = None

    if mc is None:
        if random or ring:
            return Exc_Inh, Inh_Inh, Inh_Exc, Exc_Exc, (matrix1, matrix2, matrix3)
        else:
            return Exc_Inh, Inh_Inh, Inh_Exc, Exc_Exc
    else:
        if random:
            return Exc_Inh, Inh_Inh, Inh_Exc, Exc_Exc, Exc_MC, MC_Inh, MC_Exc, (matrix1, matrix2, matrix3, matrix4, matrix5, matrix6)
        else:
            return Exc_Inh, Inh_Inh, Inh_Exc, Exc_Exc, Exc_MC, MC_Inh, MC_Exc


def FF(pre_exc, post_exc, post_inh, mode, amp, sigma=1, target='FF_e', orientation='x', scale=0, syn_exc=None, syn_exc2='None', matrix_path='FF_weights.npy', sparse=False):
    """

        Creates unidirectional projections (with feedforward-inhibition)

        * pre_exc: excitatory pre-population
        * post_exc: excitatory post-population
        * post_inh: inhibitory post-population
        * mode: whether to use random, geometric gaussian, 1to1 or pre-defined weights
        * amp: amplitude of the Gaussian kernel (1x2 or 2x1 values)
        * sigma: sigma of the Gaussian kernel (1x2 or 2x1 values)
        * target: Feedback or Feedforward
        * orientation: relative geometric orientation of the populations
        * syn_exc: Synapse Type
        * matrix_path: path to connectivity matrix if mode=='m'
        * sparse(ness): of random connections

    """

    pre_post, pre_post_inh = None, None

    if isinstance(amp, (float,int)):
        amp = [amp,amp]
    if isinstance(sigma, (float,int)):
        sigma = [sigma,sigma]
    if isinstance(scale, (float,int)):
        scale = [scale,scale]

    if syn_exc2 is 'None':
        syn_exc2 = syn_exc


    pre_post = Projection(
        pre=pre_exc, post=post_exc, target=target,
        synapse=syn_exc
    )

    if post_inh is not None:
        pre_post_inh = Projection(
            pre=pre_exc, post=post_inh, target=target,
            synapse=syn_exc2
        )

    if mode=='1to1' or mode=='1':
        print("connecting 1to1: "+pre_exc.name+"-->"+post_exc.name+" ("+target+")\nW:", amp)
        pre_post.connect_one_to_one(amp[0])
        if post_inh is not None:
            pre_post_inh.connect_all_to_all(amp[1])

    elif mode=='random' or mode=='r':
        print("connecting randomly: "+pre_exc.name+"-->"+post_exc.name+" ("+target+")\nMean:", amp, "\nSD  :", sigma)

        matrix = np.array(np.random.uniform(amp[0]-sigma[0], amp[0]+sigma[0], (post_exc.size, pre_exc.size)), dtype='object')
        matrix[matrix<0] = 0
        if sparse:
            for i in np.arange(pre_exc.size):
                knockouts = np.random.choice(np.arange(post_exc.size), int(post_exc.size*(sparse)), replace=False)
                matrix[knockouts, i] = None
        pre_post.connect_from_matrix(matrix)

        if post_inh is not None:
            matrix = np.array(np.random.uniform(amp[1]-sigma[1], amp[1]+sigma[1], (post_inh.size, pre_exc.size)), dtype='object')
            matrix[matrix<0] = 0
            pre_post_inh.connect_from_matrix(matrix)

    elif mode=='gauss' or mode=='g':
        if len(pre_exc.geometry)==1 and len(post_exc.geometry)==1:

            pre_post.connect_gaussian(amp=amp[0], sigma=sigma[0], delays = 2.0)
            if post_inh is not None:
                pre_post_inh.connect_gaussian(amp=amp[1], sigma=sigma[1])

        elif xor(len(pre_exc.geometry)==2, len(post_exc.geometry)==2):
            if orientation in ['v', 'vertikal']:
                if pre_exc.dimension==2:
                    FF_connector = connect_gaussian2dTo1d_v
                else:
                    FF_connector = connect_gaussian1dTo2d_v
            elif orientation in ['h', 'horizontal']:
                if pre_exc.dimension==2:
                    FF_connector = connect_gaussian2dTo1d_h
                else:
                    FF_connector = connect_gaussian1dTo2d_h
            elif orientation in ['d', 'diagonal']:
                if pre_exc.dimension==2:
                    FF_connector = connect_gaussian2dTo1d_diag
                else:
                    FF_connector = connect_gaussian1dTo2d_diag
            else:
                print("Using all_to_all connectivity?")

            pre_post.connect_with_func(method=FF_connector, mv=amp[0], radius=sigma[0], mgd=0, delay=2.0, scale=scale[0])
            if post_inh is not None:
                pre_post_inh.connect_with_func(method=FF_connector, mv=amp[1], radius=sigma[1], mgd=0, scale=scale[1])

    elif mode=='matrix' or mode=='m':

        print("connecting from matrix: "+pre_exc.name+"-->"+post_exc.name+" ("+target+")\n", matrix_path)

        matrix_dict = {'v': 0, 'h': 1, 'd':2}
        i = matrix_dict[orientation]

        matrix = np.load(matrix_path, allow_pickle=True)
        if 'FF' in target:
            pre_post.connect_from_matrix(matrix[i], delays = 2.0)
            if post_inh is not None:
                pre_post_inh.connect_from_matrix(matrix[i, np.random.choice(matrix.shape[1], post_inh.size), :])
        else:
            pre_post.connect_from_matrix(matrix[i].T, delays = 2.0)
            if post_inh is not None:
                pre_post_inh.connect_from_matrix(matrix[i, :,  np.linspace(0, matrix.shape[2]-1, post_inh.size, dtype='int')])

    else:

        print("[Warning] No synapses created!")

    if post_inh is None:
        return pre_post
    else:
        return pre_post, pre_post_inh


def FF_FB(pre_exc, pre_inh, post_exc, post_inh, orientation, random, scale, amp, sigma):
    """

        Creates a recurrent excitatory loop (with feedforward-inhibition)

        * pre_exc: excitatory pre-population
        * pre_inh: inhibitory pre-population
        * post_exc: excitatory post-population
        * post_inh: inhibitory post-population
        * orientation: relative geometric orientation of the populations
        * random: whether to use random feedforward and feedback connecions
        * amp: amplitude of the Gaussian kernel (1x4 or 4x1 values)
        * sigma: sigma of the Gaussian kernel (1x4 or 4x1 values)

    """

    pre_post, pre_post_inh, post_pre, post_pre_inh = None, None, None, None

    if isinstance(amp, (float,int)):
        amp = [amp,amp,amp,amp]
    if isinstance(sigma, (float,int)):
        sigma = [sigma,sigma,sigma,sigma]
    if isinstance(scale, (float,int)):
        scale = [scale,scale,scale,scale]

    if amp[0]:
        pre_post = Projection(
            pre=pre_exc, post=post_exc, target='FF_e',
            synapse=HebbTeichmann
        )
    if amp[1]:
        pre_post_inh = Projection(
            pre=pre_exc, post=post_inh, target='FF_e',
            synapse=HebbTeichmann
        )
    if amp[2]:
        post_pre = Projection(
            pre=post_exc, post=pre_exc, target='FB_e',
            synapse=HebbTeichmann
        )
    if amp[3]:
        post_pre_inh = Projection(
            pre=post_exc, post=pre_inh, target='FB_e',
            synapse=HebbTeichmann
        )

    if random:
        print("connecting randomly:\nMean:", amp, "\nSD  :", sigma)
        pre_post.connect_all_to_all(weights=Uniform(amp[0]-sigma[0], amp[0]+sigma[0]), delays = 2.0)
        pre_post_inh.connect_all_to_all(weights=Uniform(amp[1]-sigma[1], amp[1]+sigma[1]))
        post_pre.connect_all_to_all(weights=Uniform(amp[2]-sigma[2], amp[2]+sigma[2]), delays = 2.0)
        post_pre_inh.connect_all_to_all(weights=Uniform(amp[3]-sigma[3], amp[3]+sigma[3]))

    else:
        if len(pre_exc.geometry)==1 and len(post_exc.geometry)==1:

            pre_post.connect_gaussian(amp=amp[0], sigma=sigma[0], delays = 2.0)
            pre_post_inh.connect_gaussian(amp=amp[1], sigma=sigma[1])
            post_pre.connect_gaussian(amp=amp[2], sigma=sigma[2], delays = 2.0)
            post_pre_inh.connect_gaussian(amp=amp[3], sigma=sigma[3])

        elif xor(len(pre_exc.geometry)==2, len(post_exc.geometry)==2):
            if orientation in ['v', 'vertikal']:
                FF_connector = connect_gaussian1dTo2d_v
                FB_connector = connect_gaussian2dTo1d_v
            elif orientation in ['h', 'horizontal']:
                FF_connector = connect_gaussian1dTo2d_h
                FB_connector = connect_gaussian2dTo1d_h
            elif orientation in ['d', 'diagonal']:
                FF_connector = connect_gaussian2dTo1d_diag
                FB_connector = connect_gaussian1dTo2d_diag
            else:
                print("Using all_to_all connectivity?")

            pre_post.connect_with_func(method=FF_connector, mv=amp[0], radius=sigma[0], mgd=1, delay=2.0, scale=scale[0])
            pre_post_inh.connect_with_func(method=FF_connector, mv=amp[1], radius=sigma[1], mgd=0, scale=scale[1])
            post_pre.connect_with_func(method=FB_connector, mv=amp[2], radius=sigma[2], mgd=1, delay=2.0, scale=scale[2])
            post_pre_inh.connect_with_func(method=FB_connector, mv=amp[3], radius=sigma[3], mgd=0, scale=scale[3])


    return pre_post, pre_post_inh, post_pre, post_pre_inh