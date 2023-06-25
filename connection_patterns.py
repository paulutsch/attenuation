"""
@author: juschu

defing own connection pattern
"""

import math
import time
import numpy as np

from ANNarchy import CSR

# width = geometry[0], height = geometry[1]

saveConnections = False
if saveConnections:
    # get directory for saving
    import globalParams
    import os
    saveDir = globalParams.saveDir + "connections/"
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

MIN_CONNECTION_VALUE = 0.001

############################
#### Connection pattern ####
############################


def connect_gaussian1d(pre, post, mv, radius, mgd, delay=0):
    '''
    connect two maps with an 1-dimensional gaussian receptive field

    params: pre, post -- layers, that should be connected
            mv        -- highest value for one connection
            radius    -- width of receptive field (in deg)
            mgd       -- width of receptive field (in number of neurons)

    return: synapses  -- CSR-object with connections
    '''

    time0 = time.time()

    prW = pre.width
    poW = post.width

    print("\ngaussian1d:", mv, radius, mgd,
          "(connecting", pre.name, "to", post.name, ")")

    # Normalization along width of sigma values on afferent map
    if prW % 2 == 1:
        print("ATTENTION: this is a bit fishy")
        sigma_w = radius * (prW-1)
    else:
        sigma_w = radius * prW

    # Ratio of size between maps
    ratio_w = (prW-1)/float(poW-1)
    offset_w = 0

    synapse = CSR()

    # for speedup
    m_exp = math.exp

    if saveConnections:
        fn = pre.name + "_" + post.name + "_" + "gaussian1d"
        strToWrite = "connect " + pre.name + " to " + \
            post.name + " with gaussian1d" + "\n\n"

    for w_post in range(poW):

        values = []
        pre_ranks = []

        for w_pre in range(prW):

            saveVal = 0

            if (pre != post) or (w_post != w_pre):
                dist_w = (ratio_w*w_post-w_pre+offset_w/2.0)**2
                if (mgd == 0) or (dist_w < mgd*mgd):
                    val = mv * m_exp(-dist_w/sigma_w/sigma_w)
                    if val > mv/10.0:
                        # connect
                        values.append(val)
                        pre_ranks.append(w_pre)

                        saveVal = val

            if saveConnections:
                strToWrite += "(" + str(w_pre) + ") -> (" + \
                    str(w_post) + ") with " + str(saveVal) + "\n"

        synapse.add(w_post, pre_ranks, values, [delay])

    if saveConnections:
        f = open(saveDir+fn+'.txt', 'w')
        f.write(strToWrite)
        f.close()

    time1 = time.time()
    print('created', synapse.nb_synapses, 'synapses out of ',
          poW*prW, ' in', time1-time0, 's')
    return synapse


def connect_gaussian2d(pre, post, mv, radius, mgd, delay=0):
    '''
    connect two maps with a 2-dimensional gaussian receptive field

    params: pre, post -- layers, that should be connected
            mv        -- highest value for one connection
            radius    -- width of receptive field (in deg)
            mgd       -- width of receptive field (in number of neurons)

    return: synapses  -- CSR-object with connections
    '''

    time0 = time.time()

    prW = pre.width
    prH = pre.height
    poW = post.width
    poH = post.height

    print("\ngaussian2d:", mv, radius, mgd,
          "(connecting", pre.name, "to", post.name, ")")

    # Normalization along width of sigma values on afferent map
    if prW % 2 == 1 or prH % 2 == 1:
        print("ATTENTION: this is a bit fishy")
        sigma_w = radius * (prW-(prW % 2))
        sigma_h = radius * (prH-(prH % 2))
    else:
        sigma_w = radius * prW
        sigma_h = radius * prH

    # Ratio of size between maps
    ratio_w = prW/float(poW)
    ratio_h = prH/float(poH)
    offset_w = ratio_w*poW - prW
    offset_h = ratio_h*poH-prH

    synapse = CSR()

    # for speedup
    m_exp = math.exp

    if saveConnections:
        fn = pre.name + "_" + post.name + "_" + "gaussian2d"
        strToWrite = "connect " + pre.name + " to " + \
            post.name + " with gaussian2d" + "\n\n"

    for w_post in range(poW):
        for h_post in range(poH):

            post_rank = post.rank_from_coordinates((w_post, h_post))

            values = []
            pre_ranks = []

            for w_pre in range(prW):
                for h_pre in range(prH):

                    saveVal = 0

                    if (pre != post) or (w_post != w_pre) or (h_post != h_pre):
                        dist_w = (ratio_w*w_post-w_pre+offset_w/2.0)**2
                        dist_h = (ratio_h*h_post-h_pre+offset_h/2.0)**2
                        if (mgd == 0) or ((dist_w < mgd*mgd) and (dist_h < mgd*mgd)):
                            val = mv * \
                                m_exp(-(dist_w/sigma_w/sigma_w +
                                      dist_h/sigma_h/sigma_h))
                            if val > mv/10.0:
                                # connect
                                pre_rank = pre.rank_from_coordinates(
                                    (w_pre, h_pre))
                                values.append(val)
                                pre_ranks.append(pre_rank)

                                saveVal = val

                    if saveConnections:
                        strToWrite += "(" + str(w_pre) + "," + str(h_pre) + ") -> (" + str(
                            w_post) + "," + str(h_post) + ") with " + str(saveVal) + "\n"

            synapse.add(post_rank, pre_ranks, values, [delay])

    if saveConnections:
        f = open(saveDir+fn+'.txt', 'w')
        f.write(strToWrite)
        f.close()

    time1 = time.time()
    print('created', synapse.nb_synapses, 'synapses out of ',
          poW*prW*poH*prH, ' in', time1-time0, 's')
    return synapse


def connect_gaussian1dTo2d_v(pre, post, mv, radius, mgd, delay=0, scale=0):
    '''
    connect two maps with a gaussian receptive field 1d to 2d
    independent of first dimension of 2d map

    params: pre, post -- layers, that should be connected
            mv        -- highest value for one connection
            radius    -- width of receptive field (in deg)
            mgd       -- width of receptive field (in number of neurons)

    return: synapses  -- CSR-object with connections
    '''

    time0 = time.time()

    prW = pre.width
    poW = post.width
    poH = post.height

    print("\ngaussian1dTo2d_v:", mv, radius, mgd,
          "(connecting", pre.name, "to", post.name, ")")

    # Normalization along width of sigma values on afferent map
    if prW % 2 == 1:
        print("ATTENTION: this is a bit fishy")
        sigma = radius * (prW-1)
    else:
        sigma = radius * prW

    # Ratio of size between maps
    ratio = (prW-1)/float(poH-1)
    offset = 0

    synapse = CSR()

    # for speedup
    m_exp = math.exp

    if saveConnections:
        fn = pre.name + "_" + post.name + "_" + "gaussian1dTo2d_v"
        strToWrite = "connect " + pre.name + " to " + \
            post.name + " with gaussian1dTo2d_v" + "\n\n"

    for w_post in range(poW):
        for h_post in range(poH):

            post_rank = post.rank_from_coordinates((w_post, h_post))

            values = []
            pre_ranks = []

            for h_pre in range(prW):

                saveVal = 0

                if (pre != post) or (h_post != h_pre):
                    dist = (ratio*h_post-h_pre+offset/2.0)**2
                    if (mgd == 0) or (dist < mgd*mgd):
                        val = mv * m_exp(-dist/sigma/sigma)
                        if val > MIN_CONNECTION_VALUE:
                            # connect
                            values.append(val)
                            pre_ranks.append(h_pre)

                            saveVal = val

                if saveConnections:
                    strToWrite += "(" + str(h_pre) + ") -> (" + str(w_post) + \
                        "," + str(h_post) + ") with " + str(saveVal) + "\n"

            synapse.add(post_rank, pre_ranks, values, [delay])

    if saveConnections:
        f = open(saveDir+fn+'.txt', 'w')
        f.write(strToWrite)
        f.close()

    time1 = time.time()
    print('created', synapse.nb_synapses, 'synapses out of ',
          poW*prW*poH, ' in', time1-time0, 's')
    return synapse


def connect_gaussian1dTo2d_h(pre, post, mv, radius, mgd, delay=0, scale=0):
    '''
    connect two maps with a gaussian receptive field 1d to 2d
    independent of last dimension of 2d map

    params: pre, post -- layers, that should be connected
            mv        -- highest value for one connection
            radius    -- width of receptive field (in deg)
            mgd       -- width of receptive field (in number of neurons)

    return: synapses  -- CSR-object with connections
    '''

    time0 = time.time()

    prW = pre.width
    poW = post.width
    poH = post.height

    print("\ngaussian1dTo2d_h:", mv, radius, mgd,
          "(connecting", pre.name, "to", post.name, ")")

    # Normalization along width of sigma values on afferent map
    if prW % 2 == 1:
        print("ATTENTION: this is a bit fishy")
        sigma = radius * (prW-1)
    else:
        sigma = radius * prW

    # Ratio of size between maps
    ratio = (prW-1)/float(poW-1)
    offset = 0

    synapse = CSR()

    # for speedup
    m_exp = math.exp

    if saveConnections:
        fn = pre.name + "_" + post.name + "_" + "gaussian1dTo2d_h"
        strToWrite = "connect " + pre.name + " to " + \
            post.name + " with gaussian1dTo2d_h" + "\n\n"

    for w_post in range(poW):
        for h_post in range(poH):

            post_rank = post.rank_from_coordinates((w_post, h_post))

            values = []
            pre_ranks = []

            for w_pre in range(prW):

                saveVal = 0

                if (pre != post) or (w_post != w_pre):
                    dist = (ratio*w_post - w_pre + offset/2.0)**2
                    if (mgd == 0) or (dist < mgd*mgd):
                        val = mv * m_exp(-dist/sigma/sigma)
                        if val > MIN_CONNECTION_VALUE:
                            # connect
                            values.append(val)
                            pre_ranks.append(w_pre)

                            saveVal = val

                if saveConnections:
                    strToWrite += "(" + str(w_pre) + ") -> (" + str(w_post) + \
                        "," + str(h_post) + ") with " + str(saveVal) + "\n"

            synapse.add(post_rank, pre_ranks, values, [delay])

    if saveConnections:
        f = open(saveDir+fn+'.txt', 'w')
        f.write(strToWrite)
        f.close()

    time1 = time.time()
    print('created', synapse.nb_synapses, 'synapses out of ',
          poW*prW*poH, ' in', time1-time0, 's')
    return synapse


def connect_gaussian2dTo1d_v(pre, post, mv, radius, mgd, delay=0, scale=0):
    '''
    @Valentin
    connect two maps with a gaussian receptive field 2d to 1d
    independent of first dimension of 2d map

    params: pre, post -- layers, that should be connected
            mv        -- highest value for one connection
            radius    -- width of receptive field (in deg)
            mgd       -- width of receptive field (in number of neurons)

    return: synapses  -- CSR-object with connections
    '''

    time0 = time.time()

    prW = pre.width
    prH = pre.height
    poW = post.width

    print("\ngaussian2dTo1d_v:", mv, radius, mgd,
          "(connecting", pre.name, "to", post.name, ")")

    # Normalization along width of sigma values on afferent map
    if prH % 2 == 1:
        print("ATTENTION: this is a bit fishy")
        sigma = radius * (prH-1)
    else:
        sigma = radius * prH

    # Ratio of size between maps
    ratio = (prH-1)/float(poW-1)
    offset = 0

    synapse = CSR()

    # for speedup
    m_exp = math.exp

    if saveConnections:
        fn = pre.name + "_" + post.name + "_" + "gaussian2dTo1d_h"
        strToWrite = "connect " + pre.name + " to " + \
            post.name + " with gaussian2dTo1d_h" + "\n\n"

    for w_post in range(poW):

        values = []
        pre_ranks = []

        for h_pre in range(prH):

            dist = (ratio*w_post - h_pre + offset/2.0)**2
            if (mgd == 0) or (dist < mgd*mgd):
                val = mv * m_exp(-dist/sigma/sigma)
                if val > MIN_CONNECTION_VALUE:

                    for w_pre in range(prW):

                        # connect
                        pre_rank = pre.rank_from_coordinates((w_pre, h_pre))
                        values.append(val)
                        pre_ranks.append(pre_rank)

                        if saveConnections:
                            strToWrite += "(" + str(w_pre) + "," + str(h_pre) + \
                                ") -> (" + str(w_post) + \
                                ") with " + str(val) + "\n"

        synapse.add(w_post, pre_ranks, values, [delay])

    if saveConnections:
        f = open(saveDir+fn+'.txt', 'w')
        f.write(strToWrite)
        f.close()

    time1 = time.time()
    print('created', synapse.nb_synapses, 'synapses out of ',
          poW*prW*prH, ' in', time1-time0, 's')

    return synapse


def connect_gaussian2dTo1d_h(pre, post, mv, radius, mgd, delay=0, scale=0):
    '''
    connect two maps with a gaussian receptive field 2d to 1d
    independent of last dimension of 2d map

    params: pre, post -- layers, that should be connected
            mv        -- highest value for one connection
            radius    -- width of receptive field (in deg)
            mgd       -- width of receptive field (in number of neurons)

    return: synapses  -- CSR-object with connections
    '''

    time0 = time.time()

    prW = pre.width
    prH = pre.height
    poW = post.width

    print("\ngaussian2dTo1d_h:", mv, radius, mgd,
          "(connecting", pre.name, "to", post.name, ")")

    # Normalization along width of sigma values on afferent map
    if prW % 2 == 1:
        print("ATTENTION: this is a bit fishy")
        sigma = radius * (prW-1)
    else:
        sigma = radius * prW

    # Ratio of size between maps
    ratio = (prW-1)/float(poW-1)
    offset = 0

    synapse = CSR()

    # for speedup
    m_exp = math.exp

    if saveConnections:
        fn = pre.name + "_" + post.name + "_" + "gaussian2dTo1d_h"
        strToWrite = "connect " + pre.name + " to " + \
            post.name + " with gaussian2dTo1d_h" + "\n\n"

    for w_post in range(poW):

        values = []
        pre_ranks = []

        for w_pre in range(prW):

            dist = (ratio*w_post - w_pre + offset/2.0)**2
            if (mgd == 0) or (dist < mgd*mgd):
                val = mv * m_exp(-dist/sigma/sigma)
                if val > MIN_CONNECTION_VALUE:

                    for h_pre in range(prH):

                        # connect
                        pre_rank = pre.rank_from_coordinates((w_pre, h_pre))
                        values.append(val)
                        pre_ranks.append(pre_rank)

                        if saveConnections:
                            strToWrite += "(" + str(w_pre) + "," + str(h_pre) + \
                                ") -> (" + str(w_post) + \
                                ") with " + str(val) + "\n"

        synapse.add(w_post, pre_ranks, values, [delay])

    if saveConnections:
        f = open(saveDir+fn+'.txt', 'w')
        f.write(strToWrite)
        f.close()

    time1 = time.time()
    print('created', synapse.nb_synapses, 'synapses out of ',
          poW*prW*prH, ' in', time1-time0, 's')

    return synapse


def connect_gaussian2dTo1d_diag(pre, post, mv, radius, mgd, delay=0, scale=0):
    '''
    connect two maps with a gaussian receptive field 2d to 1d diagonally

    params: pre, post -- layers, that should be connected
            mv        -- highest value for one connection
            radius    -- width of receptive field (in deg)
            mgd       -- width of receptive field (in number of neurons)
            delay     -- delay of projection
            scale     -- scaling factor for relative population size

    return: synapses  -- CSR-object with connections
    '''

    time0 = time.time()

    prW = pre.width
    prH = pre.height
    poW = post.width

    if scale:
        ratio = scale*(prW+prH-2)/float(poW-1)
    else:
        ratio = 1

    factor = 1.0  # same size
    if prH != prW:
        factor = 2.0  # prH is smaller than prW and poW
        print("\n\n\nATTENTION: Scaling factor used!\n\n\n")

    print("\ngaussian2dTo1d_diag:", mv, radius, mgd,
          "(connecting", pre.name, "to", post.name, ")")

    # Normalization along width of sigma values on afferent map
    if poW % 2 == 1 or prW % 2 == 1 or prH % 2 == 1:
        print("ATTENTION: this is a bit fishy")
        # not very consistent (probably better to normalize along diagonal)
        sigma = radius * (prH-(prH % 2))
        offset = ((prW+prH-2) - ratio*(poW-1))/2
        print("OFFSET:", offset, "RATIO:", ratio, "\n\n")

    else:
        # not very consistent (probably better to normalize along diagonal)
        sigma = radius * prH
        if prW == prH and prH == poW:
            offset = prW/2.0
        else:
            #offset = (prW-poW)/2.0
            offset = poW//2

    synapse = CSR()

    # for speedup
    m_exp = math.exp

    if saveConnections:
        fn = pre.name + "_" + post.name + "_" + "gaussian2dTo1d_diag"
        strToWrite = "connect " + pre.name + " to " + \
            post.name + " with gaussian2dTo1d_diag" + "\n\n"

    for w_post in range(poW):

        values = []
        pre_ranks = []

        for w_pre in range(prW):
            for h_pre in range(prH):

                saveVal = 0

                if (pre != post) or (w_post != w_pre):
                    #dist = (w_post - (w_pre+h_pre) + offset)**2
                    dist = (ratio*w_post -
                            (math.ceil(w_pre/factor)+h_pre) + offset)**2
                    if (mgd == 0) or (dist < mgd*mgd):
                        val = mv * m_exp(-dist/sigma/sigma)
                        if val > MIN_CONNECTION_VALUE:
                            # connect
                            pre_rank = pre.rank_from_coordinates(
                                (w_pre, h_pre))
                            values.append(val)
                            pre_ranks.append(pre_rank)

                            saveVal = val

                if saveConnections:
                    strToWrite += "(" + str(w_pre) + "," + str(h_pre) + \
                        ") -> (" + str(w_post) + ") with " + \
                        str(saveVal) + "\n"

        synapse.add(w_post, pre_ranks, values, [delay])

    if saveConnections:
        f = open(saveDir+fn+'.txt', 'w')
        f.write(strToWrite)
        f.close()

    time1 = time.time()
    print('created', synapse.nb_synapses, 'synapses out of ',
          poW*prW*prH, ' in', time1-time0, 's')
    return synapse


def connect_gaussian1dTo2d_diag(pre, post, mv, radius, mgd, delay=0, scale=0):
    '''
    connect two maps with a gaussian receptive field 1d to 2d diagonally

    params: pre, post -- layers, that should be connected
            mv        -- highest value for one connection
            radius    -- width of receptive field (in deg)
            mgd       -- width of receptive field (in number of neurons)

    return: synapses  -- CSR-object with connections
    '''

    time0 = time.time()

    prW = pre.width
    poW = post.width
    poH = post.height

    if scale:
        ratio = scale*(poW+poH-2)/float(prW-1)
    else:
        ratio = 1

    factor = 1.0  # same size
    if poH != poW:
        factor = 2.0  # poH is smaller than prW and poW
        print("\n\n\nATTENTION: Scaling factor used!\n\n\n")

    print("\ngaussian1dTo2d_diag:", mv, radius, mgd,
          "(connecting", pre.name, "to", post.name, ")")

    synapse = CSR()

    # Normalization along width of sigma values on afferent map
    if prW % 2 == 1 or poW % 2 == 1 or poH % 2 == 1:
        print("ATTENTION: this is a bit fishy")
        sigma = radius * (poH-(poH % 2))
        offset = ((poW+poH-2) - ratio*(prW-1))/2
        print("OFFSET:", offset, "RATIO:", ratio, "\n\n")

    else:
        sigma = radius * poH
        if prW == poW and poW == poH:
            offset = prW/2.0
        else:
            offset = (poW-prW)/2.0

    # for speedup
    m_exp = math.exp

    if saveConnections:
        fn = pre.name + "_" + post.name + "_" + "gaussian1dTo2d_diag"
        strToWrite = "connect " + pre.name + " to " + \
            post.name + " with gaussian1dTo2d_diag" + "\n\n"

    for w_post in range(poW):
        for h_post in range(poH):

            post_rank = post.rank_from_coordinates((w_post, h_post))

            values = []
            pre_ranks = []

            for w_pre in range(prW):

                saveVal = 0

                if (pre != post) or (w_post != w_pre):
                    dist = (math.ceil(w_post/factor) +
                            h_post - ratio*w_pre - offset)**2
                    if (mgd == 0) or (dist < mgd*mgd):
                        val = mv * m_exp(-dist/sigma/sigma)
                        if val > MIN_CONNECTION_VALUE:
                            # connect
                            values.append(val)
                            pre_ranks.append(w_pre)

                            saveVal = val

                if saveConnections:
                    strToWrite += "(" + str(w_pre) + ") -> (" + str(w_post) + \
                        "," + str(h_post) + ") with " + str(saveVal) + "\n"

            synapse.add(post_rank, pre_ranks, values, [delay])

    if saveConnections:
        f = open(saveDir+fn+'.txt', 'w')
        f.write(strToWrite)
        f.close()

    time1 = time.time()
    print('created', synapse.nb_synapses, 'synapses out of ',
          poW*prW*poH, ' in', time1-time0, 's')
    return synapse


def connect_gaussian2d_diagTo2d_v(pre, post, mv, radius, mgd, inv):
    '''
    connect two maps with a gaussian receptive field 2d to 2d diagonally

    params: pre, post -- layers, that should be connected
            mv        -- highest value for one connection
            radius    -- width of receptive field (in deg)
            mgd       -- width of receptive field (in number of neurons)
            inv       -- should the connection be done in inverse order?

    return: synapses  -- CSR-object with connections
    '''

    time0 = time.time()

    prW = pre.width
    prH = pre.height
    poW = post.width
    poH = post.height

    print("\ngaussian2d_diagTo2d_v:", mv, radius, mgd, inv,
          "(connecting", pre.name, "to", post.name, ")")

    # Normalization along width of sigma values on afferent map
    # sigma = radius * prH #<-- not very consistent (probably better to normalize along diagonal)
    # Normalization along height of post layer
    if poH in [21, 41]:
        print("ATTENTION: this is a bit fishy")
        sigma = radius * (poH-1)
    else:
        sigma = radius * poH

    if prW in [21, 41]:
        offset = (prW-1)/2.0
    else:
        offset = prW/2.0

    synapse = CSR()

    # Ratio of size between maps
    #ratio  = prH/float(poH)
    #offset = ratio*poH-prH
    # print(ratio, offset

    # for speedup
    m_exp = math.exp

    if saveConnections:
        fn = pre.name + "_" + post.name + "_" + "gaussian2d_diagTo2d_v"
        strToWrite = "connect " + pre.name + " to " + \
            post.name + " with gaussian2d_diagTo2d_v" + "\n\n"

    for w_post in range(poW):
        for h_post in range(poH):

            post_rank = post.rank_from_coordinates((w_post, h_post))

            values = []
            pre_ranks = []

            for w_pre in range(prW):
                for h_pre in range(prH):

                    saveVal = 0

                    if (pre != post) or (h_post != h_pre):
                        #dist = (ratio*h_post - (w_pre+h_pre) + min(prW,prH)/2.0 + offset/2.0)**2
                        dist = (h_post - (w_pre+h_pre) + offset)**2
                        if (mgd == 0) or (dist < mgd*mgd):
                            val = mv * m_exp(-dist/sigma/sigma)
                            if val > MIN_CONNECTION_VALUE:
                                # connect
                                if inv:
                                    pre_rank = pre.rank_from_coordinates(
                                        (prW-1-w_pre, prH-1-h_pre))
                                else:
                                    pre_rank = pre.rank_from_coordinates(
                                        (w_pre, h_pre))

                                values.append(val)
                                pre_ranks.append(pre_rank)

                                saveVal = val

                    if saveConnections:
                        strToWrite += "(" + str(w_pre) + "," + str(h_pre) + ") -> (" + str(
                            w_post) + "," + str(h_post) + ") with " + str(saveVal) + "\n"

            synapse.add(post_rank, pre_ranks, values, [0])

    if saveConnections:
        print("save at ", saveDir+fn+'.txt')
        f = open(saveDir+fn+'.txt', 'w')
        f.write(strToWrite)
        f.close()

    time1 = time.time()
    print('created', synapse.nb_synapses, 'synapses out of ',
          poW*prW*poH*prH, ' in', time1-time0, 's')
    return synapse


def connect_dog1dTo2d_diag(pre, post, mp, sp, mn, sn, mgd):
    '''
    connect two maps with a difference-of-gaussian receptive field 1d to 2d diagonally

    params: pre, post -- layers, that should be connected
            mp, mn    -- highest value for one connection for each gaussian
            sp, sn    -- width of receptive field for each gaussian (in deg)
            mgd       -- width of receptive field (in number of neurons)

    return: synapses  -- CSR-object with connections
    '''

    time0 = time.time()

    prW = pre.width
    poW = post.width
    poH = post.height

    factor = 1.0  # same size
    if poH != poW:
        factor = 2.0  # poH is smaller than prW and poW

    print("connect_dog1dTo2d_diag:", mp, sp, mn, sn, mgd,
          "(connecting", pre.name, "to", post.name, ")")

    # Normalization along width of sigma values
    if prW % 2 == 1 or poW % 2 == 1 or poH % 2 == 1:
        print("ATTENTION: this is a bit fishy")
        sigma_pos = sp * (poH-(poH % 2))
        sigma_neg = sn * (poH-(poH % 2))
        if prW == poW and poW == poH:
            offset = (prW-(prW % 2))/2.0
        else:
            offset = ((poW-(poW % 2)) - (prW-(prW % 2)))/2.0
    else:
        sigma_pos = sp * poH
        sigma_neg = sn * poH
        if prW == poW and poW == poH:
            offset = prW/2.0
        else:
            offset = (poW-prW)/2.0

    synapse = CSR()

    # for speedup
    m_exp = math.exp

    if saveConnections:
        fn = pre.name + "_" + post.name + "_" + "dog1dTo2d_diag"
        strToWrite = "connect " + pre.name + " to " + \
            post.name + " with dog1dTo2d_diag" + "\n\n"

    for w_post in range(poW):
        for h_post in range(poH):

            post_rank = post.rank_from_coordinates((w_post, h_post))

            values = []
            pre_ranks = []

            for w_pre in range(prW):

                saveVal = 0

                if (pre != post) or (w_post != w_pre):
                    #dist = (w_post + h_post - w_pre - offset)**2
                    dist = (math.ceil(w_post/factor) +
                            h_post - w_pre - offset)**2
                    if (mgd == 0) or (dist < mgd*mgd):
                        val = mp*m_exp(-dist/sigma_pos/sigma_pos) - \
                            mn*m_exp(-dist/sigma_neg/sigma_neg)
                        if val > MIN_CONNECTION_VALUE:
                            # connect
                            values.append(val)
                            pre_ranks.append(w_pre)

                            saveVal = val

                if saveConnections:
                    strToWrite += "(" + str(w_pre) + ") -> (" + str(w_post) + \
                        "," + str(h_post) + ") with " + str(saveVal) + "\n"

            synapse.add(post_rank, pre_ranks, values, [0])

    if saveConnections:
        f = open(saveDir+fn+'.txt', 'w')
        f.write(strToWrite)
        f.close()

    time1 = time.time()
    print('created', synapse.nb_synapses, 'synapses out of ',
          poW*prW*poH, ' in', time1-time0, 's')
    return synapse


def connect_all2all_exp1d(pre, post, factor, radius, mgd):
    '''
    connecting two 1-dimensional maps (normally these maps are equal)
    with gaussian field depending on distance

    params: pre, post -- layers, that should be connected
            factor    -- highest value for one connection
            radius    -- width of receptive field (in deg)
            mgd       -- width of receptive field (in number of neurons)

    return: synapses  -- CSR-object with connections
    '''

    time0 = time.time()

    prW = pre.width
    poW = post.width

    print("all2all_exp1d:", factor, radius, mgd,
          "(connecting", pre.name, "to", post.name, ")")

    # Normalization along width of sigma values on afferent map
    if prW % 2 == 1:
        print("ATTENTION: this is a bit fishy")
        sigma = radius * (prW-1)
        mgd = mgd * (prW-1)
    else:
        sigma = radius * prW
        mgd = mgd * prW

    # Ratio of size between maps
    ratio = prW/float(poW)
    offset = ratio*poW - prW

    synapse = CSR()

    # for speedup
    m_exp = math.exp

    if saveConnections:
        fn = pre.name + "_" + post.name + "_" + "all2all_exp1d"
        strToWrite = "connect " + pre.name + " to " + \
            post.name + " with all2all_exp1d" + "\n\n"

    for w_post in range(poW):

        values = []
        pre_ranks = []

        for w_pre in range(prW):

            saveVal = 0

            # distance between 2 neurons
            #dist = (w_pre-w_post)**2
            dist = (ratio*w_post-w_pre+offset/2.0)**2

            if (mgd == 0) or (dist < mgd*mgd):
                val = factor * m_exp(-dist/sigma/sigma)
                # connect
                values.append(val)
                pre_ranks.append(w_pre)

                saveVal = val

            if saveConnections:
                strToWrite += "(" + str(w_pre) + ") -> (" + \
                    str(w_post) + ") with " + str(saveVal) + "\n"

        synapse.add(w_post, pre_ranks, values, [0])

    if saveConnections:
        f = open(saveDir+fn+'.txt', 'w')
        f.write(strToWrite)
        f.close()

    time1 = time.time()
    print('created', synapse.nb_synapses, 'synapses out of ',
          poW*prW, ' in', time1-time0, 's')
    return synapse


def connect_all2all_exp2d(pre, post, factor, radius, mgd):
    '''
    connecting two 2-dimensional maps (normally these maps are equal)
    with gaussian field depending on distance

    params: pre, post -- layers, that should be connected
            factor    -- highest value for one connection
            radius    -- width of receptive field (in deg)
            mgd       -- width of receptive field (in number of neurons)

    return: synapses  -- CSR-object with connections
    '''

    time0 = time.time()

    prW = pre.width
    prH = pre.height
    poW = post.width
    poH = post.height

    print("all2all_exp2d:", factor, radius, mgd,
          "(connecting", pre.name, "to", post.name, ")")

    # Normalization along width of sigma values on afferent map
    if prW % 2 == 1 or prH % 2 == 1:
        print("ATTENTION: this is a bit fishy")
        sigma_w = radius * (prW-(prW % 2))
        mgd_w = mgd * (prW-(prW % 2))

        sigma_h = radius * (prH-(prH % 2))
        mgd_h = mgd * (prH-(prH % 2))
    else:
        sigma_w = radius * prW
        mgd_w = mgd * prW

        sigma_h = radius * prH
        mgd_h = mgd * prH

    # Ratio of size between maps
    ratio_w = prW/float(poW)
    ratio_h = prH/float(poH)
    offset_w = ratio_w*poW - prW
    offset_h = ratio_h*poH-prH

    synapse = CSR()

    # for speedup
    m_exp = math.exp

    if saveConnections:
        fn = pre.name + "_" + post.name + "_" + "all2all_exp2d"
        strToWrite = "connect " + pre.name + " to " + \
            post.name + " with all2all_exp2d" + "\n\n"

    for w_post in range(poW):
        for h_post in range(poH):

            post_rank = post.rank_from_coordinates((w_post, h_post))

            values = []
            pre_ranks = []

            for w_pre in range(prW):
                for h_pre in range(prH):

                    saveVal = 0

                    # distance between 2 neurons
                    dist_w = (ratio_w*w_post-w_pre+offset_w/2.0)**2
                    dist_h = (ratio_h*h_post-h_pre+offset_h/2.0)**2
                    if (mgd == 0) or ((dist_w < mgd_w*mgd_w) and (dist_h < mgd_h*mgd_h)):
                        #val = factor * m_exp(-dist/sigma/sigma)
                        val = factor * \
                            m_exp(-(dist_w/sigma_w/sigma_w +
                                  dist_h/sigma_h/sigma_h))
                        # connect
                        pre_rank = pre.rank_from_coordinates((w_pre, h_pre))
                        values.append(val)
                        pre_ranks.append(pre_rank)

                        saveVal = val

                    if saveConnections:
                        strToWrite += "(" + str(w_pre) + "," + str(h_pre) + ") -> (" + str(
                            w_post) + "," + str(h_post) + ") with " + str(saveVal) + "\n"

            synapse.add(post_rank, pre_ranks, values, [0])

    if saveConnections:
        f = open(saveDir+fn+'.txt', 'w')
        f.write(strToWrite)
        f.close()

    time1 = time.time()
    print('created', synapse.nb_synapses, 'synapses out of ',
          poW*prW*poH*prH, ' in', time1-time0, 's')
    return synapse


def connect_one2one1dTo2d_v(pre, post, val):
    '''
    connect two maps one-to-one 1d to 2d
    independent of first dimension of 2d map

    params: pre, post -- layers, that should be connected
            val       -- highest value for one connection

    return: synapses  -- CSR-object with connections
    '''

    time0 = time.time()

    prW = pre.width
    poW = post.width
    poH = post.height

    print("one2one1dTo2d_v:", val, "(connecting", pre.name, "to", post.name, ")")

    synapse = CSR()

    if saveConnections:
        fn = pre.name + "_" + post.name + "_" + "gaussian1dTo2d_v"
        strToWrite = "connect " + pre.name + " to " + \
            post.name + " with gaussian1dTo2d_v" + "\n\n"

    for w_post in range(poW):
        for h_post in range(poH):

            post_rank = post.rank_from_coordinates((w_post, h_post))

            h_pre = h_post

            saveVal = 0

            # connect
            saveVal = val

            if saveConnections:
                strToWrite += "(" + str(h_pre) + ") -> (" + str(w_post) + \
                    "," + str(h_post) + ") with " + str(saveVal) + "\n"

            synapse.add(post_rank, [h_pre], [val], [0])

    if saveConnections:
        f = open(saveDir+fn+'.txt', 'w')
        f.write(strToWrite)
        f.close()

    time1 = time.time()
    print('created', synapse.nb_synapses, 'synapses out of ',
          poW*prW*poH, ' in', time1-time0, 's')
    return synapse
