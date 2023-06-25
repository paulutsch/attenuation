import numpy as np
from little_helpers import (gauss_generator,
                            gauss_generator_ring,
                            dog_generator,
                            long_generator)


def generate_input(mode, up_time, res, res_o, pad, n_seq, n_settle, n_permute, p_permute, p_common, limit,
                   continuous_inputs=(1,0), insert_blanks=0, jitter=0, jitter_test=[1,], poisson_inputs=False,
                   sigma=0.75, G=2, constant_G=True, p_o_moves=0.5, uniform_o=False, ring=False, bias=False, baseline=0,
                   no_vision_test=False, double_test=False, RHI_test=False, Rel_test=False):
    """
        generates Gaussian input patterns following the relationship o = r + e = t + s

            * mode: number of inputs / type of task
            * res: resolution of the input populations
            * res_o: resolution of the output population
            * pad: padding boarder space
            * n_seq: number of training trials
            * n_settle: number of trials with input similar to testing conditions
            * n_permute: number of testing trials
            * p_common: percentage of trials with common cause for both modalities (only for mode 2)
            * limit: percentage of inputs where e+s combinations with missing retinal input are ommitted
            * continuous_inputs: length of continuous input sequences
            * continuous_inputs2: probability of staying in the same spatial constellation between to input sequences
            * insert blanks: insert blank trials after sequence
            * poisson_inputs: ~
            * noisy_inputs: additive uncorrelated noise for every input neuron parameterized by "noise"
            * noise: [pop1_training, pop1_test, pop2_training, pop2_test, ...]
            * sigma: of Gaussian
            * G: maximum (average) height of Gaussian
            * constant_G: constant G across movement sequences
            * p_o_moves: percentage of trials where object moves
            * uniform_o: uniform object position distribution
            * ring: if True creates inputs in ring attractor style
            * bias: whether some positions are underrepresented (left shoulder positions)
            * baseline: activity
            * no_vision_test: add test with occluded vision
            * RHI_test: add test with biased inputs for first modality
            * Rel_test: add test with varying input reliabilities

    """

    if type(no_vision_test)==tuple:
        no_vision_base = no_vision_test[1]
        no_vision_test = no_vision_test[0]
    else:
        no_vision_base = False

    if type(poisson_inputs)==tuple:
        poisson_inputs_test = poisson_inputs[1]
        poisson_inputs = poisson_inputs[0]


    tests = [1, (1 if no_vision_test else 0), (1 if double_test else 0), (1 if RHI_test else 0), (1 if Rel_test else 0)]

    n_trials = (n_seq + n_settle*2 + n_permute*2*sum(tests))

    print("\nGenerating", n_trials, "Trials, Mode", mode)

    if ring:
        gen = gauss_generator_ring
    else:
        if type(G)==tuple and type(sigma)==tuple:
            gen = dog_generator
            print("...using DOG pattern")
        else:
            gen = gauss_generator

    multi_G = False
    if not isinstance(G, (list, tuple)):
        G = (G,)*4
    elif isinstance(G[0], (list, tuple)):
        multi_G = True
        G_l = np.array(G)[:, 0]
        G_u = np.array(G)[:, 1]

    if not isinstance(sigma, (list, tuple)):
        sigma = (sigma,)*4

    res_t = res_o if mode==3 else res
    #jitter_test = np.array(jitter_test) * 1 / jitter if jitter else jitter_test

    input_o = np.zeros((n_trials, res_o)) + baseline
    o = np.zeros(n_trials, dtype='complex')
    input_s = np.zeros((n_trials, res)) + baseline
    s = np.zeros(n_trials, dtype='complex')
    input_e = np.zeros((n_trials, res)) + baseline
    e = np.zeros(n_trials, dtype='complex')
    input_r = np.zeros((n_trials, res)) + baseline
    r = np.zeros(n_trials, dtype='complex')
    input_t = np.zeros((n_trials, res_t)) + baseline
    t = np.zeros(n_trials, dtype='complex')

    G_s = np.random.uniform(G_l[2], G_u[2], n_trials)      if multi_G and mode==2 else (np.zeros(n_trials) + np.mean(G[2]))
    G_e = np.random.uniform(G_l[1], G_u[1], n_trials)      if multi_G and mode==2 else (np.zeros(n_trials) + np.mean(G[1]))
    G_r = np.random.uniform(G_l[0], G_u[0], n_trials)      if multi_G else (np.zeros(n_trials) + G[0])
    G_t = np.random.uniform(G_l[3], G_u[3], n_trials)      if multi_G else (np.zeros(n_trials) + G[3])

    if constant_G:

        len_seq = continuous_inputs[0]+insert_blanks

        _G_s = G_s[:n_seq:len_seq]
        G_s[:n_seq] = np.repeat(_G_s, len_seq)
        _G_e = G_e[:n_seq:len_seq]
        G_e[:n_seq] = np.repeat(_G_e, len_seq)
        _G_r = G_r[:n_seq:len_seq]
        G_r[:n_seq] = np.repeat(_G_r, len_seq)
        _G_t = G_t[:n_seq:len_seq]
        G_t[:n_seq] = np.repeat(_G_t, len_seq)

    J_s = np.random.normal(0, (sigma[2] if poisson_inputs else 1/(G_s**0.5)) * jitter, n_trials) if jitter>0 else np.zeros(n_trials)
    J_e = np.random.normal(0, (sigma[1] if poisson_inputs else 1/(G_e**0.5)) * jitter, n_trials) if jitter>0 else np.zeros(n_trials)
    J_r = np.random.normal(0, (sigma[0] if poisson_inputs else 1/(G_r**0.5)) * jitter, n_trials) if jitter>0 else np.zeros(n_trials)
    J_t = np.random.normal(0, (sigma[3] if poisson_inputs else 1/(G_t**0.5)) * jitter, n_trials) if jitter>0 else np.zeros(n_trials)


    # one or two objects
    cond0 = np.zeros(n_trials)
    # shoulder or tactile first
    cond1 = np.zeros(n_trials)
    # eye or retinal first
    cond2 = np.zeros(n_trials)
    # continuously moving object, shoulder or eye
    cond3 = np.zeros(n_trials) - 1 # -1 = new spatial configuration, 0 = object moves, 1 = shoulder moves, 2 = eyes move
    # staying in place between movement sequences
    cond4 = np.zeros(n_trials)
    # insert blank trial after changing positions between sequences
    cond5 = np.zeros(n_trials)
    # jitter_test
    cond6 = np.ones(n_trials)

    if pad:
        pad = tuple(np.array(sigma)*2) + ((res-1)/2 + max(sigma),)
    else:
        pad = (0,)*5

    ## Training Stimuli ##
    if mode in (3,4):
        k = 0
        for i in range(0, n_seq):

            cond0[i] = 1 if np.random.uniform() < p_common else 2 # determine number of objects

            if k==1: # 2. step in sequence
                # determine movement condition
                if mode==4:
                    cond3[i] = (np.random.uniform() > p_o_moves) * np.random.randint(1,3)
                else:
                    cond3[i] = (np.random.uniform() > p_o_moves) * 2
            elif k > 1:
                cond3[i] = cond3[i-1] # keep movement condition over sequence
            elif i>0 and cond5[i]==0.5: # 1. step in sequence
                cond4[i] = 1 if continuous_inputs[1]>np.random.uniform() else 0 # determine configuration condition
                cond5[i:i+insert_blanks] = 0 if cond4[i] else 1 # set blank trials

            if insert_blanks and cond5[i]:
                pass
            else:
                # object position
                if (cond3[i]==-1 or cond3[i]==0) and cond4[i]==0:
                    if cond3[i]==0: # object moves
                        if k==1 or cond4[i-1]==1: # 2. step or 1. step with same configuration
                            dir_o = np.random.randint(0,2)*2-1 # determine object movement direction
                        mu_o = mu_o + dir_o # move
                        mu_t = mu_o - mu_s
                        mu_r = mu_o - mu_e
                        if (mu_o < 0+pad[4] or mu_o > res_o-1-pad[4] or
                            (mu_t < 0+pad[3] and dir_o<0) or (mu_t > (res_t-1)-pad[3] and dir_o>0) or
                            (mu_r < 0+pad[0] and dir_o<0) or (mu_r > (res-1)-pad[0] and dir_o>0)):
                            dir_o *= -1 # turn direction
                            mu_o = mu_o + 2*dir_o # step back
                    elif uniform_o:
                        mu_o = np.random.uniform(0+pad[4], res_o-1-pad[4])
                    else:
                        mu_o = -1
                        while mu_o < 0+pad[4] or mu_o > res_o-1-pad[4]:
                            mu_o = np.random.normal(res-1, (res-1)/2.4)
                input_o[i, :] += gen(1, mu_o, 0.4, res_o)
                o[i] = mu_o


                cond1[i] = np.random.randint(0,2) # draw shoulder or tactile position first
                if mode==3:
                    cond1[i] = 0
                if cond1[i]==0 or cond3[i]>=0 or cond4[i]==1:
                    # shoulder position
                    if (cond3[i]==-1 or cond3[i]==1) and cond4[i]==0:
                        if cond3[i]==1: # shoulder moves
                            if k==1 or cond4[i-1]==1:
                                dir_s = np.random.randint(0,2)*2-1 # determine shoulder movement direction
                            mu_s = mu_s + dir_s # move
                            if ((mu_s < 0+pad[2] and dir_s<0) or (mu_s > (res-1)-pad[2] and dir_s>0) or
                                (mu_t < 0+pad[3] and dir_s>0) or (mu_t > (res-1)-pad[3] and dir_s<0)):
                                dir_s *= -1 # turn direction
                                mu_s = mu_s + 2*dir_s # step back
                        elif np.random.uniform() < limit:
                            mu_s = np.random.uniform(max(0+pad[2], o[i]-(res-1-pad[3])), min(o[i]-pad[3], res-1-pad[2]))
                        else:
                            mu_s = np.random.uniform(0+pad[2], res-1-pad[2])
                        if mode==3:
                            mu_s = 0
                    input_s[i,:] += gen(G_s[i], mu_s + J_s[i], sigma[2], res)
                    s[i] = mu_s
                    # tactile position
                    mu_t = np.array(o[i] - s[i], dtype='float64')
                    input_t[i,:] += gen(G_t[i], mu_t + J_t[i], sigma[3], res_t)
                    t[i] = mu_t
                else:
                    # tactile position
                    if np.random.uniform() < 1:
                        mu_t = np.random.uniform(max(0+pad[3], o[i]-(res-1-pad[2])), min(o[i]-pad[2], res-1-pad[3]))
                    else:
                        mu_t = np.random.uniform(0+pad[3], res-1-pad[3])
                    input_t[i,:] += gen(G_t[i], mu_t + J_t[i], sigma[3], res)
                    t[i] = mu_t
                    # shoulder position
                    mu_s = np.array(o[i] - t[i], dtype='float64')
                    input_s[i,:] += gen(G_s[i], mu_s + J_s[i], sigma[2], res)
                    s[i] = mu_s


                cond2[i] = np.random.randint(0,2) # draw eye or retinal position first
                if cond2[i]==0 or cond3[i]>=0 or cond4[i]==1:
                    # eye position
                    if (cond3[i]==-1 or cond3[i]==2) and cond4[i]==0:
                        if cond3[i]==2: # eye moves
                            if k==1 or cond4[i-1]==1:
                                dir_e = np.random.randint(0,2)*2-1 # determine eye movement direction
                            mu_e = mu_e + dir_e # move
                            if ((mu_e < 0+pad[1] and dir_e<0) or (mu_e > (res-1)-pad[1] and dir_e>0) or
                                (mu_r < 0+pad[0] and dir_e>0) or (mu_r > (res-1)-pad[0] and dir_e<0)):
                                dir_e *= -1 # turn direction
                                mu_e = mu_e + 2*dir_e # step back
                        elif np.random.uniform() < limit:
                            mu_e = np.random.uniform(max(0+pad[1], o[i]-(res-1-pad[0])), min(o[i]-pad[0], res-1-pad[1]))
                        else:
                            mu_e = np.random.uniform(0+pad[1], res-1-pad[1])
                    input_e[i,:] += gen(G_e[i], mu_e + J_e[i], sigma[1], res)
                    e[i] = mu_e
                    # retinal position
                    mu_r = np.array(o[i] - e[i], dtype='float64')
                    input_r[i,:] += gen(G_r[i], mu_r + J_r[i], sigma[0], res)
                    r[i] = mu_r
                else:
                    # retinal position
                    if np.random.uniform() < 1:
                        mu_r = np.random.uniform(max(0+pad[0], o[i]-(res-1-pad[1])), min(o[i]-pad[1], res-1-pad[0]))
                    else:
                        mu_r = np.random.uniform(0+pad[0], res-1-pad[0])
                    input_r[i,:] += gen(G_r[i], mu_r + J_r[i], sigma[0], res)
                    r[i] = mu_r
                    # eye position
                    mu_e = np.array(o[i] - r[i], dtype='float64')
                    input_e[i,:] += gen(G_e[i], mu_e + J_e[i], sigma[1], res)
                    e[i] = mu_e

                k = (k + 1) % continuous_inputs[0]
                if k==0:
                    if i+1 < n_seq:
                        cond5[i+1] = 0.5

    elif mode == 2:
        for i in range(0, n_seq):

            cond0[i] = 1 if np.random.uniform() < p_common else 2 # determine number of objects

            if cond0[i]==1:

                # retinal position
                if uniform_o:
                    mu_r = np.random.uniform(0+pad[0], res-1-pad[0])
                else:
                    mu_r = -1
                    while mu_r < 0+pad[0] or mu_r > res-1-pad[0]:
                        mu_r = np.random.normal((res-1)/2, (res-1)/2.4)
                input_r[i,:] += gen(G_r[i], mu_r + J_r[i], sigma[0], res)
                r[i] = mu_r
                # tactile position
                mu_t = mu_r
                input_t[i,:] += gen(G_t[i], mu_t + J_t[i], sigma[3], res)
                t[i] = mu_t*1j
                # object position
                mu_o = mu_r + mu_t
                input_o[i, :] += gen(1, mu_o, 0.4, res_o)
                o[i] = mu_o*(1+1j)

            else:

                # retinal position
                if uniform_o:
                    mu_r = np.random.uniform(0+pad[0], res-1-pad[0])
                else:
                    mu_r = -1
                    while mu_r < 0+pad[0] or mu_r > res-1-pad[0]:
                        mu_r = np.random.normal((res-1)/2, (res-1)/2.4)
                input_r[i,:] += gen(G_r[i], mu_r + J_r[i], sigma[0], res)
                r[i] = mu_r
                # tactile position
                if uniform_o:
                    mu_t = np.random.uniform(0+pad[3], res-1-pad[3])
                else:
                    mu_t = -1
                    while mu_t < 0+pad[3] or mu_t > res-1-pad[3]:
                        mu_t = np.random.normal((res-1)/2, (res-1)/2.4)
                input_t[i,:] += gen(G_t[i], mu_t + J_t[i], sigma[3], res)
                t[i] = mu_t*1j
                # object positions
                mu_o = mu_r*2
                input_o[i, :] += gen(1, mu_o, 0.4, res_o)
                o[i] = mu_o
                mu_o2 = mu_t*2
                input_o[i, :] += gen(1, mu_o2, 0.4, res_o)
                o[i] += mu_o2*1j

    elif mode == 3:
        for i in range(0, n_seq):

            if uniform_o:

                # object position
                mu_o = np.random.uniform(0+pad[4], res_o-1-pad[4])
                input_o[i, :] += gen(1, mu_o, 0.4, res_o)
                o[i] = mu_o

                cond2[i] = np.random.randint(0,2) # draw eye or retinal position first
                if cond2[i]==0 or cond3[i]>=0 or cond4[i]==1:
                    # eye position
                    if np.random.uniform() < limit:
                        mu_e = np.random.uniform(max(0+pad[1], o[i]-(res-1-pad[0])), min(o[i]-pad[0], res-1-pad[1]))
                    else:
                        mu_e = np.random.uniform(0+pad[1], res-1-pad[1])
                    input_e[i,:] += gen(G_e[i], mu_e + J_e[i], sigma[1], res)
                    e[i] = mu_e
                    # retinal position
                    mu_r = np.array(o[i] - e[i], dtype='float64')
                    input_r[i,:] += gen(G_r[i], mu_r + J_r[i], sigma[0], res)
                    r[i] = mu_r
                else:
                    # retinal position
                    if np.random.uniform() < limit:
                        mu_r = np.random.uniform(max(0+pad[0], o[i]-(res-1-pad[1])), min(o[i]-pad[1], res-1-pad[0]))
                    else:
                        mu_r = np.random.uniform(0+pad[0], res-1-pad[0])
                    input_r[i,:] += gen(G_r[i], mu_r + J_r[i], sigma[0], res)
                    r[i] = mu_r
                    # eye position
                    mu_e = np.array(o[i] - r[i], dtype='float64')
                    input_e[i,:] += gen(G_e[i], mu_e + J_e[i], sigma[1], res)
                    e[i] = mu_e

                # tactile position
                mu_t = mu_o
                input_t[i,:] += gen(G_t[i], mu_t + J_t[i], sigma[3], res_t)
                t[i] = mu_t


            else:

                # retinal position
                mu_r = np.random.uniform(0+pad[0], res-1-pad[0])
                input_r[i,:] += gen(G_r[i], mu_r + J_r[i], sigma[0], res)
                r[i] = mu_r
                # eye position
                mu_e = np.random.uniform(0+pad[3], res-1-pad[3])
                input_e[i,:] += gen(G_e[i], mu_e + J_e[i], sigma[3], res)
                e[i] = mu_e
                # object position
                mu_o = mu_r + mu_e
                input_o[i, :] += gen(1, mu_o, 0.4, res_o)
                o[i] = mu_o

                # tactile position
                mu_t = mu_o
                input_t[i,:] += gen(G_t[i], mu_t + J_t[i], sigma[3], res_t)
                t[i] = mu_t

    else:
        print("[Error] No training trials created!")

    ## Pre-Test Stimuli ##
    for i in np.arange(n_seq, n_seq + n_settle*2, 2):
        trial = np.random.randint(0, n_seq)
        o[i], r[i], e[i], s[i], t[i] = o[trial], r[trial], e[trial], s[trial], t[trial]
        input_r[i], input_e[i], input_s[i], input_t[i] = input_r[trial], input_e[trial], input_s[trial], input_t[trial]


    ## Permutations for Testing RFs ##
    p = int(0.08*res)
    n = n_seq+n_settle*2

    k = 0
    for j in jitter_test:
        q = 0
        for mu_o in np.arange(p*2, res_o-p*2):
            i_e = np.arange(max(mu_o-(res-1), p), min(res-1-p, mu_o)+1) if mode in (3,4) else np.arange(p, res-p)
            for mu_e in i_e:
                if mode==4:
                    i_s = i_e
                elif mode==2:
                    i_s = [mu_e,]
                elif mode==3:
                    i_s = [0,]
                for mu_s in i_s:
                    if mode<3 or (mu_o-mu_e > p and mu_o-mu_e < res-p):
                        q += p_permute
                        if q >= 1:
                            q -= 1
                            k += 1
                            if k > n_permute:
                                print("[Warning] k exceeded n_permute in test data generation!")
                                break

                            cond0[n + k*2-2] = 1
                            cond6[n + k*2-2] = j

                            input_o[n + k*2-2,:] += np.round(gen(1, mu_o, 1, res_o), 2)
                            o[n + k*2-2] = mu_o
                            input_s[n + k*2-2,:] += np.round(gen(G_s[n + k*2-2], mu_s + J_s[n + k*2-2] * j, sigma[2], res), 2)
                            s[n + k*2-2] = mu_s
                            mu_t = (mu_o-mu_s) if mode in (3,4) else mu_o/2
                            input_t[n + k*2-2,:] += np.round(gen(G_t[n + k*2-2], mu_t + J_t[n + k*2-2] * j, sigma[3], res_t), 2)
                            t[n + k*2-2] = mu_t
                            input_e[n + k*2-2,:] += np.round(gen(G_e[n + k*2-2], mu_e + J_e[n + k*2-2] * j, sigma[1], res), 2)
                            e[n + k*2-2] = mu_e
                            mu_r = (mu_o-mu_e) if mode in (3,4) else mu_o/2
                            input_r[n + k*2-2,:] += np.round(gen(G_r[n + k*2-2], mu_r + J_r[n + k*2-2] * j, sigma[0], res), 2)
                            r[n + k*2-2] = mu_r

                if k > n_permute:
                    break
            if k > n_permute:
                break

    ## Other Testing Conditions ##
    if no_vision_test:
        n = n_seq+n_settle*2+n_permute*2

        if no_vision_base:
            if mode==3:
                no_vision_base = no_vision_base * np.mean(gen(np.mean(G_t), (res_t-1)/2, sigma[3], res))
            else:
                no_vision_base = no_vision_base * np.mean(gen(np.mean(G_r), (res-1)/2, sigma[0], res))
        else:
            no_vision_base = 0

        k = 0
        for j in jitter_test:
            q = 0
            for mu_o in np.arange(p*2, res_o-p*2):
                i_e = np.arange(max(mu_o-(res-1), p), min(res-1-p, mu_o)+1) if mode in (3,4) else np.arange(p, res-p)
                for mu_e in i_e:
                    if mode==4:
                        i_s = i_e
                    elif mode==2:
                        i_s = [mu_e,]
                    elif mode==3:
                        i_s = [0,]
                    for mu_s in i_s:
                        if mode<3 or (mu_o-mu_e > p and mu_o-mu_e < res-p):
                            q += p_permute
                            if q >= 1:
                                q -= 1
                                k += 1
                                if k > n_permute:
                                    break

                                cond0[n + k*2-2] = 1
                                cond6[n + k*2-2] = j

                                input_o[n + k*2-2,:] += np.round(gen(1, mu_o, 1, res_o), 2)
                                o[n + k*2-2] = mu_o
                                input_s[n + k*2-2,:] += np.round(gen(G_s[n + k*2-2], mu_s + J_s[n + k*2-2] * j, sigma[2], res), 2)
                                s[n + k*2-2] = mu_s
                                mu_t = (mu_o-mu_s) if mode in (3,4) else mu_o/2
                                input_t[n + k*2-2,:] += no_vision_base if mode==3 else np.round(gen(G_t[n + k*2-2], mu_t + J_t[n + k*2-2] * j, sigma[3], res_t), 2)
                                t[n + k*2-2] = mu_t
                                input_e[n + k*2-2,:] += np.round(gen(G_e[n + k*2-2], mu_e + J_e[n + k*2-2] * j, sigma[1], res), 2)
                                e[n + k*2-2] = mu_e
                                mu_r = (mu_o-mu_e) if mode in (3,4) else mu_o/2
                                input_r[n + k*2-2,:] += no_vision_base if not mode==3 else np.round(gen(G_r[n + k*2-2], mu_r + J_r[n + k*2-2] * j, sigma[0], res), 2)
                                r[n + k*2-2] = mu_r

                    if k > n_permute:
                        break
                if k > n_permute:
                    break

    if double_test:
        n = n_seq+n_settle*2+n_permute*2*sum(tests[:2])

        k = 0
        for j in jitter_test:
            q = 0
            for mu_o in np.arange(p*2, res_o-p*2):
                i_e = np.arange(max(mu_o-(res-1), p), min(res-1-p, mu_o)+1) if mode in (3,4) else np.arange(p, res-p)
                for mu_e in i_e:
                    if mode==4:
                        i_s = i_e
                    elif mode==2:
                        i_s = [mu_e,]
                    elif mode==3:
                        i_s = [0,]
                    for mu_s in i_s:
                        if mode<3 or (mu_o-mu_e > p and mu_o-mu_e < res-p):
                            q += p_permute
                            if q >= 1:
                                q -= 1
                                k += 1
                                if k > n_permute:
                                    break

                                cond0[n + k*2-2] = 2
                                cond6[n + k*2-2] = j

                                input_o[n + k*2-2,:] += np.round(gen(1, mu_o, 1, res_o), 2)
                                o[n + k*2-2] = mu_o
                                input_s[n + k*2-2,:] += np.round(gen(G_s[n + k*2-2], mu_s + J_s[n + k*2-2] * j, sigma[2], res), 2)
                                s[n + k*2-2] = mu_s
                                mu_t = (mu_o-mu_s) if mode in (3,4) else mu_o/2
                                input_t[n + k*2-2,:] += np.round(gen(G_t[n + k*2-2], mu_t + J_t[n + k*2-2] * j, sigma[3], res_t), 2)
                                t[n + k*2-2] = mu_t
                                input_e[n + k*2-2,:] += np.round(gen(G_e[n + k*2-2], mu_e + J_e[n + k*2-2] * j, sigma[1], res), 2)
                                e[n + k*2-2] = mu_e
                                mu_r = (mu_o-mu_e) if mode in (3,4) else mu_o/2
                                input_r[n + k*2-2,:] += np.round(gen(G_r[n + k*2-2], mu_r + J_r[n + k*2-2] * j, sigma[0], res), 2)
                                r[n + k*2-2] = mu_r

                                mu_o2 = np.round(np.random.uniform(p, res_o-1-p), 2)
                                c = 0 if mode in (3,4) else 100
                                while c < 100 and (np.abs(mu_o2-mu_o)<=1 or (mu_o2-mu_s)<p or (mu_o2-mu_s)>res-1-p or (mu_o2-mu_e)<p or (mu_o2-mu_e)>res-1-p):
                                    mu_o2 = np.round(np.random.uniform(p, res_o-1-p), 2)
                                    c += 1
                                input_o[n + k*2-2,:] += np.round(gen(1, mu_o2, 1, res_o), 2)
                                o[n + k*2-2] += mu_o2*1j
                                mu_t2 = (mu_o2-mu_s) if mode in (3,4) else mu_o2/2
                                input_t[n + k*2-2,:] += np.round(gen(G_t[n + k*2-2], mu_t2 + J_t[n + k*2-2] * j, sigma[3], res), 2)
                                s[n + k*2-2] += mu_s*1j
                                t[n + k*2-2] += mu_t2*1j
                                mu_r = (mu_o2-mu_e) if mode in (3,4) else mu_o2/2
                                input_r[n + k*2-2,:] += np.round(gen(G_r[n + k*2-2], mu_r + J_r[n + k*2-2] * j, sigma[0], res), 2)
                                e[n + k*2-2] += mu_e*1j
                                r[n + k*2-2] += mu_r*1j

                    if k > n_permute:
                        break
                if k > n_permute:
                    break


    if RHI_test:
        n = n_seq+n_settle*2+n_permute*2*sum(tests[:3])

        k = 0
        for j in jitter_test:
            q = 0
            for mu_o in np.arange(p*2, res_o-p*2):
                i_e = np.arange(max(mu_o-(res-1), p), min(res-1-p, mu_o)+1) if mode in (3,4) else np.arange(p, res-p)
                for mu_e in i_e:
                    if mode==4:
                        i_s = i_e
                    elif mode==2:
                        i_s = [mu_e,]
                    elif mode==3:
                        i_s = [0,]
                    for mu_s in i_s:
                        if mode<3 or (mu_o-mu_e > p and mu_o-mu_e < res-p):
                            q += p_permute
                            if q >= 1:
                                q -= 1
                                k += 1
                                if k > n_permute:
                                    break

                                cond0[n + k*2-2] = 2
                                cond6[n + k*2-2] = j

                                input_o[n + k*2-2,:] += np.round(gen(1, mu_o, 1, res_o), 2)
                                o[n + k*2-2] = mu_o
                                input_e[n + k*2-2,:] += np.round(gen(G_e[n + k*2-2], mu_e + J_e[n + k*2-2] * j, sigma[1], res), 2)
                                e[n + k*2-2] = mu_e
                                mu_r = (mu_o-mu_e) if mode in (3,4) else mu_o/2
                                input_r[n + k*2-2,:] += np.round(gen(G_r[n + k*2-2], mu_r + J_r[n + k*2-2] * j, sigma[0], res), 2)
                                r[n + k*2-2] = mu_r

                                mu_o2 = np.round(np.random.uniform(p, res_o-1-p), 2)
                                c = 0 if mode in (3,4) else 100
                                while c < 100 and (np.abs(mu_o2-mu_o)<=1 or (mu_o2-mu_s)<p or (mu_o2-mu_s)>res-1-p or (mu_o2-mu_e)<p or (mu_o2-mu_e)>res-1-p):
                                    mu_o2 = np.round(np.random.uniform(p, res_o-1-p), 2)
                                    c += 1
                                input_o[n + k*2-2,:] += np.round(gen(0.5, mu_o2, 1, res_o), 2)
                                o[n + k*2-2] += mu_o2*1j
                                input_s[n + k*2-2,:] += np.round(gen(G_s[n + k*2-2], mu_s + J_s[n + k*2-2] * j, sigma[2], res), 2)
                                s[n + k*2-2] = mu_s
                                s[n + k*2-2] += mu_s*1j
                                e[n + k*2-2] += mu_e*1j
                                mu_t = (mu_o2-mu_s) if mode in (3,4) else mu_o2/2
                                input_t[n + k*2-2,:] += np.round(gen(G_t[n + k*2-2], mu_t + J_t[n + k*2-2] * j, sigma[3], res_t), 2)
                                t[n + k*2-2] = (mu_o-mu_s) if mode in (3,4) else mu_o/2
                                t[n + k*2-2] += mu_t*1j
                                r[n + k*2-2] += ((mu_o2-mu_e) if mode in (3,4) else mu_o2/2)*1j

                    if k > n_permute:
                        break
                if k > n_permute:
                    break


    if Rel_test:
        pass


    ## timing of inputs ##
    timings = np.zeros((n_trials*up_time, 2))
    start = np.zeros((n_trials, 2), dtype='i')
    for i in np.arange(n_trials):
        if cond0[i]==2:
            idx = np.random.randint(2)
            start[i, idx] = np.random.randint(up_time//2)
            timings[i*up_time:start[i, idx]+i*up_time, idx] = 1


    ## pack inputs ##
    if mode == 4:
        inputs = np.stack((input_r, input_e, input_s, input_t), axis=-1)
        input_check = np.vstack((o, r, e, s, t, o-e, s+t-e, cond1, cond2, cond3, cond4, cond5, cond6, cond0)).T
        gains = np.vstack((G_r, G_e, G_s, G_t)).T
        jitter = np.vstack((J_r, J_e, J_s, J_t)).T
    elif mode == 2:
        inputs = np.stack((input_r, input_t), axis=-1)
        input_check = np.vstack((o, r, t, np.array(o, dtype='float')/2, (o-np.array(o, dtype='float'))/2, cond6, cond0)).T
        gains = np.vstack((G_r, G_t)).T
        jitter = np.vstack((J_r, J_t)).T
    elif mode == 3:
        inputs = np.stack((np.hstack((input_r, np.zeros((n_trials, res_t-res)))),
                           np.hstack((input_e, np.zeros((n_trials, res_t-res)))),
                           input_t), axis=-1)
        input_check = np.vstack((o, r, e, t, t-e, t-r, cond6, cond0)).T
        gains = np.vstack((G_r, G_e, G_t)).T
        jitter = np.vstack((J_r, J_e, J_t)).T
    timings = {'steps': timings, 't_start': start}

    jitter = jitter * input_check[:, -2, None]

    ## randomize permutations ##
    for i, t in enumerate(tests):
        if t:
            n = n_seq+n_settle*2+n_permute*2*sum(tests[:i])
            id_x = np.arange(n, n+n_permute*2)
            id_s = np.arange(n, n+n_permute*2)
            np.random.shuffle(id_s[::2])

            inputs[id_s] = inputs[id_x]
            input_o[id_s] = input_o[id_x]
            input_check[id_s] = input_check[id_x]
            gains[id_s] = gains[id_x]
            jitter[id_s] = jitter[id_x]

    inputs = np.moveaxis(inputs, -1, 0)
    ## poissonify inputs ##
    if poisson_inputs:
        inputs[:, :n_seq+n_settle*2] = np.random.poisson(inputs[:, :n_seq+n_settle*2])
    if poisson_inputs_test:
        if no_vision_test:
            n = n_seq+n_settle*2
            inputs[:, n:n+n_permute*2] = np.random.poisson(inputs[:, n:n+n_permute*2])
            n = n_seq+n_settle*2+n_permute*2
            for i in [0,1] if mode==3 else [1,2,3]:
                inputs[i, n:n+n_permute*2] = np.random.poisson(inputs[i, n:n+n_permute*2])
            inputs[:, n_seq+n_settle*2+n_permute*4:] = np.random.poisson(inputs[:, n_seq+n_settle*2+n_permute*4:])
        else:
            inputs[:, n_seq+n_settle*2:] = np.random.poisson(inputs[:, n_seq+n_settle*2:])

    if mode==4:
        print("Input check-sums:", np.round(np.sum(input_check[:, 1]-input_check[:, 5]), 3),
                                   np.round(np.sum(input_check[:, 1]-input_check[:, 6]), 3))
    elif mode==2:
        print("Input check-sums:", np.round(np.sum(input_check[:, 1]-input_check[:, 3]), 3),
                                   np.round(np.sum(input_check[:, 2]-input_check[:, 4]), 3))
    elif mode==3:
        print("Input check-sums:", np.round(np.sum(input_check[:, 1]-input_check[:, 4]), 3),
                                   np.round(np.sum(input_check[:, 2]-input_check[:, 5]), 3))

    return n_trials, inputs, gains, jitter, timings, input_o, input_check
