import numpy as np
import matplotlib.pyplot as mpl
from little_helpers import gauss_generator, gauss_generator_ring, dog_generator, printT
from Whiten import whiten, whiten2


def gen_coords3(res=19, l=10000, G=[0.5, 5], norm=True, dog=False, ring=False, pad=0, k=1):
    """Generate Coordinate Transformation Learning Problem for 3 Populations of Size=res"""

    printT("Generating", l, "training trials")
    coords = np.zeros((l, 3, res))

    # coordinate transformation problem of the form h = e + r
    # r = retinal position of stimulus
    # e = eye position
    # h = head-centered coordinates of object
    r = np.random.uniform(pad, res-1-pad, l)
    e = np.random.uniform(pad, res-1-pad, l)
    h = r + e

    # range of encoded positions per modality
    scale_r = np.linspace(-50, 50, res)
    scale_e = np.linspace(-75, 75, res)
    scale_h = np.linspace(-125, 125, res)

    # random gain per stimulus
    if type(G) == list and len(G) == 2:
        G_r = np.random.uniform(G[0], G[1], l)
        G_e = np.repeat(np.mean(G), l)
        G_h = np.random.uniform(G[0], G[1], l)
    else:
        G_r = np.random.choice(G, l)
        G_e = np.repeat(np.mean(G), l)
        G_h = np.random.choice(G, l)

    # add noise to true position
    J_r = np.random.normal(0, k / G_r**0.5)
    r_ = r + J_r
    J_e = np.random.normal(0, k / G_e**0.5)
    e_ = e + J_e
    J_h = np.random.normal(0, k / G_h**0.5)
    h_ = h + J_h

    # width of tuning curve
    FWHM = res/8  # target full width at half maximum
    sigma_I = FWHM / 2.355

    if ring:  # circular instead of linear encoding
        gen = gauss_generator_ring
    else:
        gen = gauss_generator

    if dog:
        gen = dog_generator
        for i in range(l):
            coords[i, 0] = gen([G_r[i], G_r[i]/2],
                               [r_[i], r_[i]],     [sigma_I, sigma_I*1.5], res)
            coords[i, 1] = gen([G_e[i], G_e[i]/2],
                               [e_[i], e_[i]],     [sigma_I, sigma_I*1.5], res)
            coords[i, 2] = gen([G_h[i], G_h[i]/2],
                               [h_[i]/2, h_[i]/2], [sigma_I, sigma_I*1.5], res)
            while np.sum(np.isnan(coords[i])):

                coords[i, 0] = gen(
                    [G_r[i], G_r[i]/2], [r_[i], r_[i]],     [sigma_I, sigma_I*1.5], res)
                coords[i, 1] = gen(
                    [G_e[i], G_e[i]/2], [e_[i], e_[i]],     [sigma_I, sigma_I*1.5], res)
                coords[i, 2] = gen(
                    [G_h[i], G_h[i]/2], [h_[i]/2, h_[i]/2], [sigma_I, sigma_I*1.5], res)

                printT([G_r[i], G_r[i]/2], [r_[i], r_[i]],
                       [sigma_I, sigma_I*1.5], res)
                printT([G_e[i], G_e[i]/2], [e_[i], e_[i]],
                       [sigma_I, sigma_I*1.5], res)
                printT([G_h[i], G_h[i]/2], [h_[i]/2, h_[i]/2],
                       [sigma_I, sigma_I*1.5], res)
    else:
        for i in range(l):
            coords[i, 0] = gen(G_r[i], r_[i],   sigma_I, res)
            coords[i, 1] = gen(G_e[i], e_[i],   sigma_I, res)
            coords[i, 2] = gen(G_h[i], h_[i]/2, sigma_I, res)

    # normalize inputs to have the same mean and maximum consistent with max(G)
    if norm:
        idmax = np.argmax(coords[..., res//4:res-res//4])
        for i in range(l):
            normalizer = np.sum(coords[i], 1)[..., None]
            normalizer[normalizer <= 0.2] = 0.2
            coords[i] = coords[i] / normalizer * \
                np.vstack((G_r[i], G_e[i], G_h[i])).T[..., None]
        coords = coords / coords[..., res//4:res -
                                 res//4].flatten()[idmax] * max(G)

    return coords, r, e, h, G_r, G_e, G_h, J_r, J_e, J_h, scale_r, scale_e, scale_h


def gen_coords2(res=19, l=10000, G=[0.5, 5], norm=False, white=False, ring=True, pad=0, k=1, cor=False):
    """Generate Sensory Integration Learning Problem for 2 Populations of Size=res"""

    printT("Generating", l, "training trials")
    coords = np.zeros((l, 2, res))

    # sensory integration problem of two source
    # v = visual direction of movement
    # h = vestibular direction of movement
    r = np.random.uniform(pad, res-pad, l)
    if cor:
        cor = 1-cor
        h = r * 1
        new_h = np.random.uniform(pad, res-pad, len(h[::int(1/cor)]))
        h[::int(1/cor)] = new_h
    else:
        h = np.random.uniform(pad, res-pad, l)

    # range of encoded positions per modality
    scale_r = np.linspace(-180, 180, res)
    scale_h = np.linspace(-180, 180, res)

    # random gain per stimulus
    if type(G) == list and len(G) == 2:
        G_r = np.random.uniform(G[0], G[1], l)
        G_h = np.random.uniform(G[0], G[1], l)
    else:
        G_r = np.random.choice(G, l)
        G_h = np.random.choice(G, l)

    # add noise to true position
    J_r = np.random.normal(0, k / G_r**0.5)
    r_ = r + J_r
    J_h = np.random.normal(0, k / G_h**0.5)
    h_ = h + J_h

    # width of tuning curve
    FWHM = res/8  # target full width at half maximum
    sigma_I = FWHM / 2.355

    if ring:  # circular instead of linear encoding
        gen = gauss_generator_ring
    else:
        gen = gauss_generator

    for i in range(l):
        coords[i, 0] = gen(G_r[i], r_[i], sigma_I, res)
        coords[i, 1] = gen(G_h[i], h_[i], sigma_I, res)

    # normalize inputs to have the same mean and maximum consistent with max(G)
    if norm:
        idmax = np.argmax(coords[..., res//4:res-res//4])
        coords = coords / \
            np.sum(coords, 2)[..., None] * np.vstack((G_r, G_h)).T[..., None]
        coords = coords / coords[..., res//4:res -
                                 res//4].flatten()[idmax] * max(G)

    if white:  # whitening to remove correlations of inputs in covariance matrix
        A_max = np.max(coords)
        coords = whiten(coords) if white == 1 else whiten2(coords)
        coords = coords / np.max(coords) * A_max

    return coords, r, h, G_r, G_h, J_r, J_h, scale_r, scale_h


def gen_coords_test(N=3, res=19, G=[0.5, 5], norm=True, white=False, dog=False, ring=False, pad=0, rnd=True, select=None, **kwargs):
    """Generate Random Probe Stimuli for N Populations of Size=res"""

    if type(res) in [int, float]:
        res = range(res)
    if type(G) in [int, float]:
        G = list(G)

    G_ = G*1

    # create all possible input combinations
    coords_scalar = np.array(np.meshgrid(
        *[res for _ in range(N)], *[G for _ in range(N)]))
    l = coords_scalar[0].size
    coords_scalar = coords_scalar.reshape(N*2, l)

    # select certain combinations
    if select is not None:
        if type(select) in [list, tuple]:
            for _, s in enumerate(select):
                coords_scalar = s(coords_scalar, *kwargs[s.__name__])
        else:
            coords_scalar = select(coords_scalar, **kwargs)
        l = coords_scalar[0].size
    printT("Generating", l, "test trials")
    if rnd:
        ids = np.arange(l)
        np.random.shuffle(ids)
        coords_scalar = coords_scalar[:, ids]

    res = int(max(res))+1
    coords = np.zeros((l, N, res))

    # gain per stimulus
    G = coords_scalar[N:]

    # width of tuning curve
    FWHM = res/8  # target full width at half maximum
    sigma_I = FWHM / 2.355

    if ring:  # circular instead of linear encoding
        gen = gauss_generator_ring
    else:
        gen = gauss_generator
    if dog:
        gen = dog_generator
        for j in range(N):
            for i in range(l):
                coords[i, j] = gen(
                    [G[j, i], G[j, i]/2], [coords_scalar[j, i], ]*2, [sigma_I, sigma_I*1.5], res)
    else:
        for j in range(N):
            for i in range(l):
                coords[i, j] = gen(G[j, i], coords_scalar[j, i], sigma_I, res)

    # normalize inputs to have the same mean and maximum consistent with max(A)
    if norm:
        idmax = np.argmax(coords[..., res//4:res-res//4])
        if N == 2:
            coords = coords / np.sum(coords, 2)[..., None] * G.T[..., None]
        else:
            for i in range(l):
                normalizer = np.sum(coords[i], 1)[..., None]
                normalizer[normalizer <= 0.2] = 0.2
                coords[i] = coords[i] / normalizer * G[:, i][..., None]
        coords = coords / coords[..., res//4:res -
                                 res//4].flatten()[idmax] * max(G_)

    if white:  # whitening to remove correlations of inputs in covariance matrix
        A_max = np.max(coords)
        coords = whiten(coords) if white == 1 else whiten2(coords)
        coords = coords / np.max(coords) * A_max

    return coords, coords_scalar[:N], G, l


def random_select(a, l):
    """Select l random samples from array"""
    size = a[0].size
    if size >= l:
        idx = np.random.choice(range(size), l, False)
    else:
        idx = np.random.choice(range(size), l)
    return a[:, idx]


def range_select(a, lower, upper):
    """Select inputs within position range"""
    for i in range(a.shape[0] // 2):
        a = a[:, a[i] >= lower]
        a = a[:, a[i] <= upper]
    return a


def G_select(a, col, g):
    """Select inputs within position range"""
    if type(col) in (int, float):
        col = [int(col), ]
    for i in col:
        a = a[:, a[i+a.shape[0]//2] == g]
    return a
