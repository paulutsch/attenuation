import numpy as np
import os
from .kinematic import (
    forward_kinematics,
    closest_point_on_vector_to_point,
    angle_of_pos_relative_to_head_2d,
    inverse_kinematics_2d,
    random_theta_within_limits,
)
from .constants import (
    theta_limits_2d,
    forearm_thickness,
    forearm_length,
    upper_arm_length,
    shoulder_distance,
)
from little_helpers import (
    normalized_input,
    gauss_generator,
    gauss_generator_ring,
    dog_generator,
    printT,
)
import matplotlib.pyplot as mpl


def load_data_from_file(data_path, file_name, is_2d=True):
    file_path = os.path.join(data_path, file_name)
    with open(file_path, "rb") as f:
        data = np.loadtxt(f, ndmin=2) if is_2d else np.loadtxt(f)
    return data


def write_data_to_file(data_path, file_name, data):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    file_path = os.path.join(data_path, file_name)
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            pass  # this creates the file if it doesn't exist
    matrix_data = np.matrix(data)
    with open(file_path, "wb") as f:
        for line in matrix_data:
            np.savetxt(f, line)


def plotDataSet(data, theta_static, static_side, ratio_within_forearm_region):
    print("\npreparing data plots ...")
    if static_side == "left":
        dynamic_side = "right"
    else:
        dynamic_side = "left"

    n, N = data.shape
    data_wrist_pos_dynamic = data[:, 0:3] * 1

    data_shyaw_dynamic = data[0, 3] * 1  # should be constant
    data_shptch_dynamic = data[0, 4] * 1  # should be constant
    data_shrl_dynamic = data[:, 5] * 1
    data_elbw_dynamic = data[:, 6] * 1

    data_angl_to_dynamic = data[:, 10] * 1
    data_touch = data[:, 8] * 1
    data_tctl_on_static = data[:, 7] * 1
    data_tctl_touch = data_tctl_on_static[data_touch == 1.0]

    standard_elbow_pos_dynamic, standard_wrist_pos_dynamic = forward_kinematics(
        np.radians([90, 90, 60, -60]),
        dynamic_side,
        round=True,
        return_elbow=True,
        return_wrist=True,
    )

    theta_dynamic = np.array(
        [
            [
                data_shyaw_dynamic,
                data_shptch_dynamic,
                data_shrl_dynamic[i],
                data_elbw_dynamic[i],
            ]
            for i in range(n)
        ]
    )

    data_elbow_pos_static, data_wrist_pos_static = forward_kinematics(
        theta_static, static_side, return_wrist=True, return_elbow=True
    )
    data_elbow_pos_dynamic = np.zeros((n, 3))

    for i in range(n):
        elbow_pos_dynamic = forward_kinematics(
            theta_dynamic[i],
            dynamic_side,
            return_wrist=False,
            return_elbow=True,
        )

        data_elbow_pos_dynamic[i] = elbow_pos_dynamic

    total_length = upper_arm_length + forearm_length + shoulder_distance

    x_static = np.array(
        [-shoulder_distance, data_elbow_pos_static[2], data_wrist_pos_static[2]]
    )  # 2: z, here x
    y_static = np.array(
        [0, data_elbow_pos_static[1] * (-1), data_wrist_pos_static[1] * (-1)]
    )  # 1: y, here y

    # 2d plot of arms
    rows = 4
    columns = 4

    print("\nplotting ...")

    fig1 = mpl.figure()
    for i in range(rows * columns):
        rndm = np.random.randint(0, n)
        index = i + 1

        ax = fig1.add_subplot(rows, columns, index)
        ax.set_xlim([-total_length, total_length])
        ax.set_ylim([-0.1, total_length])

        ax.text(
            -total_length + 0.05,
            0.0,
            "angle: "
            + str("{:.1f}".format(data_angl_to_dynamic[rndm]))
            + "\nshoulder roll: "
            + str("{:.1f}".format(np.degrees(data_shrl_dynamic[rndm])))
            + "\nelbow: "
            + str("{:.1f}".format(np.degrees(data_elbw_dynamic[rndm]))),
            fontsize=5,
        )

        x_dynamic = np.array(
            [
                shoulder_distance,
                data_elbow_pos_dynamic[rndm, 2],
                data_wrist_pos_dynamic[rndm, 2],
            ]
        )
        y_dynamic = np.array(
            [
                0,
                data_elbow_pos_dynamic[rndm, 1] * (-1),
                data_wrist_pos_dynamic[rndm, 1] * (-1),
            ]
        )

        # head
        ax.plot(0, 0, "yo", markersize=3)

        # static arm
        ax.plot(x_static, y_static, "-o", color="blue", markersize=3)

        # dynamic arm
        clr = "red" if data_touch[rndm] == 1.0 else "green"
        ax.plot(x_dynamic, y_dynamic, "-o", color=clr, markersize=3)

    mpl.show()

    n_within_forearm_region = int(n * ratio_within_forearm_region)
    n_random_theta = int(n - n_within_forearm_region)

    # additional info
    n_touching = np.count_nonzero(data_touch == 1.0)
    n_not_touching = np.count_nonzero(data_touch == 0.0)
    n_touching_within_forearm_region = np.count_nonzero(
        data_touch[n_random_theta:] == 1.0
    )
    n_touching_random_theta = np.count_nonzero(data_touch[:n_random_theta] == 1.0)
    n_not_touching_within_forearm_region = np.count_nonzero(
        data_touch[n_random_theta:] == 0.0
    )
    n_not_touching_random_theta = np.count_nonzero(data_touch[:n_random_theta] == 0.0)

    x_dynamic = np.array(
        [
            shoulder_distance,
            standard_elbow_pos_dynamic[2],
            standard_wrist_pos_dynamic[2],
        ]
    )
    y_dynamic = np.array(
        [0, standard_elbow_pos_dynamic[1] * (-1), standard_wrist_pos_dynamic[1] * (-1)]
    )

    # plotting histograms of first part of data (random thetas)
    fig2 = mpl.figure()
    fig2.suptitle("random thetas")

    ax = fig2.add_subplot(2, 3, 1)
    ax.set_xlim([-total_length, total_length])
    ax.set_ylim([-0.1, total_length])
    ax.set_title("wrist pos with random thetas")

    ax.plot(
        data_wrist_pos_dynamic[:n_random_theta][:, 2],
        data_wrist_pos_dynamic[:n_random_theta][:, 1] * (-1),
        "ro",
        markersize=1,
        alpha=0.02,
    )

    # head
    ax.plot(0, 0, "yo", markersize=3)
    # static arm
    ax.plot(-shoulder_distance, 0, "bo", markersize=3)
    ax.plot(x_static, y_static, "-o", color="blue", markersize=3)
    # dynamic arm
    ax.plot(shoulder_distance, 0, "bo", markersize=3)
    ax.plot(x_dynamic, y_dynamic, "-o", color="green", markersize=3)

    ax = fig2.add_subplot(2, 3, 2)
    ax.set_title("touching")
    ax.bar([0, 1], [n_not_touching_random_theta, n_touching_random_theta])

    ax = fig2.add_subplot(2, 3, 3)
    ax.set_title("shoulder roll")
    counts, bins = np.histogram(np.degrees(data_shrl_dynamic[:n_random_theta]))
    ax.stairs(counts, bins)

    ax = fig2.add_subplot(2, 3, 4)
    ax.set_title("elbow")
    counts, bins = np.histogram(np.degrees(data_elbw_dynamic[:n_random_theta]))
    ax.stairs(counts, bins)

    ax = fig2.add_subplot(2, 3, 5)
    ax.set_title("head angle")
    counts, bins = np.histogram(data_angl_to_dynamic[:n_random_theta])
    ax.stairs(counts, bins)

    ax = fig2.add_subplot(2, 3, 6)
    ax.set_title("tactile")
    data_tctl_touch_random_theta = data_tctl_on_static[:n_random_theta][
        data_touch[:n_random_theta] == 1.0
    ]
    counts, bins = np.histogram(data_tctl_touch_random_theta)
    ax.stairs(counts, bins)

    mpl.show()

    # plotting histograms of second part of data (dynamic wrist pos within forearm region)
    fig2 = mpl.figure()
    fig2.suptitle("dynamic wrist pos within forearm region")

    ax = fig2.add_subplot(2, 3, 1)
    ax.set_xlim([-total_length, total_length])
    ax.set_ylim([-0.1, total_length])
    ax.set_title("wrist pos")

    ax.plot(
        data_wrist_pos_dynamic[n_random_theta:][:, 2],
        data_wrist_pos_dynamic[n_random_theta:][:, 1] * (-1),
        "ro",
        markersize=1,
        alpha=0.02,
    )

    # head
    ax.plot(0, 0, "yo", markersize=3)
    # static arm
    ax.plot(-shoulder_distance, 0, "bo", markersize=3)
    ax.plot(x_static, y_static, "-o", color="blue", markersize=3)
    # dynamic arm
    ax.plot(shoulder_distance, 0, "bo", markersize=3)
    ax.plot(x_dynamic, y_dynamic, "-o", color="green", markersize=3)

    ax = fig2.add_subplot(2, 3, 2)
    ax.set_title("touching")
    ax.bar(
        [0, 1], [n_not_touching_within_forearm_region, n_touching_within_forearm_region]
    )

    ax = fig2.add_subplot(2, 3, 3)
    ax.set_title("shoulder roll")
    counts, bins = np.histogram(np.degrees(data_shrl_dynamic[n_random_theta:]))
    ax.stairs(counts, bins)

    ax = fig2.add_subplot(2, 3, 4)
    ax.set_title("elbow")
    counts, bins = np.histogram(np.degrees(data_elbw_dynamic[n_random_theta:]))
    ax.stairs(counts, bins)

    ax = fig2.add_subplot(2, 3, 5)
    ax.set_title("head angle")
    counts, bins = np.histogram(data_angl_to_dynamic[n_random_theta:])
    ax.stairs(counts, bins)

    ax = fig2.add_subplot(2, 3, 6)
    ax.set_title("tactile")
    data_tctl_on_static_touch_forearm_region = data_tctl_on_static[n_random_theta:][
        data_touch[n_random_theta:] == 1.0
    ]
    counts, bins = np.histogram(data_tctl_on_static_touch_forearm_region)
    ax.stairs(counts, bins)

    mpl.show()

    # plotting histograms of all data (random thetas + in forearm region)
    fig2 = mpl.figure()
    fig2.suptitle("all data")

    ax = fig2.add_subplot(2, 3, 1)
    ax.set_xlim([-total_length, total_length])
    ax.set_ylim([-0.1, total_length])
    ax.set_title("wrist pos")

    ax.plot(
        data_wrist_pos_dynamic[:, 2],
        data_wrist_pos_dynamic[:, 1] * (-1),
        "ro",
        markersize=1,
        alpha=0.02,
    )

    # head
    ax.plot(0, 0, "yo", markersize=3)
    # static arm
    ax.plot(-shoulder_distance, 0, "bo", markersize=3)
    ax.plot(x_static, y_static, "-o", color="blue", markersize=3)
    # dynamic arm
    ax.plot(shoulder_distance, 0, "bo", markersize=3)
    ax.plot(x_dynamic, y_dynamic, "-o", color="green", markersize=3)

    ax = fig2.add_subplot(2, 3, 2)
    ax.set_title("touching")
    ax.bar([0, 1], [n_not_touching, n_touching])

    ax = fig2.add_subplot(2, 3, 3)
    ax.set_title("shoulder roll")
    counts, bins = np.histogram(np.degrees(data_shrl_dynamic))
    ax.stairs(counts, bins)

    ax = fig2.add_subplot(2, 3, 4)
    ax.set_title("elbow")
    counts, bins = np.histogram(np.degrees(data_elbw_dynamic))
    ax.stairs(counts, bins)

    ax = fig2.add_subplot(2, 3, 5)
    ax.set_title("head angle")
    counts, bins = np.histogram(data_angl_to_dynamic)
    ax.stairs(counts, bins)

    ax = fig2.add_subplot(2, 3, 6)
    ax.set_title("tactile")
    counts, bins = np.histogram(data_tctl_touch)
    ax.stairs(counts, bins)

    mpl.show()


def create_input_test(
    theta_static,
    static_side,
    res,
    sigma_I,
    eye_angle_degrees=0,
    data_path="./data",
    create_new_data=False,
    only_within_forearm_region=False,
    filter=False,
    norm=True,
    plotNorm=False,
):
    data_path = data_path + "/static_" + static_side

    data_test = create_data_test(
        theta_static,
        static_side,
        data_path=data_path,
        eye_angle_degrees=eye_angle_degrees,
        create_new_data=create_new_data,
        only_within_forearm_region=only_within_forearm_region,
    )

    if filter:
        fltr = (data_test[:, 6] < np.radians(-106)) & (
            data_test[:, 6] > np.radians(-113)
        )
        data_test = data_test[fltr]

    (
        input_test_shrl,
        input_test_elbw,
        input_test_angl,
        input_test_tctl,
        X_test,
    ) = create_input(
        data_test, res, sigma_I, shuffle=False, norm=norm, plotNorm=plotNorm
    )

    test_data_path = os.path.join(data_path, "test")
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path)
    np.save(test_data_path + "/data", data_test)
    np.save(test_data_path + "/input_shrl", input_test_shrl)
    np.save(test_data_path + "/input_elbw", input_test_elbw)
    np.save(test_data_path + "/input_angl", input_test_angl)
    np.save(test_data_path + "/input_tctl", input_test_tctl)
    np.save(test_data_path + "/X", X_test)

    return input_test_shrl, input_test_elbw, input_test_angl, input_test_tctl, X_test


def create_input_train(
    n,
    res,
    theta_static,
    static_side,
    sigma_I,
    ratio_within_forearm_region,
    data_path,
    eye_angle_degrees=0,
    shuffle=True,
    plot=False,
    create_new_data=False,
    filter=False,
    norm=True,
    plotNorm=False,
):
    data_train = create_data_train(
        n,
        theta_static,
        static_side,
        ratio_within_forearm_region,
        b=2 * forearm_thickness,
        eye_angle_degrees=eye_angle_degrees,
        data_path=data_path,
        create_new_data=create_new_data,
    )

    if filter:
        fltr = (data_train[:, 6] < np.radians(-106)) & (
            data_train[:, 6] > np.radians(-113)
        )
        data_train = data_train[fltr]

    if plot:
        plotDataSet(data_train, theta_static, static_side, ratio_within_forearm_region)

    (
        input_train_shrl,
        input_train_elbw,
        input_train_angl,
        input_train_tctl,
        X_train,
    ) = create_input(
        data_train, res, sigma_I, shuffle=shuffle, norm=norm, plotNorm=plotNorm
    )

    return (
        data_train,
        input_train_shrl,
        input_train_elbw,
        input_train_angl,
        input_train_tctl,
        X_train,
    )


def create_input(data, res, sigma, shuffle=True, norm=True, plotNorm=False):
    print("creating input from data...")
    n, N = data.shape

    if shuffle:
        np.random.shuffle(data)

    data_shrl = data[:, 5] * 1
    min_shrl = np.min(data_shrl)
    max_shrl = np.max(data_shrl)

    data_elbw = data[:, 6] * 1
    min_elbw = np.min(data_elbw)
    max_elbw = np.max(data_elbw)

    data_angl = data[:, 10] * 1
    min_angl = np.min(data_angl)
    max_angl = np.max(data_angl)

    data_tctl = data[:, 7] * 1
    min_tctl = 0.0
    max_tctl = 1.0

    mu_shrl = 0
    mu_elbw = 0
    mu_angl = 0
    mu_tctl = 0

    data_tchs = data[:, 8] * 1

    input_shrl = np.zeros((n, res[0]))
    input_elbw = np.zeros((n, res[1]))
    input_angl = np.zeros((n, res[2]))
    input_tctl = np.zeros((n, res[3]))
    X = np.zeros((N, n))

    # creating inputs from data
    # creating array X containing mean of gaussian relative to max mean for each input and data point ("mu_i")
    for i in range(n):
        # shoulder roll population
        shrl = data_shrl[i]
        x_shrl = (shrl - min_shrl) / (max_shrl - min_shrl)
        X[0, i] = x_shrl
        mu_shrl = x_shrl * (res[0] - 1)
        input_shrl[i] = gauss_generator(1, mu_shrl, sigma[0], res[0])

        # elbow population
        elbw = data_elbw[i]
        x_elbw = (elbw - min_elbw) / (max_elbw - min_elbw)
        X[1, i] = x_elbw
        mu_elbw = x_elbw * (res[1] - 1)
        input_elbw[i] = gauss_generator(1, mu_elbw, sigma[1], res[1])

        # angle_yz population
        angl = data_angl[i]
        x_angl = (angl - min_angl) / (max_angl - min_angl)
        X[2, i] = x_angl
        mu_angl = x_angl * (res[2] - 1)
        input_angl[i] = gauss_generator(1, mu_angl, sigma[2], res[2])

        # tactile population
        touches = data_tchs[i]
        if touches:
            tctl = data_tctl[i]
            x_tctl = (tctl - min_tctl) / (max_tctl - min_tctl)
            X[3, i] = x_tctl
            mu_tctl = x_tctl * (res[3] - 1)
            input_tctl[i] = gauss_generator(1, mu_tctl, sigma[3], res[3])
        else:
            input_tctl[i] = np.zeros(res[3])
            X[3, i] = -1

    # if ring:  # circular instead of linear encoding
    #     gen = gauss_generator_ring
    # else:
    #     gen = gauss_generator
    # if dog:
    #     gen = dog_generator
    #     for j in range(N):
    #         for i in range(l):
    #             coords[i, j] = gen(
    #                 [G[j, i], G[j, i]/2], [coords_scalar[j, i], ]*2, [sigma_I, sigma_I*1.5], res)
    # else:
    #     for j in range(N):
    #         for i in range(l):
    #             coords[i, j] = gen(G[j, i], coords_scalar[j, i], sigma_I, res)

    # normalize inputs to have the same mean and maximum consistent with max(A)
    if norm:
        print("\nnormalizing inputs ...")
        input_shrl = normalized_input(input_shrl, plot=plotNorm)
        print("  1/4")
        input_elbw = normalized_input(input_elbw, plot=plotNorm)
        print("  2/4")
        input_angl = normalized_input(input_angl, plot=plotNorm)
        print("  3/4")
        input_tctl = normalized_input(input_tctl, plot=plotNorm)
        print("  4/4")

    # if white:  # whitening to remove correlations of inputs in covariance matrix
    #     A_max = np.max(coords)
    #     coords = whiten(coords) if white == 1 else whiten2(coords)
    #     coords = coords / np.max(coords) * A_max

    return input_shrl, input_elbw, input_angl, input_tctl, X


def create_data_train(
    n,
    theta_static,
    static_side,
    ratio_within_forearm_region,
    b,
    eye_angle_degrees=0,
    data_path="./data",
    create_new_data=False,
):
    if static_side == "left":
        dynamic_side = "right"
    else:
        dynamic_side = "left"

    data_path = data_path + "/static_" + static_side

    N = 22
    data = np.zeros((n, N))

    # in 2D
    theta_limits = theta_limits_2d * 1

    # dynamic arm data – we stay on 2d yz area: shoulder_pitch is constantly 90 degrees
    wrist_pos_dynamic = np.zeros(3)
    theta_dynamic = np.zeros(4)

    # static and dynamic arm data – t_ is norm of vector containing information about point on forearm closest to right wrist
    t_static = 0
    d_static = np.zeros(3)
    distance_to_static = 0
    touching_on_static = False

    t_dynamic = 0
    d_dynamic = np.zeros(3)
    distance_to_dynamic = 0
    touching_on_dynamic = False

    # calculate area of dynamic wrist position on static forearm
    t_max = forearm_length
    elbow_pos_static, wrist_pos_static = forward_kinematics(
        theta_static,
        static_side,
        round=True,
        return_wrist=True,
        return_elbow=True,
    )
    forearm_static = np.subtract(wrist_pos_static, elbow_pos_static)

    wps = np.array(wrist_pos_static)

    # head angles yz
    angle_to_dynamic = 0
    angle_to_static = angle_of_pos_relative_to_head_2d(
        wps, eye_angle_degrees, return_as_degrees=True
    )

    # check if data already exists.
    data_prev_path = data_path + "/data_train.txt"

    create_new_A = True
    if not create_new_data and os.path.isfile(data_prev_path):
        # previous data exists – retrieve it.
        with open(data_path + "/theta_static.txt", "rb") as f_tl_read:
            theta_static_prev = np.loadtxt(f_tl_read)

            # compare previous theta to given theta.
            if not (theta_static_prev == theta_static).all():
                # no equal thetas – create new data.
                create_new_data = True
            else:
                # equal thetas
                create_new_A = False

                with open(data_path + "/data_train.txt", "rb") as f_data_read:
                    data_prev = np.loadtxt(f_data_read)

                    if (data_prev.shape[0] == n) and (data_prev.shape[1] == N):
                        data = data_prev
                        print("\nretrieved existing kinematic data (train).")
                    else:
                        create_new_data = True
    else:
        create_new_data = True

    if create_new_data:
        if create_new_A:
            print("\ncreating new A data (train) ...")

            # [n_A = n_A_a * n_A_b] && [n_A_a / n_A_b = forearm_length / b]
            # leads to below
            n_A = 20000
            n_A_a = int(np.sqrt((forearm_length / b) * n_A))
            n_A_b = int(n_A / n_A_a)

            (
                wrist_pos_dynamic_A,
                elbow_pos_dynamic_A,
                theta_dynamic_A,
                touching_on_static_A,
                t_static_A,
                distance_to_static_A,
                touching_on_dynamic_A,
                t_dynamic_A,
                distance_to_dynamic_A,
                angle_to_dynamic_A,
            ) = create_A(
                n_A_a, n_A_b, b, static_side, forearm_static, t_max, elbow_pos_static
            )

            print("done.")

            write_data_to_file(data_path, "theta_static.txt", theta_static)
            write_data_to_file(
                data_path + "/A", "wrist_pos_dynamic_A.txt", wrist_pos_dynamic_A
            )
            write_data_to_file(
                data_path + "/A", "elbow_pos_dynamic_A.txt", elbow_pos_dynamic_A
            )
            write_data_to_file(data_path + "/A", "theta_dynamic_A.txt", theta_dynamic_A)
            write_data_to_file(
                data_path + "/A", "touching_on_static_A.txt", touching_on_static_A
            )
            write_data_to_file(data_path + "/A", "t_static_A.txt", t_static_A)
            write_data_to_file(
                data_path + "/A", "distance_to_static_A.txt", distance_to_static_A
            )
            write_data_to_file(
                data_path + "/A", "touching_on_dynamic_A.txt", touching_on_dynamic_A
            )
            write_data_to_file(data_path + "/A", "t_dynamic_A.txt", t_dynamic_A)
            write_data_to_file(
                data_path + "/A", "distance_to_dynamic_A.txt", distance_to_dynamic_A
            )
            write_data_to_file(
                data_path + "/A", "angle_to_dynamic_A.txt", angle_to_dynamic_A
            )
        else:
            wrist_pos_dynamic_A = load_data_from_file(
                data_path + "/A", "wrist_pos_dynamic_A.txt"
            )
            elbow_pos_dynamic_A = load_data_from_file(
                data_path + "/A", "elbow_pos_dynamic_A.txt"
            )
            theta_dynamic_A = load_data_from_file(
                data_path + "/A", "theta_dynamic_A.txt"
            )
            touching_on_static_A = load_data_from_file(
                data_path + "/A", "touching_on_static_A.txt", False
            )
            t_static_A = load_data_from_file(data_path + "/A", "t_static_A.txt", False)
            distance_to_static_A = load_data_from_file(
                data_path + "/A", "distance_to_static_A.txt", False
            )
            touching_on_dynamic_A = load_data_from_file(
                data_path + "/A", "touching_on_dynamic_A.txt", False
            )
            t_dynamic_A = load_data_from_file(
                data_path + "/A", "t_dynamic_A.txt", False
            )
            distance_to_dynamic_A = load_data_from_file(
                data_path + "/A", "distance_to_dynamic_A.txt", False
            )
            angle_to_dynamic_A = load_data_from_file(
                data_path + "/A", "angle_to_dynamic_A.txt", False
            )

            print("\nretrieved existing A data (train).")

        print("\ncreating new kinematic data (train) ...")
        n_within_A = int(ratio_within_forearm_region * n)
        n_everywhere = int(n - n_within_A)

        # create data points with random theta
        for i in range(n_everywhere):
            print("  {:.1f}".format((i / n) * 100), "%", end="\r")

            touching_on_static = True
            theta_dynamic = random_theta_within_limits(radians=True)
            elbow_pos_dynamic, wrist_pos_dynamic = forward_kinematics(
                theta_dynamic,
                dynamic_side,
                round=True,
                return_wrist=True,
                return_elbow=True,
            )
            forearm_dynamic = np.subtract(wrist_pos_dynamic, elbow_pos_dynamic)

            t_static, d_static = closest_point_on_vector_to_point(
                forearm_static, np.subtract(wrist_pos_dynamic, elbow_pos_static)
            )
            distance_to_static = np.linalg.norm(d_static.astype(float))
            t_dynamic, d_dynamic = closest_point_on_vector_to_point(
                forearm_dynamic, np.subtract(wrist_pos_static, elbow_pos_dynamic)
            )
            distance_to_dynamic = np.linalg.norm(d_dynamic.astype(float))

            touching_on_static = (distance_to_static <= forearm_thickness) and (
                0 <= t_static <= 1
            )
            touching_on_dynamic = (distance_to_dynamic <= forearm_thickness) and (
                0 <= t_dynamic <= 1
            )

            wpd = np.array(wrist_pos_dynamic)
            angle_to_dynamic = angle_of_pos_relative_to_head_2d(
                wpd, eye_angle_degrees, return_as_degrees=True
            )

            data[i, :] = [
                wrist_pos_dynamic[0],
                wrist_pos_dynamic[1],
                wrist_pos_dynamic[2],
                theta_dynamic[0],
                theta_dynamic[1],
                theta_dynamic[2],
                theta_dynamic[3],
                t_static,
                touching_on_static,
                distance_to_static,
                angle_to_dynamic,
                wrist_pos_static[0],
                wrist_pos_static[1],
                wrist_pos_static[2],
                theta_static[0],
                theta_static[1],
                theta_static[2],
                theta_static[3],
                t_dynamic,
                touching_on_dynamic,
                distance_to_dynamic,
                angle_to_static,
            ]

        # create data points within forearm region
        i = n_everywhere
        rndm_idx = 0

        while i < n:
            print("  {:.1f}".format((i / n) * 100), "%", end="\r")
            # pick random A
            rndm_idx = np.random.randint(0, len(touching_on_static_A))

            wrist_pos_dynamic = np.array(wrist_pos_dynamic_A[rndm_idx, :])
            elbow_pos_dynamic = np.array(elbow_pos_dynamic_A[rndm_idx, :])
            theta_dynamic = np.array(theta_dynamic_A[rndm_idx, :])

            touching_on_static = touching_on_static_A[rndm_idx]
            t_static = t_static_A[rndm_idx]
            distance_to_static = distance_to_static_A[rndm_idx]

            touching_on_dynamic = touching_on_dynamic_A[rndm_idx]
            t_dynamic = t_dynamic_A[rndm_idx]
            distance_to_dynamic = distance_to_dynamic_A[rndm_idx]

            angle_to_dynamic = angle_to_dynamic_A[rndm_idx]

            data[i, :] = [
                wrist_pos_dynamic[0],
                wrist_pos_dynamic[1],
                wrist_pos_dynamic[2],
                theta_dynamic[0],
                theta_dynamic[1],
                theta_dynamic[2],
                theta_dynamic[3],
                t_static,
                touching_on_static,
                distance_to_static,
                angle_to_dynamic,
                wrist_pos_static[0],
                wrist_pos_static[1],
                wrist_pos_static[2],
                theta_static[0],
                theta_static[1],
                theta_static[2],
                theta_static[3],
                t_dynamic,
                touching_on_dynamic,
                distance_to_dynamic,
                angle_to_static,
            ]

            i += 1

        print("\nsaving created data (train) ...")
        write_data_to_file(data_path, "data_train.txt", data)

    return data


def create_data_test(
    theta_static,
    static_side,
    data_path,
    create_new_data,
    eye_angle_degrees=0,
    only_within_forearm_region=False,
):
    if static_side == "left":
        dynamic_side = "right"
    else:
        dynamic_side = "left"

    data_prev_path = data_path + "/data_test.txt"

    if not create_new_data and os.path.isfile(data_prev_path):
        # previous data exists – retrieve it.
        with open(data_path + "/theta_static.txt", "rb") as f_tl_read:
            theta_static_prev = np.loadtxt(f_tl_read)

            # compare previous theta with given theta.
            if not (theta_static_prev == theta_static).all():
                # no equal thetas – create new data.
                create_new_data = True
            else:
                # equal thetas

                with open(data_path + "/data_test.txt", "rb") as f_data_read:
                    data = np.loadtxt(f_data_read)
                    print("\nretrieved existing kinematic data (test).")
                    print(data.shape, "\n")

    else:
        create_new_data = True

    if create_new_data:
        print("\ncreating kinematic data (test).")
        if only_within_forearm_region:
            # wrist_pos_right only within forearm region
            wrist_pos_dynamic_A = load_data_from_file(
                data_path + "/A", "wrist_pos_dynamic_A.txt", is_2d=True
            )
            elbow_pos_dynamic_A = load_data_from_file(
                data_path + "/A", "elbow_pos_dynamic_A.txt", is_2d=True
            )
            theta_dynamic_A = load_data_from_file(
                data_path + "/A", "theta_dynamic_A.txt", is_2d=True
            )
            touching_on_static_A = load_data_from_file(
                data_path + "/A", "touching_on_static_A.txt", is_2d=False
            )
            t_static_A = load_data_from_file(
                data_path + "/A", "t_static_A.txt", is_2d=False
            )
            distance_to_static_A = load_data_from_file(
                data_path + "/A", "distance_to_static_A.txt", is_2d=False
            )
            touching_on_dynamic_A = load_data_from_file(
                data_path + "/A", "touching_on_dynamic_A.txt", is_2d=False
            )
            t_dynamic_A = load_data_from_file(
                data_path + "/A", "t_dynamic_A.txt", is_2d=False
            )
            distance_to_dynamic_A = load_data_from_file(
                data_path + "/A", "distance_to_dynamic_A.txt", is_2d=False
            )
            angle_to_dynamic_A = load_data_from_file(
                data_path + "/A", "angle_to_dynamic_A.txt", is_2d=False
            )

            print("\nretrieved existing A data (train).")

            data = np.array(
                [
                    wrist_pos_dynamic_A[:, 0],
                    wrist_pos_dynamic_A[:, 1],
                    wrist_pos_dynamic_A[:, 2],
                    theta_dynamic_A[:, 0],  # shyaw (static)
                    theta_dynamic_A[:, 1],  # shptch (static)
                    theta_dynamic_A[:, 2],  # shrl
                    theta_dynamic_A[:, 3],  # elbw
                    t_dynamic_A,
                    touching_on_static_A,
                    distance_to_static_A,
                    angle_to_dynamic_A,
                ]
            )
            data = data.T
        else:
            # create all possible theta combinations
            N = 11

            min_shrl = theta_limits_2d[2, 0]
            max_shrl = theta_limits_2d[2, 1]
            min_elbw = theta_limits_2d[3, 0]
            max_elbw = theta_limits_2d[3, 1]

            all_shrl = np.arange(min_shrl, max_shrl + 1)
            all_elbw = np.arange(min_elbw, max_elbw + 1)

            n = len(all_shrl) * len(all_elbw)

            data = np.zeros((n, N))

            wrist_pos_dynamic = np.zeros(3)
            theta_dynamic = np.zeros(4)

            # static arm data – t_static is norm of vector containing information about point on forearm closest to right wrist
            t_static = 0
            d_static = np.zeros(3)
            distance_to_static = 0
            touching_on_static = False

            # head angle yz
            angle_to_dynamic = 0

            # calculate static forearm
            elbow_pos_static, wrist_pos_static = forward_kinematics(
                theta_static,
                static_side,
                round=True,
                return_wrist=True,
                return_elbow=True,
            )
            forearm_static = np.subtract(wrist_pos_static, elbow_pos_static)

            i = 0
            for s, shrl in enumerate(all_shrl):
                for e, elbw in enumerate(all_elbw):
                    i = (s * len(all_elbw)) + e

                    print("  {:.1f}".format((i / n) * 100), "%", end="\r")

                    theta_dynamic = np.array([90, 90, shrl, elbw])
                    theta_dynamic = np.radians(theta_dynamic)
                    wrist_pos_dynamic = forward_kinematics(theta_dynamic, dynamic_side)

                    t_static, d_static = closest_point_on_vector_to_point(
                        forearm_static, np.subtract(wrist_pos_dynamic, elbow_pos_static)
                    )
                    distance_to_static = np.linalg.norm(d_static.astype(float))

                    touching_on_static = (distance_to_static <= forearm_thickness) and (
                        0 <= t_static <= 1
                    )

                    wpr = np.array(wrist_pos_dynamic)
                    angle_to_dynamic = angle_of_pos_relative_to_head_2d(
                        wpr, eye_angle_degrees, return_as_degrees=True
                    )

                    data[i, :] = [
                        wrist_pos_dynamic[0],
                        wrist_pos_dynamic[1],
                        wrist_pos_dynamic[2],
                        theta_dynamic[0],
                        theta_dynamic[1],
                        theta_dynamic[2],
                        theta_dynamic[3],
                        t_static,
                        touching_on_static,
                        distance_to_static,
                        angle_to_dynamic,
                    ]

        print("\nsaving created data (test) ...")
        write_data_to_file(data_path, "data_test.txt", data)

    return data


def create_A(
    n_A_a,
    n_A_b,
    b,
    static_side,
    forearm_static,
    t_max,
    elbow_pos_static,
    eye_angle_degrees=0,
):
    if static_side == "left":
        dynamic_side = "right"
    else:
        dynamic_side = "left"

    wrist_pos_static = np.add(elbow_pos_static, forearm_static)

    index = 0
    j_range = n_A_b / 2

    wrist_pos_dynamic_A = np.zeros([n_A_a * n_A_b, 3])
    elbow_pos_dynamic_A = np.zeros([n_A_a * n_A_b, 3])
    theta_dynamic_A = np.zeros([n_A_a * n_A_b, 4])

    touching_on_static_A = np.zeros([n_A_a * n_A_b])
    t_static_A = np.zeros([n_A_a * n_A_b])
    distance_to_static_A = np.zeros([n_A_a * n_A_b])

    touching_on_dynamic_A = np.zeros([n_A_a * n_A_b])
    t_dynamic_A = np.zeros([n_A_a * n_A_b])
    distance_to_dynamic_A = np.zeros([n_A_a * n_A_b])

    angle_to_dynamic_A = np.zeros([n_A_a * n_A_b])

    t_increm = t_max / n_A_a

    pos_on_forearm_static = np.zeros(3)
    unit_forearm_static = forearm_static / np.linalg.norm(forearm_static)

    forearm_static_orthogonal = np.array(
        [0, unit_forearm_static[2], -unit_forearm_static[1]]
    )
    forearm_static_orthogonal = forearm_static_orthogonal * (b / 2)

    for i in range(n_A_a):
        t_ = t_increm * i
        pos_on_forearm_static = np.add(elbow_pos_static, t_ * unit_forearm_static)

        for j in range(n_A_b):
            index = j + i * n_A_b
            print("  {:.1f}".format((i / n_A_a) * 100), "%", end="\r")

            if j <= j_range:
                j_increm = (j_range - j) / j_range
            else:
                j_increm = -(j - j_range) / j_range

            wrist_pos_dynamic = np.add(
                pos_on_forearm_static, forearm_static_orthogonal * j_increm
            )

            theta_dynamic_A[index, :], good_enough = inverse_kinematics_2d(
                wrist_pos_dynamic,
                dynamic_side,
                0.001,
                display=True,
            )

            elbow_pos_dynamic, wrist_pos_dynamic = forward_kinematics(
                theta_dynamic_A[index, :],
                dynamic_side,
                round=True,
                return_wrist=True,
                return_elbow=True,
            )
            wrist_pos_dynamic_A[index, :] = np.array(wrist_pos_dynamic)
            elbow_pos_dynamic_A[index, :] = np.array(elbow_pos_dynamic)
            forearm_dynamic = np.subtract(wrist_pos_dynamic, elbow_pos_dynamic)

            t_static_A[index], d_static = closest_point_on_vector_to_point(
                forearm_static, np.subtract(wrist_pos_dynamic, elbow_pos_static)
            )
            distance_to_static_A[index] = np.linalg.norm(d_static.astype(float))

            if distance_to_static_A[index] > (forearm_thickness / 2):
                touching_on_static_A[index] = False
            else:
                touching_on_static_A[index] = True

            t_dynamic_A[index], d_dynamic = closest_point_on_vector_to_point(
                forearm_dynamic, np.subtract(wrist_pos_static, elbow_pos_dynamic)
            )
            distance_to_dynamic_A[index] = np.linalg.norm(d_dynamic.astype(float))

            if distance_to_dynamic_A[index] > (forearm_thickness / 2):
                touching_on_dynamic_A[index] = False
            else:
                touching_on_dynamic_A[index] = True

            angle_to_dynamic_A[index] = angle_of_pos_relative_to_head_2d(
                wrist_pos_dynamic * 1, eye_angle_degrees, return_as_degrees=True
            )

            if not good_enough:
                print("warning: optimization not good enough.")

    return (
        wrist_pos_dynamic_A,
        elbow_pos_dynamic_A,
        theta_dynamic_A,
        touching_on_static_A,
        t_static_A,
        distance_to_static_A,
        touching_on_dynamic_A,
        t_dynamic_A,
        distance_to_dynamic_A,
        angle_to_dynamic_A,
    )
