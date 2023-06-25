import numpy as np
from scipy.optimize import minimize
from sympy.abc import t
from .constants import (
    shoulder_distance,
    upper_arm_length,
    forearm_length,
    head_distance,
    theta_limits_2d,
    theta_limits_2d_rad,
)


def transformation_matrix_A(theta, d, a, alpha):
    """
    calculate transformation matrix A,i containing position and orientation of joint frame i relative to joint frame i-1.
    based on Denavit-Hartenberg convention.
    calculating transformation matrix A,i is a sequence of four basic transformations.

    premises of DH convention: (1) x,i must be perpendicular to z,i-1 and (2) x,i must intersect with z,i-1.
    fulfilling these premises reduces amount of necessary information in transformation matrix A,i.
    transformation matrix A,i fulfills these premises.

    explanation of the four function parameters.
    each parameter describes one feature of frame i relative to frame i-1.

    theta: rotation around axis z,i-1
    d: offset in direction of z,i-1
    a: common normal between z,i-1 and z,i (offset in direction of x,i)
    alpha: angle around common normal axis (around x,i)

    each of the four basic frame transformations follow with explanation.
    each of the four basic frame transformations fulfills premises (1) and (2).

    "theta" transformation
    first, the frame will be rotated around the old z,i-1 axis.
    the z axis is always the axis around which joint i rotates.
    """

    Rot_z_theta_i = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    """"
    "d" transfomation
    the second transformation moves (translates) the frame alongside the old z,i-1 axis.
    """
    bt = np.c_[np.identity(3), np.array([0, 0, d])]
    Trans_z_d_i = np.vstack([bt, [0, 0, 0, 1]])

    """
    "a" transformation
    the third transformation moves (translates) the frame alongside the new x,i axis.
    given the first three transformations, we already know the final position of o,n (origin of frame i) relative to frame i-1.
    """
    ct = np.c_[np.identity(3), np.array([a, 0, 0])]
    Trans_x_a_i = np.vstack([ct, [0, 0, 0, 1]])

    """
    "alpha" transformation
    fourth, the frame will be rotated around the new x,i axis.
    naturally fulfilling premises (1) and (2) for any given alpha.
    """
    Rot_x_alpha_i = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha), 0],
            [0, np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 0, 1],
        ]
    )

    """
    finally, the four basic transformation matrices are multiplied.
    this results in the transformation matrix A,i, containing the position and orientation of frame i.
    """
    matrix_A = np.matmul(
        np.matmul(np.matmul(Rot_z_theta_i, Trans_z_d_i), Trans_x_a_i), Rot_x_alpha_i
    )
    return matrix_A


# wrist position relative to point between shoulders
def forward_kinematics(
    theta, wrist_left_right, round=True, return_wrist=True, return_elbow=False
):
    """
    root frame (from robot's view): x pointing "back", y pointing "right", z pointing "up".
    joint angles relative to z,i axis of frame i of joint i
    rotation according to right hand rule (rotation axis is thumb)
    """
    shoulder_yaw_angle = theta[0]
    shoulder_pitch_angle = theta[1]
    shoulder_roll_angle = theta[2]
    elbow_angle = theta[3]

    if wrist_left_right == "left":
        factor = 1
    elif wrist_left_right == "right":
        factor = -1

    """
    (1) theta: rotation around old z,i-1 axis
    (2) d: movement alongside old z,i-1 axis
    (3) a: movement alongside new x,i axis
    (4) alpha: rotation around new x,i axis
    """

    # calculating n transformation matrices A,i, where i = 1 to n

    A_01 = transformation_matrix_A(
        theta=shoulder_pitch_angle,
        d=factor * (-shoulder_distance),
        a=0,
        alpha=(-np.pi / 2),
    )

    A_12 = transformation_matrix_A(
        theta=factor * ((np.pi / 2) - shoulder_roll_angle),
        d=0,
        a=0,
        alpha=factor * (-np.pi / 2),
    )

    A_23 = transformation_matrix_A(
        theta=factor * ((np.pi / 2) + shoulder_yaw_angle),
        d=upper_arm_length,
        a=0,
        alpha=factor * (np.pi / 2),
    )

    A_34 = transformation_matrix_A(
        theta=factor * ((np.pi / 2) + elbow_angle), d=0, a=forearm_length, alpha=0
    )

    # multiplying all transformation matrices to get transformation matrix A_04, which contains wrist position relative to root reference frame.
    A_02 = np.matmul(A_01, A_12)
    # A_03 contains elbow position relative to root reference frame.
    A_03 = np.matmul(A_02, A_23)
    # A_04 contains wrist position relative to root reference frame.
    A_04 = np.matmul(A_03, A_34)

    """
    explanation of columns of A_i
    column 1, rows 1 to 3: direction of x axis relative to root coordinate frame.
    column 2, rows 1 to 3: direction of y axis relative to root coordinate frame.
    column 3, rows 1 to 3: direction of z axis relative to root coordinate frame.
    column 4, rows 1 to 3: joint position (x, y, z) relative to root coordinate frame.
    """

    wrist_pos = A_04.dot(np.array([0, 0, 0, 1]).T)[0:3]
    elbow_pos = A_03.dot(np.array([0, 0, 0, 1]).T)[0:3]
    all_pos = np.array([elbow_pos, wrist_pos])

    # round
    if round:
        for i, joint in enumerate(all_pos):
            for j, axis in enumerate(joint):
                all_pos[i, j] = float("{:.4f}".format(axis))

    if return_elbow and return_wrist:
        return all_pos
    elif return_elbow:
        return all_pos[0]
    else:
        return all_pos[1]


min_wrist_pos_right_2d = forward_kinematics(theta_limits_2d_rad[:, 0], "right")
max_wrist_pos_right_2d = forward_kinematics(theta_limits_2d_rad[:, 1], "right")

# inverse kinematics using optimization


def inverse_kinematics(target_pos, wrist_left_right, threshold, display):
    # cost function based on difference between target_pos and wrist_position(theta)
    def cost(theta):
        wrist_pos = forward_kinematics(theta, wrist_left_right, round=False)
        vec_difference = np.subtract(target_pos, wrist_pos)
        cost = np.matmul(vec_difference, vec_difference)
        return cost

    # initial theta
    theta = np.array([0, 0, 0, 0])
    good_enough = False

    distance_vec = np.zeros(3)
    distance = 0

    # keep optimizing until (new_pos(optimized_theta) == target_pos) or (i >= 10)
    i = 0
    while not (good_enough) and i < 10:
        min = minimize(cost, theta, method="SLSQP", options={"disp": display})
        theta = min.x
        new_pos = forward_kinematics(theta, wrist_left_right, round=True)

        distance_vec = np.subtract(new_pos, target_pos)
        distance = np.linalg.norm(distance_vec)
        if distance <= threshold:
            good_enough = True
        # perfect = np.array_equal(new_pos, target_pos)
        i += 1

    for j, angle in enumerate(theta):
        theta[j] = angle * (180 / np.pi)

    return theta


# inverse kinematics solely for creating the 2d data set
def inverse_kinematics_2d(target_pos, wrist_left_right, distance_threshold, display):
    # cost function based on difference between target_pos and wrist_position(theta)
    def cost(theta):
        wrist_pos = forward_kinematics(
            np.array([(np.pi / 2), (np.pi / 2), theta[0], theta[1]]),
            wrist_left_right,
            round=False,
        )
        vec_difference = np.subtract(target_pos, wrist_pos)
        cost_vec_difference = np.matmul(vec_difference, vec_difference)

        dif_max_shrl = theta_limits_2d_rad[2, 1] - theta[0]
        if dif_max_shrl > 0:
            dif_max_shrl = 0
        cost_max_shrl = 10 * dif_max_shrl**2

        dif_min_shrl = theta[0] - theta_limits_2d_rad[2, 0]
        if dif_min_shrl > 0:
            dif_min_shrl = 0
        cost_min_shrl = 10 * dif_min_shrl**2

        dif_max_elbw = theta_limits_2d_rad[3, 1] - theta[1]
        if dif_max_elbw > 0:
            dif_max_elbw = 0
        cost_max_elbw = 10 * dif_max_elbw**2

        dif_min_elbw = theta[1] - theta_limits_2d_rad[3, 0]
        if dif_min_elbw > 0:
            dif_min_elbw = 0
        cost_min_elbw = 10 * dif_min_elbw**2

        cost_theta_limits = (
            cost_max_shrl + cost_min_shrl + cost_max_elbw + cost_min_elbw
        )

        cost_total = cost_vec_difference + cost_theta_limits

        return cost_total

    # initial theta
    theta = random_theta_within_limits(radians=True)

    good_enough = False

    # keep optimizing until [np.linalg.norm(np.subtract(target_pos, new_pos(optimized_theta))) <= distance_threshold] OR [i > 9]
    i = 0
    while (not good_enough) and i < 10:
        opt_theta = theta[2::]
        min = minimize(cost, theta[2::], method="Nelder-Mead", options={"disp": False})
        opt_theta = min.x
        theta = np.array([(np.pi / 2), (np.pi / 2), opt_theta[0], opt_theta[1]])
        new_pos = forward_kinematics(theta, wrist_left_right, round=True)

        distance_vec = np.subtract(target_pos, new_pos)
        distance = np.linalg.norm(distance_vec)

        if distance <= distance_threshold:
            good_enough = True
        i += 1

    return theta, good_enough


def wrist_pos(theta, wrist_left_right, round=True):
    wrist_pos = forward_kinematics(
        theta, wrist_left_right, round, return_wrist=True, return_elbow=False
    )
    return wrist_pos


def elbow_pos(theta, wrist_left_right, round=True):
    elbow_pos = forward_kinematics(
        theta, wrist_left_right, round, return_wrist=False, return_elbow=True
    )
    return elbow_pos


def random_point_on_forearm(theta, wrist_left_right):
    elbow_pos, wrist_pos = forward_kinematics(
        theta,
        wrist_left_right=wrist_left_right,
        round=True,
        return_wrist=True,
        return_elbow=True,
    )
    forearm_vec = np.subtract(wrist_pos, elbow_pos)
    forearm_norm = np.linalg.norm(forearm_vec)
    forearm_unit_vec = forearm_vec / forearm_norm

    t_ = np.random.random()
    random_len = forearm_norm * t_
    random_point = np.add(elbow_pos, forearm_vec * random_len)

    return t_, random_point


"""
returns (1) closest point on forearm relative to wrist_pos_right, and (2) distance vector d
"""


def closest_point_on_left_forearm_to_wrist_pos(theta_left, wrist_pos_right, round=True):
    elbow_pos_left, wrist_pos_left = forward_kinematics(
        theta_left,
        wrist_left_right="left",
        round=round,
        return_wrist=True,
        return_elbow=True,
    )

    # forearm vector & unit vector of forearm (left elbow as origin)
    forearm = np.subtract(wrist_pos_left, elbow_pos_left)

    # right wrist vector (left elbow as origin)
    p = np.subtract(wrist_pos_right, elbow_pos_left)

    closest_point_on_left_forearm, distance_vec = closest_point_on_vector_to_point(
        forearm, p, wrist_pos_right
    )

    return closest_point_on_left_forearm, distance_vec


def closest_point_on_vector_to_point(vector, point):
    # idea: [vector] * [t*vector - point] = 0, whereas t is a real number.
    t_ = 1.0
    scalar = 0.0
    scalar_prev = (
        t_ * vector[0] ** 2
        - vector[0] * point[0]
        + t_ * vector[1] ** 2
        - vector[1] * point[1]
        + t_ * vector[2] ** 2
        - vector[2] * point[2]
    )
    direction = 1
    step_size = 0.1
    scalar_threshold = 0.00001
    iterate = True
    first_reverse = True

    while iterate:
        t_ += step_size * direction

        scalar = (
            t_ * vector[0] ** 2
            - vector[0] * point[0]
            + t_ * vector[1] ** 2
            - vector[1] * point[1]
            + t_ * vector[2] ** 2
            - vector[2] * point[2]
        )

        scalar = np.abs(scalar)

        if scalar <= scalar_threshold:
            iterate = False
        elif scalar > scalar_prev:
            direction = -direction
            if not first_reverse:
                step_size /= 10
            else:
                first_reverse = False
        scalar_prev = scalar

    t_ = float("{:.4f}".format(t_))

    # distance vector
    d = np.subtract(t_ * vector, point)
    # closest_point_on_vector = np.add(offset, distance_vec)

    return t_, d


"""
returns angle_xy, angle_xz, angle_yz
angle_xy: arcsin of [y / hypotenuse(x, y)]
angle_xz: arcsin of [z / hypotenuse(x, z)]
angle_yz: arcsin of [z / hypotenuse(y, z)]
"""


def angles_of_pos_relative_to_head(pos, return_as_degrees=True):
    pos[0] -= head_distance

    x = pos[0]
    y = pos[1]
    z = pos[2]

    vec_x = np.array([x, 0, 0])
    vec_y = np.array([0, y, 0])
    vec_z = np.array([0, 0, z])

    vec_xy = np.add(vec_x, vec_y)
    vec_xz = np.add(vec_x, vec_z)
    vec_yz = np.add(vec_y, vec_z)

    hypotenuse_xy = np.linalg.norm(vec_xy)
    hypotenuse_xz = np.linalg.norm(vec_xz)
    hypotenuse_yz = np.linalg.norm(vec_yz)

    sin_xy = y / hypotenuse_xy
    sin_xz = z / hypotenuse_xz
    sin_yz = z / hypotenuse_yz

    angle_xy = np.arcsin(sin_xy)
    angle_xz = np.arcsin(sin_xz)
    angle_yz = np.arcsin(sin_yz)

    if return_as_degrees:
        angle_xy *= 360 / (2 * np.pi)
        angle_xz *= 360 / (2 * np.pi)
        angle_yz *= 360 / (2 * np.pi)

    return angle_xy, angle_xz, angle_yz


def angle_of_pos_relative_to_head_2d(pos, eye_angle_degrees, return_as_degrees=True):
    y = pos[1]
    z = pos[2]

    vec_y = np.array([0, y, 0])
    vec_z = np.array([0, 0, z])

    vec_yz = np.add(vec_y, vec_z)

    hypotenuse_yz = np.linalg.norm(vec_yz)

    sin_yz = z / hypotenuse_yz

    angle_yz = np.arcsin(sin_yz)

    if return_as_degrees:
        angle_yz *= 360 / (2 * np.pi)
        angle_yz += eye_angle_degrees

    return angle_yz


def random_theta_within_limits(radians=True):
    theta = np.zeros(4)

    for i in range(4):
        theta_min = theta_limits_2d[i, 0]
        theta_max = theta_limits_2d[i, 1]

        if theta_min == theta_max:
            theta[i] = theta_min
        else:
            theta[i] = np.random.randint(theta_min, theta_max)

    if radians:
        theta = np.radians(theta)

    return theta
