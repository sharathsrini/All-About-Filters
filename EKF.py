import numpy as np
import math
import matplotlib.pyplot as plt

# Estimation parameter of EKF
Q = np.diag([0.1, 0.1, math.radians(1.0), 1.0])**2
R = np.diag([1.0, math.radians(40.0)])**2

#  Simulation parameter
Q_simulation = np.diag([0.5, 0.5])**2
R_simulation = np.diag([1.0, math.radians(30.0)])**2

DT = 0.1  # time tick [s]
SIMULATION_TIME = 50.0  # simulation time [s]

simulate = True


def input():
    v = 1.0  # [m/s]
    yaw_rate = 0.1  # [rad/s]
    u = np.matrix([v, yaw_rate]).T
    return u


def observation(x_True, x_d, u):

    x_True = motion_model(x_True, u)

    # add noise to gps x-y
    z_x = x_True[0, 0] + np.random.randn() * Q_simulation[0, 0]
    z_y = x_True[1, 0] + np.random.randn() * Q_simulation[1, 1]
    z = np.matrix([z_x, z_y])

    # add noise to input
    u_noise_1 = u[0, 0] + np.random.randn() * R_simulation[0, 0]
    u_noise_2 = u[1, 0] + np.random.randn() * R_simulation[1, 1]
    u_noise = np.matrix([u_noise_1, u_noise_2]).T

    x_d = motion_model(x_d, u_noise)

    return x_True, z, x_d, u_noise


def motion_model(x, u):

    F = np.matrix([[1.0, 0, 0, 0],
                   [0, 1.0, 0, 0],
                   [0, 0, 1.0, 0],
                   [0, 0, 0, 0]])

    B = np.matrix([[DT * math.cos(x[2, 0]), 0],
                   [DT * math.sin(x[2, 0]), 0],
                   [0.0, DT],
                   [1.0, 0.0]])

    x = F * x + B * u

    return x


def observation_model(x):
    #  Observation Model
    H = np.matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0]])

    z = H * x

    return z


def jacobF(x, u):
    # Jacobian of Motion Model
    yaw = x[2, 0]
    u1 = u[0, 0]
    jF = np.matrix([
        [1.0, 0.0, -DT * u1 * math.sin(yaw), DT * u1 * math.cos(yaw)],
        [0.0, 1.0, DT * math.cos(yaw), DT * math.sin(yaw)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

    return jF


def jacobH(x):
    # Jacobian of Observation Model
    j_H = np.matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    return j_H


def ekf_estimation(x_Estimated, P_Estimated, z, u):

    #  Predict
    x_Predicted = motion_model(x_Estimated, u)
    jF = jacobF(x_Predicted, u)
    P_Predicted = jF * P_Estimated * jF.T + Q

    #  Update
    j_H = jacobH(x_Predicted)
    z_Predicted = observation_model(x_Predicted)
    y = z.T - z_Predicted
    S = j_H * P_Predicted * j_H.T + R
    K = P_Predicted * j_H.T * np.linalg.inv(S)
    x_Estimated = x_Predicted + K * y
    P_Estimated = (np.eye(len(x_Estimated)) - K * j_H) * P_Predicted

    return x_Estimated, P_Estimated


def plot_covariance_ellipse(x_Estimated, P_Estimated):
    P_xy = P_Estimated[0:2, 0:2]
    eig_val, eig_vec = np.linalg.eig(P_xy)

    if eig_val[0] >= eig_val[1]:
        big_ind = 0
        small_ind = 1
    else:
        big_ind = 1
        small_ind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eig_val[big_ind])
    b = math.sqrt(eig_val[small_ind])
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eig_vec[big_ind, 1], eig_vec[big_ind, 0])
    R = np.matrix([[math.cos(angle), math.sin(angle)],
                   [-math.sin(angle), math.cos(angle)]])
    fx = R * np.matrix([x, y])
    px = np.array(fx[0, :] + x_Estimated[0, 0]).flatten()
    py = np.array(fx[1, :] + x_Estimated[1, 0]).flatten()
    plt.plot(px, py, "--r")


def main():
    print(__file__ + " start!!")

    time = 0.0

    # State Vector [x y yaw v]'
    x_Estimated = np.matrix(np.zeros((4, 1)))
    x_True = np.matrix(np.zeros((4, 1)))
    P_Estimated = np.eye(4)

    x_DR = np.matrix(np.zeros((4, 1)))  # Dead reckoning

    # history
    hx_Estimated = x_Estimated
    hx_True = x_True
    hx_DR = x_True
    hz = np.zeros((1, 2))

    while SIMULATION_TIME >= time:
        time += DT
        u = input()

        x_True, z, x_DR, u_d = observation(x_True, x_DR, u)

        x_Estimated, P_Estimated = ekf_estimation(x_Estimated, P_Estimated, z, u_d)

        # store data history
        hx_Estimated = np.hstack((hx_Estimated, x_Estimated))
        hx_DR = np.hstack((hx_DR, x_DR))
        hx_True = np.hstack((hx_True, x_True))
        hz = np.vstack((hz, z))

        if simulate:
            plt.cla()
            plt.plot(hz[:, 0], hz[:, 1], ".g")
            plt.plot(np.array(hx_True[0, :]).flatten(),
                     np.array(hx_True[1, :]).flatten(), "-b")
            plt.plot(np.array(hx_DR[0, :]).flatten(),
                     np.array(hx_DR[1, :]).flatten(), "-k")
            plt.plot(np.array(hx_Estimated[0, :]).flatten(),
                     np.array(hx_Estimated[1, :]).flatten(), "-r")
            plot_covariance_ellipse(x_Estimated, P_Estimated)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)


if __name__ == '__main__':
    main()
