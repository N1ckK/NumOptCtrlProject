import numpy as np
import casadi as ca
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from math import sin, cos, pi


grav_const = 0.1


def rk4step(ode, h, x):
    """ one step of explicit Runge-Kutta scheme of order four (RK4)

    parameters:
    ode -- odinary differential equations (your system dynamics)
    h -- step of integration
    x -- states
    """
    k1 = ode(x)
    k2 = ode(x + h * 0.5 * k1)
    k3 = ode(x + h * 0.5 * k2)
    k4 = ode(x + h * k3)
    return x + ((h / 6) * (k1 + 2 * k2 + 2 * k3 + k4))


def rk4step_u(ode, h, x, u):
    """ one step of explicit Runge-Kutta scheme of order four (RK4)

    parameters:
    ode -- odinary differential equations (your system dynamics)
    h -- step of integration
    x -- states
    u -- controls
    """
    k1 = ode(x, u)
    k2 = ode(x + h * 0.5 * k1, u)
    k3 = ode(x + h * 0.5 * k2, u)
    k4 = ode(x + h * k3, u)
    return x + ((h / 6) * (k1 + 2 * k2 + 2 * k3 + k4))



def ode(z: np.ndarray, body_masses: tuple,
        n_body: int, dimension: int) -> np.ndarray:
    '''
        The right hand side of
            z' = F(z)

        where F are the dynamics of the n-body problem in d dimensions and
        z = z.

        z is of the form:

        z = [
            body_1 x_1 position,
            body_1 x_2 position,
            ...
            body_1 x_d position,
            body_2 x_1 position,
            body_2 x_2 position,
            ...
            body_2 x_d position,
            ...
            ...
            body_n x_1 position,
            body_n x_2 position,
            ...
            body_n x_d position,
            body_1 x_1 velocity,
            body_1 x_2 velocity,
            ...
            body_1 x_d velocity,
            body_2 x_1 velocity,
            body_2 x_2 velocity,
            ...
            body_2 x_d velocity,
            ...
            ...
            body_n x_1 velocity
            body_n x_2 velocity
            ...
            body_n x_d velocity
        ]

        this vector has the dimension 2 * n * d where n = n_body is the number
        of bodies and d = dimension is the dimension.

        body_masses is a tuple (m_1, ..., m_n) of body masses.
    '''
    # dimension of the output vector / dimension of the input vector
    N = 2 * dimension * n_body
    # arrays for the new output
    rhs_velocity = np.zeros(N // 2)
    rhs_acceleration = np.zeros(N // 2)

    # update every body after one another
    for i in range(0, n_body):
        # get the velocity of the i_th body from the input vector
        body_velocity_i = z[n_body * dimension + (i * dimension):
            n_body * dimension + ((i + 1) * dimension)]

        # set the velocity in the output vector
        rhs_velocity[i * dimension : (i + 1) * dimension] = body_velocity_i

        # now compute the new accelleration for each body
        for j in range(0, n_body):
            if i == j:
                continue

            # get the body positions from the input vector
            body_position_i = z[(i * dimension): ((i + 1) * dimension)]
            body_position_j = z[(j * dimension): ((j + 1) * dimension)]

            # calculate the force vector from body i to body j:
            rhs_acceleration[i * dimension : (i + 1) * dimension] += (grav_const * body_masses[j]
                             / (np.linalg.norm(body_position_i - body_position_j) ** 3)
                             * (body_position_j - body_position_i))

    return np.array([rhs_velocity, rhs_acceleration]).flatten()


def ode_controllable(z: np.ndarray, controls: np.ndarray, body_masses: tuple,
        n_body: int, dimension: int, thrust_max: float) -> np.ndarray:
    '''
        The right hand side of
            z' = F(z, u)

        where u are controls (theta, r) for body_1 ie.
        (alpha, theta) specifies the rocket direction and r the thrust.
    '''
    N = 2 * dimension * n_body
    rhs_velocity = np.zeros(N // 2)
    rhs_acceleration = np.zeros(N // 2)

    for i in range(0, n_body):

        body_velocity_i = z[n_body * dimension + (i * dimension):
            n_body * dimension + ((i + 1) * dimension)]
        rhs_velocity[i * dimension : (i + 1) * dimension] = body_velocity_i
        for j in range(0, n_body):
            if i == j:
                continue

            body_position_i = z[(i * dimension): ((i + 1) * dimension)]
            body_position_j = z[(j * dimension): ((j + 1) * dimension)]
            rhs_acceleration[i * dimension : (i + 1) * dimension] += (grav_const * body_masses[j]
                             / (np.linalg.norm(body_position_i - body_position_j) ** 3)
                             * (body_position_j - body_position_i))

        if i == 0:
            # r < thrust_max
            # phi_new - phi_old < max_turn_speed
            # theta_new - phi_old < max_turn_speed

            # add controls
            r, theta = controls
            r = min(r, thrust_max)
            rhs_acceleration[i * dimension : (i + 1) * dimension] += np.array([r * cos(theta),
                                                                      r * sin(theta)])

    return np.array([rhs_velocity, rhs_acceleration]).flatten()


def ode_controllable_casadi(z: np.ndarray, controls: np.ndarray, body_masses: tuple,
        n_body: int, dimension: int, thrust_max: float) -> np.ndarray:
    '''
        The right hand side of
            z' = F(z, u)

        where u are controls (alpha, theta, r) for body_1 ie.
        (alpha, theta) specifies the rocket direction and r the thrust.
    '''
    N = 2 * dimension * n_body
    rhs_velocity = []
    rhs_acceleration = []

    for i in range(0, n_body):

        body_velocity_i = z[n_body * dimension + (i * dimension):
            n_body * dimension + ((i + 1) * dimension)]

        rhs_velocity = ca.vertcat(rhs_velocity, body_velocity_i)

        rhs_acceleration_i = 0

        for j in range(0, n_body):
            if i == j:
                continue

            body_position_i = z[(i * dimension): ((i + 1) * dimension)]
            body_position_j = z[(j * dimension): ((j + 1) * dimension)]
            rhs_acceleration_i += (grav_const * body_masses[j]
                             / (ca.norm_2(body_position_i - body_position_j) ** 3)
                             * (body_position_j - body_position_i))

        if i == 0:
            r = controls[0]
            theta = controls[1]
            rhs_acceleration_i[0] += r * ca.cos(theta)
            rhs_acceleration_i[1] += r * ca.sin(theta)

            #+= ca.vertcat(r * ca.cos(theta), r * ca.sin(theta))

        rhs_acceleration = ca.vertcat(rhs_acceleration, rhs_acceleration_i)

    return ca.vertcat(rhs_velocity, rhs_acceleration)
