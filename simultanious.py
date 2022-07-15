import numpy as np
import casadi as ca
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ode import *

from math import *

# Time horizon
T = 100
# Number of discrete time points
N = 1000
# Stepsize
h = T / N
# Gravitational constant
grav_const = 0.08

# masses of the simulated bodies (rocket, planet)
body_masses = (0.05, 10000)
# number of bodies
n_body = 2
# dimension to simulate in
dimension = 2
# maximum thrust of the rocket
thrust_max = 1.5

# initial positions and velocites of bodies:

# body1_pos.x, body1_pos.y, body2_pos.x, body2_pos.y,
# body1_vel.x, body1_vel.y, body2_vel.x, body2_vel.y,
x_0_bar = [100, 0, 0, 0,
           -0.7, 0.8, 0.005, 0.005]

# desired circular orbit height
orbit = 20

# just some rescale factor coming from the cost function that was chosen
cost_function_rescale_factor = 0.759835685652


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


def ode_general(z: np.ndarray, controls: np.ndarray, body_masses: tuple,
        n_body: int, dimension: int) -> np.ndarray:
    '''
        The right hand side of
            z' = f(z, u)

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
            rhs_acceleration_i[0] += r * ca.cos(theta) / body_masses[i]
            rhs_acceleration_i[1] += r * ca.sin(theta) / body_masses[i]

        rhs_acceleration = ca.vertcat(rhs_acceleration, rhs_acceleration_i)

    return ca.vertcat(rhs_velocity, rhs_acceleration)


def ode(z, controls):
    return ode_general(z, controls, body_masses, dimension, n_body)


state_dimension = 2 * n_body * dimension

x_single = ca.SX.sym('x', state_dimension)
u_single = ca.SX.sym('u', 2)
step_h = ca.SX.sym('h')

dynamics = ca.Function('d', [x_single, u_single, step_h], [rk4step_u(
    ode, step_h, x_single, u_single
)])

#  N+1 states, position and velocity, n_body bodies, dimenstion dimensions
x = ca.SX.sym('x', (N + 1) * state_dimension)
u = ca.SX.sym('u', 2 * N)


# 1. Opt.
# x_0 --- x_1 ---- x_2 --- ... --- x_n
# u_0 --- u_1 ---- u_2 --- ... --- u_n

# 2. Simul.
# x_0 --- x_1 ---- x_2 --- ... --- x_n
# u_0 --- u_1 ---- u_2 --- ... --- u_n
#      ^
# x_0 = x_00 -- x_01 -- x_02 -- x_03 = x_1
# u_0 --------------------------------- u_1
# u_0 = u_00    u_01    u02    u03  =   u_1
#
#
# x(t_k) ------------------------------ x(t_k+1)


def cost_function_continous(t_current, x_current, u_current=None):
    dist = ca.norm_2(x[0: dimension] - x[dimension: 2 * dimension]) / cost_function_rescale_factor
    return orbit * (1 - dist / orbit) / dist + (dist / orbit) ** 3
    # return t_current * exp(1/x) * exp(x)

def cost_function_integral_discrete(x, u):
    '''
        Computes the discretized cost of given state and control variables to
        be minimized, using Simpson's rule.
    '''
    cost = h / 6 * (cost_function_continous(0, x[:2*n_body*dimension])
                    + cost_function_continous(T, x[-2*n_body*dimension:]))
    x_halfstep = dynamics(x[:2*n_body*dimension], u[:2], h / 2)
    cost += h / 3 * cost_function_continous(h / 2, x_halfstep)
    for i in range(1, N):
        x_halfstep = dynamics(x[i*2*n_body*dimension:(i+1)*2*n_body*dimension], u[i*2:(i+1)*2], h / 2)
                                                #####
        cost += h / 3 * (cost_function_continous(i * h, x[i*2*n_body*dimension:(i+1)*2*n_body*dimension])
                         + 2 * cost_function_continous((i + 1/2) * h, x_halfstep))
    return cost
    #
    # function T = simpson(n, f, I)
    # len = I(2) - I(1);
    # T = (len / (6 * n)) * (f(I(1)) + f(I(2)));
    # for i = 1:n-1
    #     T = T + (len / (3 * n)) * (f(I(1) + (len / n) * i) + 2 * f(I(1) + (len / n) * (i - (1/2))));
    # end
    # T = T + (len / (3 * n)) * f(I(1) + (len / n) * (n - (1/2)));
    # end


# def int_simpson(function, a: float, b: float) -> float:
#     '''
#         Uses the simpson-quadrature to approximate the integral of function
#         from a to b.
#     '''
#         return (b - a) / 6 * (function(a) + 4 * function((a + b) / 2)
#                               + function(b))


# build nlp

constraints = []
lbg = []
ubg = []

# x_0 = x_0_bar
constraints.append(x[0:state_dimension] - x_0_bar)
lbg += [0] * state_dimension
ubg += [0] * state_dimension

for i in range(0, N):
    constraints.append(
        # x_k+1 - F(xk, uk)
        x[(i+1) * state_dimension : (i+2) * state_dimension] - dynamics(
            x[i * state_dimension : (i+1) * state_dimension],
            u[2*i:2*i+1],
            h
        )
    )
    lbg += [0] * state_dimension
    ubg += [0] * state_dimension

# thrust_max <= u_k1 = r_k <= thrust_max
constraints.append(u[::2])
lbg += [thrust_max / 3] * N
ubg += [thrust_max] * N

# -pi <= u_k2 = theta_k <= pi
constraints.append(u[1::2])
lbg += [-pi] * N
ubg += [pi] * N

constraints = ca.vertcat(*constraints)

nlp = {'x': ca.vertcat(x, u), 'f': cost_function_integral_discrete(x, u),
       'g': constraints}

solver = ca.nlpsol('solver', 'ipopt', nlp)

# build initial guess

x_initial = x_0_bar.copy()
u_initial = [thrust_max / 2, pi] * N
for i in range(N):
    x_initial = np.concatenate(
        (
            x_initial, dynamics(x_initial[i*state_dimension:(i + 1) * state_dimension],
                                  u_initial[2*i:2*i+2], h).full().flatten()
        )
    )

initial_guess = np.concatenate((x_initial,u_initial)).tolist()

#import pdb; pdb.set_trace()

# Solve the NLP
res = solver(
    x0 = initial_guess,    # solution guess
    lbx = -ca.inf,          # lower bound on x
    ubx = ca.inf,           # upper bound on x
    lbg = lbg,                # lower bound on g
    ubg = ubg,                # upper bound on g
)

optimal_variables = res["x"].full()

print(optimal_variables[(N + 1) * state_dimension+1:])

optimal_trajectory = np.reshape(
    optimal_variables[:(N + 1) * state_dimension],
    (N + 1, state_dimension)
)[:,0:4]

fig, ax = plt.subplots()
lines = []

for b_index in range(n_body):
    line, = ax.plot(optimal_trajectory[:,b_index * dimension],
                    optimal_trajectory[:,b_index * dimension + 1])
    lines.append(line)

def update(num, optimal_trajectory, lines):
    for b_index in range(n_body):
        lines[b_index].set_data(optimal_trajectory[:num,b_index * dimension],
                                optimal_trajectory[:num,b_index * dimension + 1])
    return lines

ani = animation.FuncAnimation(fig, update, fargs=[optimal_trajectory, lines],
                          interval=25, blit=True, frames=N)

plt.show()


# TODO: (Nick)
##### (initial value)
##### constraints implementieren
# n-body problem beschreiben (Herleitung)
# ocp diskretisieren

# TODO: (Michael)
# Einleitung zum Report schreiben
# Diskretisierung (Simpson-Regel) beschreiben

# TODO: (both)
# Code kommentieren
