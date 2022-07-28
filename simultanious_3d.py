import numpy as np
import casadi as ca
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ode import *

from math import *

# Time horizon
T = 750
# Stepsize
h = 2 ** (3) # T / N
# Number of discrete time points
N = floor(T/h)  # 160
# Gravitational constant
grav_const = 0.00006

# masses of the simulated bodies (rocket, planet)
sun_mass = 10000000
rocket_mass = 0.05
# rocket is always body_0, planet is always body_-1
body_masses = (rocket_mass, sun_mass)
# number of bodies in orbit
n_body = 1
# dimension to simulate in
dimension = 3
# maximum thrust of the rocket
thrust_max = 0.01

# initial positions and velocites of bodies:

# body1_pos.x, body1_pos.y, body2_pos.x, body2_pos.y,
# body1_vel.x, body1_vel.y, body2_vel.x, body2_vel.y,

rr = sqrt(0.3 ** 2 + 3 ** 2)

# radius of the planet
surface = 100

x_0_bar = [surface, 0, 0,
           0, 0, 0]

# desired circular orbit height
orbit = 190


# rotational matrix
#Q = np.identity(3)

theta_x = 0.01
theta_y = 0
theta_z = 0


rot_matrix_x = np.array([[1, 0, 0],
                        [0, ca.cos(theta_x), -ca.sin(theta_x)],
                        [0, ca.sin(theta_x), ca.cos(theta_x)]])

rot_matrix_y = np.array([[ca.cos(theta_y), 0 , ca.sin(theta_y)],
                         [0, 1, 0],
                         [-ca.sin(theta_y), 0 , ca.cos(theta_y)]])

rot_matrix_z = np.array([[ca.cos(theta_z), -ca.sin(theta_z), 0],
                        [ca.sin(theta_z), ca.cos(theta_z), 0],
                        [0, 0, 1]])


Q = rot_matrix_x @ rot_matrix_y @ rot_matrix_z


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
    rhs_velocity = []
    rhs_acceleration = []

    # iterate on all movable bodies around the planet
    for i in range(0, n_body):

        body_velocity_i = z[n_body * dimension + (i * dimension):
            n_body * dimension + ((i + 1) * dimension)]

        rhs_velocity = ca.vertcat(rhs_velocity, body_velocity_i)

        rhs_acceleration_i = 0

        # calculate the force of body_j on body_i
        for j in range(0, n_body):
            # no force of a body on itself
            if i == j:
                continue

            body_position_i = z[(i * dimension): ((i + 1) * dimension)]
            body_position_j = z[(j * dimension): ((j + 1) * dimension)]

            rhs_acceleration_i += (grav_const * body_masses[j]
                             / (ca.norm_2(body_position_i - body_position_j) ** 3)
                             * (body_position_j - body_position_i))

        # calculate the force of the planet on body_i
        body_position_i = z[(i * dimension): ((i + 1) * dimension)]
        planet_position = [0] * dimension

        rhs_acceleration_i += (grav_const * body_masses[-1]
                             / (ca.norm_2(body_position_i - planet_position) ** 3)
                             * (planet_position - body_position_i))

        # body_0 is the actuated rocket
        if i == 0:
            r = controls[0]
            phi = controls[1]
            theta = controls[2]
            rhs_acceleration_i[0] += r * ca.sin(phi) * ca.cos(theta) / body_masses[i]
            rhs_acceleration_i[1] += r * ca.sin(phi) * ca.sin(theta) / body_masses[i]
            rhs_acceleration_i[2] += r * ca.cos(phi) / body_masses[i]

        rhs_acceleration = ca.vertcat(rhs_acceleration, rhs_acceleration_i)

    return ca.vertcat(rhs_velocity, rhs_acceleration)


def ode(z, controls):
    return ode_general(z, controls, body_masses, n_body, dimension)


state_dimension = 2 * n_body * dimension

x_single = ca.SX.sym('x', state_dimension)
u_single = ca.SX.sym('u', dimension)
step_h = ca.SX.sym('h')

dynamics = ca.Function('d', [x_single, u_single, step_h], [rk4step_u(
    ode, step_h, x_single, u_single
)])

# N+1 states, position and velocity, n_body bodies, dimenstion dimensions
x = ca.SX.sym('x', (N + 1) * state_dimension)
# N states, thrust and dimension-1 angles
u = ca.SX.sym('u', N * dimension)


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

orbital_vel = sqrt(sun_mass * grav_const / orbit)


def cost_function_continous(t_current, x_current, u_current=None):
    # dist = ca.norm_2(x_current[0: dimension] - x_current[dimension: 2 * dimension])
    # return t_current * orbit * (1 - dist / orbit) / dist + (dist / orbit) ** 3
    # return t_current * (dist - orbit) ** 2 + 500 * u_current[0] ** 2
    # return t_current * (ca.exp(dist/orbit) / (dist/orbit)) / 30
    # dist_to_surface = sqrt((x_current[0]-x_current[2]) ** 2 + (x_current[1]-x_current[3]) ** 2) - surface
    # return exp(orbit/dist_to_surface) * exp(dist_to_surface/orbit)

    #return ca.sqrt(u_current[0] ** 2 + thrust_max ** 2 / 10)
    return u_current[0] ** 2


# def euclidean_of_polar2(polar_vector_a, polar_vector_b):
#     '''
#         Calculates the euclidean norm of a two dimensional vector given in polar coordinates.
#     '''
#     eucl_vector_a = [polar_vector_a[0] * cos(polar_vector_a[1]), polar_vector_a[0] * sin(polar_vector_a[1])]
#     eucl_vector_b = [polar_vector_b[0] * cos(polar_vector_b[1]), polar_vector_b[0] * sin(polar_vector_b[1])]
#     return ca.norm_2(eucl_vector_a - eucl_vector_b)


def cost_function_integral_discrete(x, u):
    '''
        Computes the discretized cost of given state and control variables to
        be minimized, using Simpson's rule.
    '''
    cost = h / 6 * (cost_function_continous(0, x[:state_dimension], u[:dimension])
                    + cost_function_continous(T, x[-state_dimension:], u[-dimension:]))
    # First and last term in Simpson, both appear only once
    x_halfstep = dynamics(x[:state_dimension], u[:dimension], h / 2)
    cost += h / 3 * cost_function_continous(h / 2, x_halfstep, u[:dimension])
    # First half step of Simpson, not treated within the for-loop
    for i in range(1, N):
        x_halfstep = dynamics(x[i*state_dimension:(i+1)*state_dimension], u[i*dimension:(i+1)*dimension], h / 2)
        cost += h / 3 * (cost_function_continous(i * h, x[i*state_dimension:(i+1)*state_dimension], u[dimension*i:dimension*(i+1)])
                         # Each of the other non half step terms appears twice
                         + 2 * cost_function_continous((i + 1/2) * h, x_halfstep, u[dimension*i:dimension*(i+1)]))
                         # The remaining half steps
        # cost += euclidean_of_polar2(u[2*(i-1):2*i], u[2*i:2*(i+1)]) / h
        # Experimental, punish changes in controls
    return cost


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
        # x_k+1 - F(x_k, u_k)
        x[(i+1) * state_dimension : (i+2) * state_dimension] - dynamics(
            x[i * state_dimension : (i+1) * state_dimension],
            u[dimension * i : dimension * (i+1)],
            h
        )
    )
    lbg += [0] * state_dimension
    ubg += [0] * state_dimension

# stay above surface and close to orbit
for i in range(0, N):
    x_current = x[i * state_dimension : (i+1) * state_dimension]
    constraints.append(ca.norm_2(x_current[0:dimension]))
    lbg += [1.0 * surface]
    ubg += [1.3 * orbit]

# thrust_max <= u_k1 = r_k <= thrust_max
constraints.append(u[::dimension])
lbg += [-thrust_max] * N
ubg += [thrust_max] * N

# -pi <= u_k2 = theta_k <= pi
# constraints.append(u[1::dimension])
# lbg += [-pi] * N
# ubg += [pi] * N

# Terminal constraints
x_terminal = x[N * state_dimension: (N+1) * state_dimension]

#x_terminal_rocket = x_terminal[0:dimension]
#x_terminal_rocket_rotated = ca.mtimes(Q, x_terminal_rocket)

# reach orbital velocity
constraints.append(
    ca.norm_2(x_terminal[n_body * dimension : (n_body + 1) * dimension]) - orbital_vel
)

lbg += [0]
ubg += [0]

# velocity perpendicular to orbit normal
constraints.append(
    ca.dot(x_terminal[n_body * dimension : (n_body + 1) * dimension], x_terminal[0:dimension])
)

lbg += [0]
ubg += [0]

#---

orbit_normal = ca.mtimes(Q.T, [0, 1, 0])

# velocity is perpendicular to orbit binormal
constraints.append(
    ca.dot(x_terminal[n_body * dimension : (n_body + 1) * dimension], orbit_normal)
)

lbg += [0]
ubg += [0]

# rhs_acceleration_0 = 0
#
# for j in range(1, n_body):
#     body_position_0 = x_terminal[(0 * dimension): ((0 + 1) * dimension)]
#     body_position_j = x_terminal[(j * dimension): ((j + 1) * dimension)]
#
#     rhs_acceleration_0 += (grav_const * body_masses[j]
#                      / (ca.norm_2(body_position_0 - body_position_j) ** 3)
#                      * (body_position_j - body_position_0))
#
# constraints.append(
#                   rhs_acceleration_0[1]
# )
#
# lbg += [0]
# ubg += [0]

#---

# rocket is on the orbit
constraints.append(
            ca.mtimes(Q, x_terminal[0:dimension])[1]
)

lbg += [0]
ubg += [0]

# rocket has correct distance to the planet
constraints.append(
    ca.norm_2(x_terminal[0:dimension]) - orbit
)

lbg += [0]
ubg += [0]


constraints = ca.vertcat(*constraints)

nlp = {'x': ca.vertcat(x, u), 'f': cost_function_integral_discrete(x, u),
       'g': constraints}

solver = ca.nlpsol('solver', 'ipopt', nlp)

# build initial guess

x_initial = [surface, 0, 0,
             rr * sin(pi/4) * cos(0.01), rr * sin(0.01) * sin(pi/4), rr * cos(pi/4)]
#u_initial = [rr, pi/3, 0] + [0] * (3 * N - 3)

u_initial = [0] * (3 * N)

for i in range(N):
    x_initial = np.concatenate(
        (
            x_initial, dynamics(x_initial[i * state_dimension : (i + 1) * state_dimension],
                                  u_initial[dimension * i : dimension * (i + 1)], h).full().flatten()
        )
    )

initial_guess = np.concatenate((x_initial, u_initial)).tolist()

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

# print(optimal_variables[(N + 1) * state_dimension+1:])

optimal_trajectory = np.reshape(
    optimal_variables[:(N + 1) * state_dimension],
    (N + 1, state_dimension)
)[:,0:6]

optimal_controls = np.reshape(
    optimal_variables[(N + 1) * state_dimension:],
    (N, 3))

#with open("initial.txt", "w") as as file:
#    file.write(optimal_controls)

optimal_controls = np.vstack([optimal_controls, np.array([0, 0, 0])])

print(ca.norm_2(optimal_trajectory[N,3:6]), orbital_vel)
print(optimal_trajectory[N,:])

print(optimal_controls[:10,:])

terminal_sim = 500

for i in range(N, N+terminal_sim):
    optimal_trajectory = np.vstack([optimal_trajectory,
                                    dynamics(optimal_trajectory[i,:], [0] * 3, h).full().flatten()])
    optimal_controls = np.vstack([optimal_controls, np.array([0, 0, 0])])

fig = plt.figure()
ax = plt.axes(projection='3d')
lines = []
dots = []
objects = []

vrf = 50 / thrust_max # vector rescale factor

# vector = ax.annotate("", xy=(
#                              optimal_trajectory[0,0 * dimension],
#                              optimal_trajectory[0,0 * dimension + 1]
#                              ), xytext=(
#                                         optimal_trajectory[0,0 * dimension] + vrf * optimal_controls[0, 0] * cos(optimal_controls[0, 1]),
#                                         optimal_trajectory[0,0 * dimension + 1] + vrf * optimal_controls[0, 0] * sin(optimal_controls[0, 1])),
#                                         arrowprops={"facecolor": "red"})

#objects.append(vector)

#objects.append(0)

for b_index in range(n_body):
    line, = ax.plot3D(optimal_trajectory[:,b_index * dimension],
                      optimal_trajectory[:,b_index * dimension + 2],
                      optimal_trajectory[:,b_index * dimension + 1],
                      '--', alpha=0.6)

    dot, = ax.plot3D(optimal_trajectory[0,b_index * dimension],
                    optimal_trajectory[0,b_index * dimension + 2],
                    optimal_trajectory[0,b_index * dimension + 1],
                    'bo', alpha=1)

    dots.append(dot)
    lines.append(line)
    objects.append(line)
    objects.append(dot)

def update(num, optimal_trajectory, objects):
    # print(cost_function_continous(num, optimal_trajectory[num,:]))

    # objects[0].xy = (
    #                  optimal_trajectory[num,0 * dimension] + vrf * optimal_controls[num, 0] * cos(optimal_controls[num, 1]),
    #                  optimal_trajectory[num,0 * dimension + 1] + vrf * optimal_controls[num, 0] * sin(optimal_controls[num, 1])
    # )
    #
    # objects[0].set_position(
    #         (optimal_trajectory[num,0 * dimension],
    #         optimal_trajectory[num,0 * dimension + 1])
    # )

    for b_index in range(n_body):
        sli = 0
        #sli = max(0, num - 30) # start line index
        objects[-1 + 1 + 2 * b_index].set_data(optimal_trajectory[sli:num,b_index * dimension],
                                optimal_trajectory[sli:num,b_index * dimension + 2])
        objects[-1 + 1 + 2 * b_index].set_3d_properties(optimal_trajectory[sli:num,b_index * dimension + 1])

        objects[-1 + 1 + 2 * b_index + 1].set_data(optimal_trajectory[num,b_index * dimension],
                                optimal_trajectory[num,b_index * dimension + 2])
        objects[-1 + 1 + 2 * b_index + 1].set_3d_properties(optimal_trajectory[num,b_index * dimension + 1])
    return objects

ani = animation.FuncAnimation(fig, update, fargs=[optimal_trajectory, objects],
                          interval=N // 5, blit=True, frames=optimal_trajectory.shape[0])

# circle = plt.Circle((0,0), 190, fill=False, alpha=0.03)
# ax.add_patch(circle)
#
# circle = plt.Circle((0,0), 100, fill=True)
# ax.add_patch(circle)

ax.set_xlim((-200, 200))
ax.set_ylim((-200, 200))
ax.set_zlim((-8, 8))

#ax.set_aspect('auto', adjustable='box')

import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter

plt.show()

# import PIL
# ani.save('swing_by_3d.gif', writer='imagemagick', fps=120)



# import pickle
# pickle.dump(optimal_variables, open('Optimal controls.pickle', 'wb'))

# TODO: (Nick)
# n-body problem beschreiben (Herleitung)
# ocp diskretisieren

# TODO: (Michael)
# Einleitung zum Report schreiben
# Diskretisierung (Simpson-Regel) beschreiben

# TODO: (both)
# Code kommentieren
