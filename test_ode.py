import numpy as np
import casadi as ca
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ode import *

from math import *


def test_ode_2b_2d():
    '''
        Tests ode defined in ode.py with two bodies in two dimensions
    '''
    dimension = 2
    n_body = 2
    body_position = np.array([np.array([1, 0]), np.array([0, 1])])
    body_velocity = np.array([np.array([1, 1]), np.array([-1, -1])])
    body_masses = (3, 4)

    N_T = 800
    solution = np.zeros((N_T, 2 * n_body * dimension))
    solution[0,0:n_body*dimension] = body_position.flatten()
    solution[0,n_body*dimension:] = body_velocity.flatten()

    ode_auto = lambda x: ode(x, body_masses, n_body, dimension)

    for i in range(N_T - 1):
        solution[i+1,:] = rk4step(ode_auto, 0.08, solution[i,:])

    fig, ax = plt.subplots()

    lines = []

    for b_index in range(n_body):
        line, = ax.plot(solution[:,b_index * dimension],solution[:,b_index * dimension + 1])
        lines.append(line)

    def update(num, solution, lines):
        for b_index in range(n_body):
            lines[b_index].set_data(solution[:num,b_index * dimension],
                                    solution[:num,b_index * dimension + 1])
        return lines

    ani = animation.FuncAnimation(fig, update, fargs=[solution, lines],
                              interval=25, blit=True, frames=N_T)

    plt.show()


def test_ode_3b_2d():
    '''
        Tests ode defined in ode.py with three bodies in two dimensions
    '''
    dimension = 2
    n_body = 3
    body_position = np.array([np.array([0, 1]), np.array([0, -1]), np.array([5, 5])])
    body_velocity = np.array([np.array([1, 0]), np.array([-1, 0]), np.array([1, -1])])
    body_masses = (3, 4, 3)

    N_T = 800
    solution = np.zeros((N_T, 2 * n_body * dimension))
    solution[0,0:n_body*dimension] = body_position.flatten()
    solution[0,n_body*dimension:] = body_velocity.flatten()

    ode_auto = lambda x: ode(x, body_masses, n_body, dimension)

    for i in range(N_T - 1):
        solution[i+1,:] = rk4step(ode_auto, 0.08, solution[i,:])

    fig, ax = plt.subplots()

    lines = []

    for b_index in range(n_body):
        line, = ax.plot(solution[:,b_index * dimension],solution[:,b_index * dimension + 1])
        lines.append(line)

    def update(num, solution, lines):
        for b_index in range(n_body):
            lines[b_index].set_data(solution[:num,b_index * dimension],
                                    solution[:num,b_index * dimension + 1])
        return lines

    ani = animation.FuncAnimation(fig, update, fargs=[solution, lines],
                              interval=25, blit=True, frames=N_T)

    plt.show()


def test_ode_2b_3d():
    '''
        Tests ode defined in ode.py with two bodies in two dimensions
    '''
    dimension = 3
    n_body = 2
    body_position = np.array([np.array([1, 0, 0]), np.array([0, 1, 0])])
    body_velocity = np.array([np.array([1, 1, 0.1]), np.array([-1, -1, 0])])
    body_masses = (3, 4)

    N_T = 800
    solution = np.zeros((N_T, 2 * n_body * dimension))
    solution[0,0:n_body*dimension] = body_position.flatten()
    solution[0,n_body*dimension:] = body_velocity.flatten()

    ode_auto = lambda x: ode(x, body_masses, n_body, dimension)

    for i in range(N_T - 1):
        solution[i+1,:] = rk4step(ode_auto, 0.08, solution[i,:])

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    lines = []

    for b_index in range(n_body):
        line, = ax.plot3D(solution[:,b_index * dimension],
                          solution[:,b_index * dimension + 1],
                          solution[:,b_index * dimension + 2])
        lines.append(line)

    def update(num, solution, lines):
        for b_index in range(n_body):
            lines[b_index].set_data(solution[:num,b_index * dimension],
                                    solution[:num,b_index * dimension + 1])
            lines[b_index].set_3d_properties(solution[:num,b_index * dimension + 2])
        return lines

    ani = animation.FuncAnimation(fig, update, fargs=[solution, lines],
                              interval=25, blit=True, frames=N_T)

    plt.show()


def test_ode_3b_3d():
    '''
        Tests ode defined in ode.py with two bodies in two dimensions
    '''
    dimension = 3
    n_body = 3
    body_position = np.array([np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 3, 3])])
    body_velocity = np.array([np.array([1, 1, 0.1]), np.array([-1, -1, 0]), np.array([-1, 1, 0])])
    body_masses = (4, 4, 4)

    N_T = 800
    solution = np.zeros((N_T, 2 * n_body * dimension))
    solution[0,0:n_body*dimension] = body_position.flatten()
    solution[0,n_body*dimension:] = body_velocity.flatten()

    ode_auto = lambda x: ode(x, body_masses, n_body, dimension)

    for i in range(N_T - 1):
        solution[i+1,:] = rk4step(ode_auto, 0.08, solution[i,:])

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    lines = []
    dots = []
    objects = []

    for b_index in range(n_body):
        line, = ax.plot3D(solution[:,b_index * dimension],
                          solution[:,b_index * dimension + 1],
                          solution[:,b_index * dimension + 2])
        dot, = ax.plot3D(solution[0,b_index * dimension],
                          solution[0,b_index * dimension + 1],
                          solution[0,b_index * dimension + 2], 'go')
        lines.append(line)
        dots.append(dot)

        objects.append(line)
        objects.append(dot)

    def update(num, solution, objects):
        for b_index in range(0, n_body):
            objects[2 * b_index].set_data(solution[:num,b_index * dimension],
                                    solution[:num,b_index * dimension + 1])
            objects[2 * b_index].set_3d_properties(solution[:num,b_index * dimension + 2])

            objects[2 * b_index + 1].set_data(solution[num,b_index * dimension],
                              solution[num,b_index * dimension + 1])
            objects[2 * b_index + 1].set_3d_properties(solution[num,b_index * dimension + 2])
        return objects

    ani = animation.FuncAnimation(fig, update, fargs=[solution, objects],
                              interval=25, blit=True, frames=N_T)

    plt.show()


def test_ode_sun_earth_3d():
    '''
        Tests ode defined in ode.py with two bodies in two dimensions
    '''
    dimension = 3
    n_body = 2
    body_position = np.array([np.array([0, 0, 0]), np.array([200, 0, 0])])
    body_velocity = np.array([np.array([0, 0, 0]), np.array([0, 10, -1])])
    body_masses = (30000, 1)

    N_T = 800
    solution = np.zeros((N_T, 2 * n_body * dimension))
    solution[0,0:n_body*dimension] = body_position.flatten()
    solution[0,n_body*dimension:] = body_velocity.flatten()

    ode_auto = lambda x: ode(x, body_masses, n_body, dimension)

    for i in range(N_T - 1):
        solution[i+1,:] = rk4step(ode_auto, 0.08, solution[i,:])

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    lines = []
    dots = []
    objects = []

    for b_index in range(n_body):
        line, = ax.plot3D(solution[:,b_index * dimension],
                          solution[:,b_index * dimension + 1],
                          solution[:,b_index * dimension + 2])
        dot, = ax.plot3D(solution[0,b_index * dimension],
                          solution[0,b_index * dimension + 1],
                          solution[0,b_index * dimension + 2], 'go')
        lines.append(line)
        dots.append(dot)

        objects.append(line)
        objects.append(dot)

    def update(num, solution, objects):
        for b_index in range(0, n_body):
            objects[2 * b_index].set_data(solution[:num,b_index * dimension],
                                    solution[:num,b_index * dimension + 1])
            objects[2 * b_index].set_3d_properties(solution[:num,b_index * dimension + 2])

            objects[2 * b_index + 1].set_data(solution[num, b_index * dimension],
                              solution[num, b_index * dimension + 1])
            objects[2 * b_index + 1].set_3d_properties(solution[num, b_index * dimension + 2])
        return objects

    ani = animation.FuncAnimation(fig, update, fargs=[solution, objects],
                              interval=25, blit=True, frames=N_T)

    plt.show()


def test_ode_2b_2d_controllable():
    '''
        Tests ode defined in ode.py with two bodies in two dimensions
    '''
    dimension = 2
    n_body = 2
    body_position = np.array([np.array([1, 0]), np.array([0, 1])])
    body_velocity = np.array([np.array([1, 1]), np.array([-1, -1])])
    constant_controls = (0.01, 0)
    thrust_max = 0.3
    body_masses = (3, 4)

    N_T = 800
    solution = np.zeros((N_T, 2 * n_body * dimension))
    solution[0,0:n_body*dimension] = body_position.flatten()
    solution[0,n_body*dimension:] = body_velocity.flatten()

    x = ca.SX.sym("x", (2 * dimension * n_body, 1))

    ODE = ca.Function("ODE", [x], [ode_controllable_casadi(x, constant_controls,
                                                           body_masses, n_body,
                                                           dimension, thrust_max)])

    for i in range(N_T - 1):
        solution[i+1,:] = rk4step(ODE, 0.08, solution[i,:]).full().flatten()

    fig, ax = plt.subplots()

    ax.arrow(0, 0, constant_controls[0] * cos(constant_controls[1]),
             constant_controls[0] * sin(constant_controls[1]))

    lines = []

    for b_index in range(n_body):
        line, = ax.plot(solution[:,b_index * dimension],solution[:,b_index * dimension + 1])
        lines.append(line)

    def update(num, solution, lines):
        for b_index in range(n_body):
            lines[b_index].set_data(solution[:num,b_index * dimension],
                                    solution[:num,b_index * dimension + 1])
        return lines

    ani = animation.FuncAnimation(fig, update, fargs=[solution, lines],
                              interval=25, blit=True, frames=N_T)

    plt.show()


def test_ode_2b_2d_controllable_2():
    '''
        Tests ode defined in ode.py with two bodies in two dimensions
    '''
    dimension = 2
    n_body = 2
    body_position = np.array([np.array([50, 0]), np.array([0, 0])])
    body_velocity = np.array([np.array([1, -3]), np.array([0, 0])])
    constant_controls = (0.01, 0)
    thrust_max = 0.3
    body_masses = (1, 800)

    N_T = 100
    solution = np.zeros((N_T, 2 * n_body * dimension))
    solution[0,0:n_body*dimension] = body_position.flatten()
    solution[0,n_body*dimension:] = body_velocity.flatten()

    x = ca.SX.sym("x", (2 * dimension * n_body, 1))

    ODE = ca.Function("ODE", [x], [ode_controllable_casadi(x, constant_controls,
                                                           body_masses, n_body,
                                                           dimension, thrust_max)])

    for i in range(N_T - 1):
        solution[i+1,:] = rk4step(ODE, 0.04, solution[i,:]).full().flatten()
        for j in range(9):
            solution[i+1,:] = rk4step(ODE, 0.04, solution[i+1,:]).full().flatten()

    fig, ax = plt.subplots()

    ax.arrow(0, 0, constant_controls[0] * cos(constant_controls[1]),
             constant_controls[0] * sin(constant_controls[1]))

    lines = []

    for b_index in range(n_body):
        line, = ax.plot(solution[:,b_index * dimension],solution[:,b_index * dimension + 1])
        lines.append(line)

    def update(num, solution, lines):
        for b_index in range(n_body):
            lines[b_index].set_data(solution[:num,b_index * dimension],
                                    solution[:num,b_index * dimension + 1])
        return lines

    ani = animation.FuncAnimation(fig, update, fargs=[solution, lines],
                              interval=25, blit=True, frames=N_T)

    plt.show()


def test_ode_2b_2d_sequential():
    dimension = 2
    n_body = 2
    N_T = 100
    body_position = np.array([np.array([8, 0]), np.array([0, 0])])
    # 0, -2
    body_velocity = np.array([np.array([0.3, -2.5]), np.array([0, 0])])
    thrust_max = 1.2
    body_masses = (1, 400)

    solution = np.zeros((N_T, 2 * n_body * dimension))
    solution[0,0:n_body*dimension] = body_position.flatten()
    solution[0,n_body*dimension:] = body_velocity.flatten()

    solution_0 = np.zeros((N_T, 2 * n_body * dimension))
    solution_0[0,0:n_body*dimension] = body_position.flatten()
    solution_0[0,n_body*dimension:] = body_velocity.flatten()

    x = ca.SX.sym("x", (2 * dimension * n_body, 1))
    u = ca.SX.sym("cc", (2, 1))

    controls = ca.SX.sym("cc", (N_T * 2, 1))

    ODE = ca.Function("ODE", [x, u], [ode_controllable_casadi(x, u,
                                                           body_masses, n_body,
                                                           dimension, thrust_max)])

    print(solution[0,:])
    def Phi(controls):
        cost = 0
        x = solution[0,:]
        for i in range(N_T - 1):
            x = rk4step_u(ODE, 0.3, x, controls[2 * i: 2 * i + 2])
            cost += 0 #controls[2 * i] / 100
            d = ca.norm_2(x[0: dimension] - x[dimension: 2 * dimension])
            #4
            # 1 / (2x) * exp(x^2 / 2)
            cost += (i ** 2 / N_T) * (ca.exp(d / 3) / (d / 3)) / 30
        return cost

    PHI = ca.Function("Phi", [controls], [Phi(controls)])
    JPhi_expr = ca.jacobian(Phi(controls), controls)
    JPHI = ca.Function("JPHI", [controls], [JPhi_expr])

    #import pdb; pdb.set_trace()

    constraints = ca.vertcat(controls[::2], controls[1::2])
    lbg = [-thrust_max] * N_T + [-pi] * N_T
    ubg = [thrust_max] * N_T + [pi] * N_T

    nlp = {'x': controls, 'f': Phi(controls), 'g': constraints}

    solver = ca.nlpsol('solver', 'ipopt', nlp)

    # Solve the NLP
    res = solver(
        x0 = [10, pi] * 5 + [0] * (2 * N_T - 10),    # solution guess
        lbx = -ca.inf,          # lower bound on x
        ubx = ca.inf,           # upper bound on x
        lbg = lbg,                # lower bound on g
        ubg = ubg,                # upper bound on g
    )

    u_opt = res["x"].full()

    print(u_opt)
    print(solution_0[0,:])

    N_T_sim = 2 * N_T

    for i in range(N_T - 1):
        solution[i+1,:] = rk4step_u(ODE, 0.15, solution[i,:],
                                    u_opt[2 * i : 2 * i + 2]).full().flatten()

        solution[i+1,:] = rk4step_u(ODE, 0.15, solution[i+1,:],
                                    u_opt[2 * i : 2 * i + 2]).full().flatten()



        solution_0[i+1,:] = rk4step_u(ODE, 0.3, solution_0[i,:],
                                   [0, 0]).full().flatten()

    fig, ax = plt.subplots()

    ax.plot(solution_0[:,0],solution_0[:,1], label="sol_0")

    lines = []

    for b_index in range(n_body):
        line, = ax.plot(solution[:,b_index * dimension],solution[:,b_index * dimension + 1])
        lines.append(line)
        # line, = ax.plot(solution[:,b_index * dimension],solution_0[:,b_index * dimension + 1])
        # lines.append(line)

    def update(num, solution, lines):
        for b_index in range(n_body):
            lines[b_index].set_data(solution[:num,b_index * dimension],
                                    solution[:num,b_index * dimension + 1])
            # lines[b_index+1].set_data(solution_0[:num,b_index * dimension],
            #                         solution_0[:num,b_index * dimension + 1])
        return lines

    ani = animation.FuncAnimation(fig, update, fargs=[solution, lines],
                              interval=75, blit=True, frames=N_T)

    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    # test_ode_2b_2d()
    # test_ode_3b_2d()
    # test_ode_2b_3d()
    # test_ode_3b_3d()
    # test_ode_sun_earth_3d()

    # test_ode_2b_2d_controllable()
    #
    # test_ode_2b_2d_controllable_2()
    test_ode_2b_2d_sequential()
