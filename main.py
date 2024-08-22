# Setup
import numpy as np

# import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
from scipy.special import ellipkinc, ellipeinc, ellipk

# numerical values

L = 1.  # m
g = 9.81  # m/s**2


def f(X, t):
    """
    The derivative function
    """
    x, y = X
    return np.array([y, -g / L * np.sin(x)])


def Euler(func, X0, t):
    """
    Euler integrator.
    """
    dt = t[1] - t[0]
    nt = len(t)
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt - 1):
        X[i + 1] = X[i] + func(X[i], t[i]) * dt
    # print(X)

    return X


# define vector and time
def choice():
    global X_euler, X_rk, X_rk_2, X_odeint
    angles = int(input("Enter 1 for 90 degrees, 2 for 1.5 degrees, 3 for 156 degrees: "))
    if angles == 1:
        X_euler = np.radians([90, 0])
        X_rk = np.radians([90, 0])
        X_rk_2 = np.radians([90, 0])
        X_odeint = np.radians([90, 0])
        X_ellip = np.radians([90, 0])
    elif angles == 2:

        X_euler = np.radians([1.5, 0])
        X_rk = np.radians([1.5, 0])
        X_rk_2 = np.radians([1.5, 0])
        X_odeint = np.radians([1.5, 0])
        X_ellip = np.radians([1.5, 0])

    elif angles == 3:
        X_euler = np.radians([156, 0])
        X_rk = np.radians([156, 0])
        X_rk_2 = np.radians([156, 0])
        X_odeint = np.radians([156, 0])
        X_ellip = np.radians([156, 0])
    else:
        print("Invalid choice")
        exit()

    return X_euler, X_rk, X_rk_2, X_odeint, X_ellip


time = 40
# t = np.linspace(0, time, 200)


t = np.linspace(0, time, 1000)
# t = np.linspace(0, time, 1000)
# t = np.linspace(0, time, 500)

def get_coords(theta):
    return L * np.sin(theta), -L * np.cos(theta)


# solve
def animation(def_x, theta0, v=0, t=t):
    fig, ax = plt.subplots()

    # pendulum rod, initial position
    x0, y0 = get_coords(theta0[0])
    line, = ax.plot([0, x0], [0, y0], lw=3, c='k')
    bob_radius = 0.08
    circle = ax.add_patch(plt.Circle((x0, y0), bob_radius, fc='r', zorder=3))

    ax.set_xlim(-1.1 * L, 1.1 * L)
    ax.set_ylim(-1.1 * L, 1.1 * L)

    # def init():
    #     line.set_data([], [])
    #     return line,

    def animate(i):
        # x = np.sin(def_x[i, 0])
        # y = -np.cos(def_x[i, 0])
        x, y = get_coords(def_x[i, 0])
        line.set_data([0, x], [0, y])
        circle.set_center((x, y))
        return line, circle

    # anim = FuncAnimation(fig, animate, init_func=init, frames=len(t), interval=50, blit=True)
    anim = FuncAnimation(fig, animate, frames=len(t), interval=20, blit=True)
    # anim = FuncAnimation(fig, animate, frames=len(t), interval=50, blit=True)
    plt.show()


def RK4(func, X0, t):
    """
    Runge and Kutta 4 integrator.
    """
    dt = t[1] - t[0]
    nt = len(t)
    X = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt - 1):
        k1 = func(X[i], t[i])
        k2 = func(X[i] + dt / 2. * k1, t[i] + dt / 2.)
        k3 = func(X[i] + dt / 2. * k2, t[i] + dt / 2.)
        k4 = func(X[i] + dt * k3, t[i] + dt)
        X[i + 1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    return X


def RK2(func, X0, t):
    dt = t[1] - t[0]
    nt = len(t)
    X = np.zeros([nt, len(X0)])
    X[0] = X0

    for i in range(nt - 1):
        k1 = func(X[i], t[i])
        k2 = func(X[i] + dt / 2. * k1, t[i] + dt / 2.)
        X[i + 1] = X[i] + dt / 6. * (k1 + 2 * k2)
    return X


def animate_euler(X_euler):
    X = Euler(f, X_euler, t)
    animation(X, X_euler, 0, t)


def animate_rk(X_rk):
    X = RK4(f, X_rk, t)
    animation(X, X_rk, 0, t)


def animate_rk_2(X_rk):
    X = RK2(f, X_rk, t)
    animation(X, X_rk, 0, t)


# def animate_odeint(X_odeint):
#     X = odeint(f, X_odeint, t)
#     animation(X, X_odeint, 0, t)


# def plot_odeint(X_odeint):
#     sol = odeint(f, X_odeint, t)
#     plt.plot(t, sol[:, 0], 'b', label='theta(t)')
#     plt.plot(t, sol[:, 1], 'g', label='omega(t)')
#     plt.legend(loc='best')
#     plt.xlabel('t')
#     plt.grid()
#     plt.show()

# solve elliptic integrals using given initial conditions and time to plot pendulum motion
def pendulum_period(g, l, theta0):
    """
    Computes the period of the pendulum.
    g: gravitational acceleration.
    l: length of the pendulum.
    theta0: initial angle of the pendulum.
    """
    k = np.sin(theta0 / 2)
    Kk = ellipk(k ** 2)
    T = 4 * np.sqrt(l / g) * Kk

    return T


# Compute the period of the pendulum
theta0 = np.pi / 2  # initial angle of the pendulum (rad)
omega0 = 0
T = pendulum_period(g, L, theta0)


def elliptic(X_elliptic, t):
    dt = t[1] - t[0]
    theta = X_elliptic[0]
    omega = X_elliptic[1]
    theta = np.array([theta])
    omega = np.array([omega])
    for i in range(len(t) - 1):
        theta = np.append(theta, theta[i] + omega[i] * dt)
        omega = np.append(omega, omega[i] - g / L * np.sin(theta[i]) * dt)
    return np.array([theta, omega]).T


# plot pendulum motion from given elliptic integrals solution
def plot_elliptic(X_elliptic):
    sol = elliptic(X_elliptic, t)
    sol2 = odeint(f, X_rk, t)
    plt.plot(t, sol[:, 0], 'b', label='theta(t)')
    plt.plot(t, sol[:, 1], 'g', label='omega(t)')
    plt.plot(t, sol2[:, 0], 'r.', markersize=1, label='theta(t) perfect')
    plt.plot(t, sol2[:, 1], 'y.', markersize=1, label='omega(t) perfect')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()


# animate pendulum motion from given elliptic integrals solution
def animate_elliptic(X_elliptic):
    X = elliptic(X_elliptic, t)
    animation(X, X_elliptic, 0, t)


# find period using elliptic integrals built in scipy
def elliptic_period(g, l, theta0):
    """
    Computes the period of the pendulum.
    g: gravitational acceleration.
    l: length of the pendulum.
    theta0: initial angle of the pendulum.
    """
    k = np.sin(theta0 / 2)
    Kk = ellipk(k ** 2)
    T = 4 * np.sqrt(l / g) * Kk

    return T


# compare period of pendulum with euler, rk4, and elliptic integrals
def rk4_period(X):
    T_rk4 = 2 * np.pi * np.average(np.diff(X[:, 0][np.where(X[:, 0] > 0)]))
    return T_rk4


def euler_period(X):
    T_euler = 2 * np.pi * np.average(np.diff(X[:, 0][X[:, 0] > 0]))
    return T_euler

def get_rk_period(X_rk):
    X = RK4(f, X_rk, t)
    T_rk4 = rk4_period(X)
    return T_rk4

def get_euler_period(X_euler):
    X = Euler(f, X_euler, t)
    T_euler = euler_period(X)
    return T_euler
def compare_period(X_euler, X_rk, X_elliptic):
    T_elliptic = elliptic_period(g, L, np.radians(X_elliptic))
    T_euler = get_euler_period(X_euler)
    T_rk = get_rk_period(X_rk)
    print("Euler period: ", T_euler)
    print("RK4 period: ", T_rk)
    print("Elliptic period: ", T_elliptic)


def plot_rk(X_rk):
    sol = RK4(f, X_rk, t)
    # sol2 = odeint(f, X_rk, t)
    plt.plot(t, sol[:, 0], 'b', label='theta(t) 4th order')
    plt.plot(t, sol[:, 1], 'g', label='omega(t) 4th order')
    # # plot comparison using dotted lines
    # plt.plot(t, sol2[:, 0], 'r.', markersize=1, label='theta(t) perfect')
    # plt.plot(t, sol2[:, 1], 'y.', markersize=1, label='omega(t) perfect')
    # plot comparison with ideal pendulum plot for the same initial conditions and time range
    sol2 = odeint(f, X_rk, t)
    plt.plot(t, sol2[:, 0], 'r.', markersize=1, label='theta(t) perfect')
    plt.plot(t, sol2[:, 1], 'y.', markersize=1, label='omega(t) perfect')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()


def plot_rk2(X_rk_2):
    sol = RK2(f, X_rk_2, t)
    sol2 = odeint(f, X_rk_2, t)
    plt.plot(t, sol[:, 0], 'b', label='theta(t) 2nd order')
    plt.plot(t, sol[:, 1], 'g', label='omega(t) 2nd order')
    # plot comparison using dotted lines
    plt.plot(t, sol2[:, 0], 'r.', markersize=1, label='theta(t) perfect')
    plt.plot(t, sol2[:, 1], 'y.', markersize=1, label='omega(t) perfect')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()


def plot_euler(X_euler):
    sol = Euler(f, X_euler, t)
    sol2 = odeint(f, X_euler, t)
    plt.plot(t, sol[:, 0], 'b', label='theta(t) Euler')
    plt.plot(t, sol[:, 1], 'g', label='omega(t) Euler')
    # compare with odeint
    plt.plot(t, sol2[:, 0], 'r.', markersize=1, label='theta(t) perfect')
    plt.plot(t, sol2[:, 1], 'y.', markersize=1, label='omega(t) perfect')

    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()


def animate_all(X_euler, X_rk_2, X_rk, X_ode):
    # animate all
    X_euler = Euler(f, X_euler, t)
    X_rk_2 = RK2(f, X_rk_2, t)
    X_rk = RK4(f, X_rk, t)
    X_ode = odeint(f, X_ode, t)
    animation(X_euler, X_rk_2, X_rk, X_ode, t)


def plot_all(X_euler, X_rk_2, X_rk, X_ode):
    X_euler = Euler(f, X_euler, t)
    X_rk_2 = RK2(f, X_rk_2, t)
    X_rk = RK4(f, X_rk, t)
    X_ode = odeint(f, X_ode, t)
    plt.plot(t, X_euler[:, 0], 'b', label='theta(t) Euler')
    plt.plot(t, X_euler[:, 1], 'g', label='omega(t) Euler')
    plt.plot(t, X_rk_2[:, 0], 'r', label='theta(t) 2nd order')
    plt.plot(t, X_rk_2[:, 1], 'y', label='omega(t) 2nd order')
    plt.plot(t, X_rk[:, 0], 'c', label='theta(t) 4th order')
    plt.plot(t, X_rk[:, 1], 'm', label='omega(t) 4th order')
    plt.plot(t, X_ode[:, 0], 'k', label='theta(t) odeint')
    plt.plot(t, X_ode[:, 1], 'k', label='omega(t) odeint')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # choose from terminal
    # 1 - Euler
    # 2 - Runge Kutta 4
    # 3 - Runge Kutta 2
    # 4 - odeint
    euler, rk2, rk, ode, ellip = choice()
    T_ellip = ellip[0]
    compare_period(euler, rk2, T_ellip)
    animate_euler(euler)
    plot_euler(euler)
    """Runge kutta"""
    animate_rk_2(rk2)
    plot_rk2(rk2)
    animate_rk(rk)
    plot_rk(rk)
    # animate all of the above in one single window
    # plot all of the above in one single window
