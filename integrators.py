import numpy as np
import matplotlib.pyplot as plt

def integrate_rect(fn, a, b, n_steps, display=False):
    _sum = 0

    step_size = (b-a)/n_steps
    x_range = np.arange(a, b, step_size)

    for x in x_range:
       _sum += fn(x) * step_size

    if display:
        figure = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x_range, fn(x_range))
        ax.set_title("Rectangular integration of {} with {} integration steps".format(fn, n_steps))

    return _sum / n_steps

def integrate_trap(fn, a, b, n_steps, display=False):
    _sum = 0

    step_size = (b-a)/n_steps
    x_range = np.arange(a + step_size, b, step_size)

    _sum += (fn(a) + fn(b)) * step_size/ 2

    for x in x_range:
       _sum += fn(x) * step_size

    if display:
        figure = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x_range, fn(x_range))
        ax.set_title("Trapezoidal integration of {} with {} integration steps".format(fn, n_steps))
    return _sum / n_steps

def integrate_simpsons(fn, a, b, n_steps, display=False):
    _sum = 0

    step_size = (b-a)/n_steps
    i_range_1 = np.arange(1, n_steps / 2)
    i_range_2 = np.arange(1, n_steps / 2 - 1)

    _sum += (fn(a) + fn(b))

    for i in i_range_1:
       _sum += 4 * fn(a + (2 * i - 1) * step_size)

    for i in i_range_2:
        _sum += 2 * fn(a + (2 * i) * step_size)

    if display:
        figure = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x_range, fn(x_range))
        ax.set_title("Simpson integration of {} with {} integration steps".format(fn, n_steps))

    return _sum  * step_size / 3

def error_scaling(integrate=integrate_rect, fn=np.exp, a= -1., b=1., log_10s=[1, 2, 3, 4], display=False):
    comparisons = [integrate(fn, a, b, 10 ** i) for i in log_10s]

    if display:
        plt.figure()
        plt.plot(log_10s, comparisons)

    return comparisons
