import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """Objective function: f(x) = f(x) = x4 − 3x3 + 2"""
    return (x ** 4) - 3 * (x**3) + 2


def f_prime(x):
    """Derivative of f(x) f′(x) = 4x3 − 9x2"""
    return 4 * (x **3) - 9* (x**2)


def gradient_descent(f_prime, x_init, learning_rate=0.01, epochs=100):
    """
    Perform gradient descent to minimize the function.

    Arguments:
    f_prime -- function: Derivative of the objective function
    x_init -- float: Initial value of x
    learning_rate -- float: Learning rate for gradient descent
    epochs -- int: Number of iterations

    Returns:
    x -- float: Final optimized value of x
    history -- list: List of x values during optimization
    """
    x = x_init
    history = [x]

    for _ in range(epochs):
        x = x - learning_rate * f_prime(x)  # Gradient update step
        history.append(x)

    return x, history


# Initialize parameters
x_init = 0.5
learning_rate = 0.01
epochs = 100

# Run gradient descent
x_optimal, history = gradient_descent(f_prime, x_init, learning_rate, epochs)

# Plot the function and optimization path
x_vals = np.linspace(-1, 3, 100)
y_vals = f(x_vals)

plt.plot(x_vals, y_vals, label="f(x)")
plt.scatter(history, [f(x) for x in history], color="red", s=10, label="Iterations")
plt.scatter(x_optimal, f(x_optimal), color="green", marker="x", s=100, label=f"Min at x={x_optimal:.3f}")

plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid()
plt.show()
