import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import rcParams

# Set font for better visualization
rcParams['font.family'] = 'Arial'  # Use a standard font

def sor_poisson_solver(f, a, b, N, omega=1.8, max_iter=10000, tol=1e-6):
    """
    SOR method for 1D Poisson equation -u'' = f(x), u(0)=a, u(1)=b
    """
    h = 1.0 / (N + 1)
    x = np.linspace(0, 1, N+2)
    u = np.zeros(N+2)
    u[0], u[-1] = a, b
    
    rhs = h**2 * f(x[1:-1])
    rhs[0] += a
    rhs[-1] += b
    
    residuals = []
    start_time = time.time()
    
    for k in range(max_iter):
        max_diff = 0.0
        for i in range(1, N+1):
            gs_update = 0.5 * (u[i-1] + u[i+1] + rhs[i-1])
            new_val = u[i] + omega * (gs_update - u[i])
            max_diff = max(max_diff, abs(new_val - u[i]))
            u[i] = new_val
        
        residuals.append(max_diff)
        if max_diff < tol:
            break
    
    elapsed_time = time.time() - start_time
    return x, u, residuals, elapsed_time

# Test problem
def f(x):
    return np.sin(np.pi * x)  # Exact solution: u(x) = sin(πx)/π²

# Parameters
a, b = 0, 0    # Boundary conditions
N = 120         # Number of internal grid points
omega = 1.8     # Optimal relaxation factor

# Solve
x, u, residuals, time_used = sor_poisson_solver(f, a, b, N, omega=omega)

# Performance report
print("=== SOR Method Performance Report ===")
print(f"Grid points: {N}")
print(f"Relaxation factor ω: {omega}")
print(f"Iterations: {len(residuals)}")
print(f"Final residual: {residuals[-1]:.3e}")
print(f"Computation time: {time_used:.4f} sec")


# Numerical vs exact solution
plt.figure(figsize=(8, 6))
true_solution = np.sin(np.pi * x) / (np.pi**2)
plt.plot(x, u, 'b-', label='SOR solution', linewidth=2)
plt.plot(x, true_solution, 'r--', label='Exact solution', linewidth=2)
plt.xlabel('x coordinate')
plt.ylabel('u(x)')
plt.title('Numerical vs Exact Solution')
plt.legend()
plt.grid(True)

# Residual convergence
plt.figure(figsize=(8, 6))
plt.semilogy(residuals)
plt.xlabel('Iteration count')
plt.ylabel('Residual (log scale)')
plt.title('SOR Convergence History')
plt.grid(True, which='both')

plt.tight_layout()
plt.show()

# Convergence analysis
if len(residuals) > 10:
    conv_rate = (residuals[-1]/residuals[0])**(1/len(residuals))
    print(f"\nEstimated convergence rate: {conv_rate:.4f}")
    print(f"Theoretical spectral radius: {omega-1:.4f} (ω={omega})")