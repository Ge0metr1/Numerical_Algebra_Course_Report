import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import rcParams

# Set font for better visualization
rcParams['font.family'] = 'Arial'

def sor_poisson(f, a, b, N, omega=1.8, max_iter=10000, tol=1e-6):
    """SOR solver for 1D Poisson equation -u'' = f(x), u(0)=a, u(1)=b"""
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
    
    # Calculate exact solution and error
    exact = np.sin(np.pi * x) / (np.pi**2) + (b - a) * x + a
    error = np.abs(u - exact)
    max_error = np.max(error)
    
    return x, u, len(residuals), residuals[-1], time.time() - start_time, max_error

# Problem definition
def f(x):
    return np.sin(np.pi * x)  # Source term

a, b = 0, 0  # Boundary conditions
omega = 1.8   # Relaxation parameter

# Test different grid sizes
N_values = [60, 70, 80, 90, 100, 110, 120, 150]
results = []

print("SOR Performance vs Grid Size (ω = 1.8)")
print("-"*65)
print(f"{'N':<8} | {'h':<8} | {'Iterations':<12} | {'Residual':<12} | {'Max Error':<12} | {'Time (s)':<10}")
print("-"*65)

for N in N_values:
    x, u, iters, resid, time_used, max_err = sor_poisson(f, a, b, N, omega)
    h = 1.0/(N+1)
    results.append((N, h, x, u, iters, resid, time_used, max_err))
    print(f"{N:<8} | {h:<8.5f} | {iters:<12} | {resid:<12.2e} | {max_err:<12.2e} | {time_used:<10.4f}")

# Create figure with 3 subplots
plt.figure(figsize=(15, 4))

# 1. Iterations vs N
plt.subplot(1, 3, 1)
plt.plot([r[0] for r in results], [r[4] for r in results], 'bo-')
plt.xlabel('Number of grid points (N)')
plt.ylabel('Iterations')
plt.title('(a) Convergence vs Grid Resolution')
plt.grid(True)

# 2. Time vs N
plt.subplot(1, 3, 2)
plt.plot([r[0] for r in results], [r[6] for r in results], 'ro-')
plt.xlabel('Number of grid points (N)')
plt.ylabel('CPU Time (s)')
plt.title('(b) Computation Time vs Grid Resolution')
plt.grid(True)

# 3. Error vs h (log-log plot)
plt.subplot(1, 3, 3)
n_values = [r[0] for r in results]
errors = [r[5] for r in results]
plt.loglog(n_values, errors, 'go-', label='Numerical error')
plt.xlabel('Number of grid points (N)')
plt.ylabel('Final error')
plt.title('(c) Error Convergence (log-log)')
plt.legend()
plt.grid(True, which="both")

plt.tight_layout()
plt.show()

# Additional error visualization for selected N
plt.figure(figsize=(10, 4))
selected_N = [50, 100, 200]
for N in selected_N:
    for r in results:
        if r[0] == N:
            plt.plot(r[2], np.abs(r[3] - (np.sin(np.pi * r[2])/(np.pi**2)), 
                    label=f'N={N}, h={r[1]:.4f}'))

plt.xlabel('x coordinate')
plt.ylabel('Absolute error')
plt.title('Error Distribution Across Domain')
plt.legend()
plt.grid(True)
plt.show()