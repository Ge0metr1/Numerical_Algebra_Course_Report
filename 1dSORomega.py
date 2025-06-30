import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import rcParams

# Set font for better visualization
rcParams['font.family'] = 'Arial'

def sor_poisson_solver(f, a, b, N, omega=1.8, max_iter=10000, tol=1e-6):
    """SOR method for 1D Poisson equation -u'' = f(x), u(0)=a, u(1)=b"""
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
    return len(residuals), residuals[-1], elapsed_time

# Test problem
def f(x):
    return np.sin(np.pi * x)

# Parameters
a, b = 0, 0
N = 100
omega_values = np.linspace(1.0, 2.2, 15)  # Test ω from 1.0 to 1.95

# Storage for results
results = {
    'iterations': [],
    'residuals': [],
    'times': []
}

# Run tests
print("=== SOR Parameter Study ===")
print(f"Testing {len(omega_values)} ω values from {omega_values[0]:.2f} to {omega_values[-1]:.2f}")
print("-"*50)

for omega in omega_values:
    iters, resid, time_used = sor_poisson_solver(f, a, b, N, omega=omega)
    results['iterations'].append(iters)
    results['residuals'].append(resid)
    results['times'].append(time_used)
    print(f"ω = {omega:.2f} | Iters: {iters:4d} | Residual: {resid:.2e} | Time: {time_used:.4f} sec")

# Find optimal omega
optimal_idx = np.argmin(results['iterations'])
optimal_omega = omega_values[optimal_idx]



# Iterations vs omega
plt.figure(figsize=(8, 6))
plt.plot(omega_values, results['iterations'], 'bo-')
plt.plot(optimal_omega, results['iterations'][optimal_idx], 'ro', markersize=8)
plt.xlabel('Relaxation parameter ω')
plt.ylabel('Iteration count')
plt.title('Convergence Speed vs ω')
plt.grid(True)

# Time vs omega
plt.figure(figsize=(8, 6))
plt.plot(omega_values, results['times'], 'go-')
plt.plot(optimal_omega, results['times'][optimal_idx], 'ro', markersize=8)
plt.xlabel('Relaxation parameter ω')
plt.ylabel('Computation time (sec)')
plt.title('Computational Cost vs ω')
plt.grid(True)

# Residual vs omega
plt.figure(figsize=(8, 6))
plt.semilogy(omega_values, results['residuals'], 'mo-')
plt.xlabel('Relaxation parameter ω')
plt.ylabel('Final residual (log scale)')
plt.title('Solution Accuracy vs ω')
plt.grid(True)

plt.tight_layout()
plt.suptitle(f'SOR Performance Analysis (N={N})', y=1.05)
plt.show()

# Theoretical optimum
h = 1.0 / (N + 1)
theoretical_opt = 2 / (1 + np.sin(np.pi * h))
print("\n=== Results Summary ===")
print(f"Experimental optimal ω: {optimal_omega:.4f}")
print(f"Theoretical optimal ω: {theoretical_opt:.4f}")
print(f"Minimum iterations: {results['iterations'][optimal_idx]} at ω = {optimal_omega:.3f}")