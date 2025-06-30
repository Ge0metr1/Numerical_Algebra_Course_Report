import numpy as np
import matplotlib.pyplot as plt
import time

def solve_poisson_1d(f, a, b, N, method='jacobi', omega=1.0, max_iter=10000, tol=1e-6):
    """
    求解一维泊松方程 -u'' = f(x), u(0)=a, u(1)=b
    
    参数:
        f : 函数，源项 f(x)
        a, b : 边界条件值
        N : 网格点数（内部点）
        method : 迭代方法 ('jacobi', 'gs', 'sor')
        omega : SOR的松弛因子
        max_iter : 最大迭代次数
        tol : 收敛容差
        
    返回:
        x : 网格点
        u : 数值解（包含边界点）
        residuals : 残差历史
        elapsed_time : 计算时间（秒）
    """
    h = 1.0 / (N + 1)
    x = np.linspace(0, 1, N+2)
    u = np.zeros(N+2)
    u[0], u[-1] = a, b
    residuals = []
    
    rhs = h**2 * f(x[1:-1])
    rhs[0] += a
    rhs[-1] += b
    
    start_time = time.time()
    
    for k in range(max_iter):
        u_old = u.copy()
        residual = 0.0
        
        for i in range(1, N+1):
            if method == 'jacobi':
                u[i] = 0.5 * (u_old[i-1] + u_old[i+1] + rhs[i-1])
            elif method == 'gs':
                u[i] = 0.5 * (u[i-1] + u_old[i+1] + rhs[i-1])
            elif method == 'sor':
                gs_update = 0.5 * (u[i-1] + u_old[i+1] + rhs[i-1])
                u[i] = u_old[i] + omega * (gs_update - u_old[i])
            
            residual = max(residual, abs(u[i] - u_old[i]))
        
        residuals.append(residual)
        if residual < tol:
            break
    
    elapsed_time = time.time() - start_time
    return x, u, residuals, elapsed_time

# 测试案例
def f(x):
    return np.sin(np.pi * x)

# 参数设置
a, b = 0, 0
N = 100
methods = ['jacobi', 'gs', 'sor']
omega = 1.8  # SOR最优松弛因子

# 结果表格
print(f"{'Method':<10} | {'Iterations':>10} | {'Final Residual':>15} | {'CPU Time (s)':>12}")
print("-" * 55)

# 求解并记录结果
results = {}
for method in methods:
    x, u, res, time_used = solve_poisson_1d(f, a, b, N, method=method, omega=omega)
    results[method] = {
        'iterations': len(res),
        'residual': res[-1],
        'time': time_used
    }
    print(f"{method.upper():<10} | {len(res):>10} | {res[-1]:>15.2e} | {time_used:>12.4f}")

# 可视化
plt.figure(figsize=(8, 6))
for method in methods:
    x, u, res, _ = solve_poisson_1d(f, a, b, N, method=method, omega=omega)
    plt.plot(x, u, label=f'{method.upper()} ({len(res)} iters)', alpha=0.8)

# 真解
true_sol = np.sin(np.pi * x) / (np.pi**2)
plt.plot(x, true_sol, 'k--', label='True Solution')

plt.xlabel('x')
plt.ylabel('u(x)')
plt.title(f'Solution Comparison (N={N})')
plt.legend()
plt.grid(True)
plt.show()
