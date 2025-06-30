import numpy as np
import matplotlib.pyplot as plt


def solve_poisson_1d(f, N, method='jacobi', omega=1.0, max_iter=10000, tol=1e-6):
    """
    求解一维泊松方程 -u'' = f(x) 在 [0,1] 上，边界条件 u(0)=u(1)=0。

    参数:
        f : 函数，源项 f(x)。
        N : 整数，网格点数（实际未知数为 N-1）。
        method : 字符串，迭代方法 ('jacobi', 'gs', 'sor')。
        omega : 浮点数，SOR的松弛因子（仅对SOR有效）。
        max_iter : 整数，最大迭代次数。
        tol : 浮点数，收敛容差。

    返回:
        u : 数组，数值解。
        residuals : 数组，每次迭代的残差。
    """
    h = 1.0 / N  # 网格间距
    x = np.linspace(0, 1, N + 1)  # 网格点
    u = np.zeros(N + 1)  # 初始猜测（包含边界条件）
    residuals = []

    for k in range(max_iter):
        u_old = u.copy()
        residual = 0.0

        for i in range(1, N):
            if method == 'jacobi':
                u[i] = 0.5 * (u_old[i - 1] + u_old[i + 1] + h ** 2 * f(x[i]))
            elif method == 'gs':
                u[i] = 0.5 * (u[i - 1] + u_old[i + 1] + h ** 2 * f(x[i]))
            elif method == 'sor':
                gs_update = 0.5 * (u[i - 1] + u_old[i + 1] + h ** 2 * f(x[i]))
                u[i] = u_old[i] + omega * (gs_update - u_old[i])

            # 计算残差（以无穷范数为例）
            residual = max(residual, abs(u[i] - u_old[i]))

        residuals.append(residual)
        if residual < tol:
            break

    return u, residuals


# 测试函数
def f(x):
    return np.sin(np.pi * x)  # 真实解为 u(x) = sin(pi*x)/(pi^2)


# 参数设置
N = 50  # 网格点数
methods = ['jacobi', 'gs', 'sor']
omega = 1.8  # SOR的理论最优松弛因子（对泊松方程）

# 创建图形
plt.figure(figsize=(15, 6))

# 子图1：数值解对比
plt.subplot(1, 2, 1)
x = np.linspace(0, 1, N + 1)
true_solution = np.sin(np.pi * x) / (np.pi ** 2)
plt.plot(x, true_solution, 'k--', label='True Solution', linewidth=2)

# 子图2：残差收敛曲线
plt.subplot(1, 2, 2)

# 求解并绘图
for method in methods:
    u, residuals = solve_poisson_1d(f, N, method=method, omega=omega)

    # 绘制数值解
    plt.subplot(1, 2, 1)
    plt.plot(x, u, label=f'{method.upper()}', alpha=0.7)

    # 绘制残差
    plt.subplot(1, 2, 2)
    plt.semilogy(residuals, label=f'{method.upper()}')

# 设置子图1属性
plt.subplot(1, 2, 1)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Comparison of Numerical Solutions')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# 设置子图2属性
plt.subplot(1, 2, 2)
plt.xlabel('Iteration')
plt.ylabel('Residual (log scale)')
plt.title('Convergence Behavior')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
