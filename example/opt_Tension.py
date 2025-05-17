import numpy as np
from scipy.optimize import minimize
import time


def limit_vector_magnitude(vectors, max_length):
    """
    将二维向量数组中的每个子向量的模长限制在给定长度内。
    参数:
    vectors (np.ndarray): 输入的二维向量数组，每一行是一个子向量。
    max_length (float): 限制的最大模长。
    返回:
    np.ndarray: 限制模长后的二维向量数组。
    """
    # 计算每个子向量的模长
    magnitudes = np.linalg.norm(vectors, axis=1)
    # 创建一个缩放因子，只有在模长超过给定长度时才会改变
    scaling_factors = np.where(magnitudes > max_length, max_length / magnitudes, 1)
    # 将每个子向量缩放到给定长度
    scaled_vectors = vectors * scaling_factors[:, np.newaxis]
    return scaled_vectors


def phi_theta2_tension(D, a_vec, g, ml):
    """ D * t  = F_rope = m_l .  * (a_vec + [0,0,g])
        input:  D=[q_1, ..., q_n], a_vec, g=9.81
        output: t = [t_1, ..., t_n]^T"""
    D_pseudoInv = D.T @ np.linalg.inv(D @ D.T)
    F_rope = ml * (a_vec + np.array([0, 0, g]))
    t_vec = D_pseudoInv @ ( F_rope)  # -1
    return t_vec


def Cart2Spher_qunit(q):
    """
    input: q = [x, y, z] (1D) or a 2D array where each row is [x, y, z]
    output: theta, phi (arrays of angles / rad)
    """
    # 检查输入的维度
    q = np.array(q)  # 确保输入为NumPy数组
    # q= q / np.linalg.norm(q)
    if q.ndim == 1:
        # 处理一维输入
        x, y, z = q
        theta = np.arcsin(z)
        phi = np.arctan2(y, x)
        return np.array([theta]), np.array([phi])
    elif q.ndim == 2:
        # 处理二维输入
        x = q[:, 0]
        y = q[:, 1]
        z = q[:, 2]
        theta = np.arcsin(z)
        phi = np.arctan2(y, x)
        phi = np.mod(phi, 2 * np.pi)  # 确保 phi 在 0 到 2π 之间
        return theta, phi
    else:
        raise ValueError("Input must be a 1D or 2D array.")


def Spher2Cart_unit(theta, phi):
    """ 定义theta为系绳与水平面夹角"""
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)
    return [x, y, z]


def cal_G(theta, phi, n):
    """计算导数矩阵 G
       input: theta, phi
       output: G"""
    G = np.zeros((3, n))
    G[0, :] = -np.sin(theta) * np.cos(phi)  # 整行运算代替循环提高效率
    G[1, :] = -np.sin(theta) * np.sin(phi)
    G[2, :] = np.cos(theta)
    return G

def compute_D(theta, phi):
    """向量化计算方向矩阵D (3xn)"""
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    return np.vstack([
        cos_theta * cos_phi,
        cos_theta * sin_phi,
        sin_theta
    ])



def calGradient(theta, phi, D, F,  n):
    """计算梯度
    Args:D: 无人机方向向量矩阵 (3xn)
         F: 载荷的系绳力 (3x1)
         theta: 无人机俯仰角 (nx1)
         n:  无人机数量
    Returns: 梯度向量 (nx1)
    """
    A = D @ D.T + 1e-8 * np.eye(3)
    X = np.linalg.solve(A, F)
    S = np.linalg.solve(A, D).T  # S = D^T A^{-1}
    t = D.T @ X  # t = D^T X
    grad = np.zeros(n)
    for i in range(n):
        # 计算 \partial d_i / \partial \theta_i
        g_i = np.array([
            -np.sin(theta[i]) * np.cos(phi[i]),
            -np.sin(theta[i]) * np.sin(phi[i]),
            np.cos(theta[i])
        ]).reshape(-1, 1)  # convert 1-array to vector with reshape
        d_i = D[:, i].reshape(-1, 1)
        term1 = g_i.T @ X * t[i]
        term2 = t.reshape(1, -1) @ S @ (g_i @ d_i.T + d_i @ g_i.T) @ X
        grad[i] = 2 * (term1.item() - term2.item())
    return grad

def update_opt_phi_theta(n, a_vec, ml):
    """ input: theta, phi, a_vec, g
        output: opt_tension"""
    start_time = time.time()
    # 初始猜测
    theta0 = np.deg2rad(70) * np.ones(n)  # 角度转换为弧度
    phi0 = np.linspace(0, 2 * np.pi, n, endpoint=False)  # 均匀分布的 phi 角
    def func(x):
        theta = x[0:n]
        D = np.array([Spher2Cart_unit(theta[i], phi0[i]) for i in range(n)]).T
        t_vec = phi_theta2_tension(D, a_vec, 9.81, ml)
        return np.linalg.norm(t_vec) ** 2

    def gradient(x):
        theta = x[0:n]
        D = np.array([Spher2Cart_unit(theta[i], phi0[i]) for i in range(n)]).T
        F = ml * (a_vec + np.array([0, 0, 9.81]))
        return calGradient(theta, phi0,D, F, n)
    # 进行优化
    bounds = [(np.deg2rad(25), np.deg2rad(70)) for _ in range(n)]
    # bounds[2] = (bounds[2][0], np.deg2rad(80))
    # bounds = optimize_upper_bounds(np.array(a_vec).reshape(1, -1), phi0)
    theta_opt = minimize(func, theta0, jac=gradient,
                         bounds=bounds, method='L-BFGS-B') # L-BFGS-B比 trust-constr更快
    end_time = time.time()
    from scipy.optimize import check_grad
    error = check_grad(func, gradient, theta_opt.x)
    print(f"梯度误差: {error:.2e}")  # 应 <1e-5
    # print(theta_opt)
    print(f'opt_theta(deg): {np.rad2deg(theta_opt.x)}')
    print(f'cost time: {end_time-start_time} fun: {theta_opt.fun}')
    return theta_opt.x, phi0


def global_optimize_theta_sequence(n, a_sequence, ml):
    """
    全局优化整个时间序列的俯仰角序列，最小化累积代价
    :param n: 无人机数量
    :param a_sequence: 加速度序列 (T, 3)，T为时间步数
    :param ml: 载荷质量
    :return: 全局优化后的theta_sequence (T, n), phi0 (n,)
    """
    T = a_sequence.shape[0]  # 时间步数
    # if n == 1 or 2:
    #     return [np.ones([T, n]) * np.deg2rad(70)], [0]  # for test
    start_time = time.time()
    phi0 = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # 初始猜测：所有时间步的theta初始化为70度
    theta0_global = np.deg2rad(70) * np.ones(T * n)
    lambda1 = 3.1
    lambda2 = 0.05

    # 定义全局代价函数和梯度
    def global_func(x):
        """全局代价函数：张力代价 + 平滑项 + 加速度项"""
        theta_sequence = x.reshape(T, n)
        cost = 0.0

        # 张力代价
        for t in range(T):
            D = compute_D(theta_sequence[t], phi0)
            t_vec = phi_theta2_tension(D, a_sequence[t], 9.81, ml)
            cost += np.linalg.norm(t_vec) ** 2
        # 平滑项（一阶差分）
        for t in range(1, T):
            delta_theta = theta_sequence[t] - theta_sequence[t - 1]
            cost += lambda1 * np.sum(delta_theta ** 2)

        # 加速度项（二阶差分）
        for t in range(2, T):
            accel = theta_sequence[t] - 2 * theta_sequence[t - 1] + theta_sequence[t - 2]
            cost += lambda2 * np.sum(accel ** 2)
        return cost
    def global_gradient(x):
        """全局梯度：每个时间步梯度拼接为长向量"""
        theta_seq = x.reshape(T, n)
        grad = np.zeros_like(x)

        # 拉力项梯度
        for t in range(T):
            D = compute_D(theta_seq[t], phi0)
            F = ml * (a_sequence[t] + np.array([0, 0, 9.81]))
            grad[t * n:(t + 1) * n] = calGradient(theta_seq[t], phi0, D, F, n)

        # 一阶平滑项梯度
        if T > 1:
            for t in range(1, T):
                delta = theta_seq[t] - theta_seq[t - 1]
                grad[t * n:(t + 1) * n] += 2 * lambda1 * delta
                grad[(t - 1) * n:t * n] -= 2 * lambda1 * delta

        # 二阶加速度项梯度
        if T > 2:
            for t in range(2, T):
                delta = theta_seq[t] - 2 * theta_seq[t - 1] + theta_seq[t - 2]
                grad[t * n:(t + 1) * n] += 2 * lambda2 * delta
                grad[(t - 1) * n:t * n] -= 4 * lambda2 * delta
                grad[(t - 2) * n:(t - 1) * n] += 2 * lambda2 * delta

        return grad

    # 定义变量边界（所有时间步的theta均需满足[25°,70°]）
    bounds = optimize_upper_bounds(a_sequence, phi0) #FIXME:效果变差，拉力差更大
    # bounds = [(np.deg2rad(25), np.deg2rad(70)) for _ in range(T * n)]
    # 全局优化
    result = minimize(
        global_func,
        theta0_global,
        method='L-BFGS-B',
        jac=global_gradient,
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-6, 'disp': False}
    )
    from scipy.optimize import check_grad
    error = check_grad(global_func, global_gradient, result.x)
    if error > 1e-3:
      print(f"梯度误差: {error:.2e}")  # 应 <1e-5
    # 将结果重塑为(T, n)的序列
    theta_sequence = result.x.reshape(T, n)
    print(f" for {a_sequence[0]} do {np.rad2deg(theta_sequence[0])}")
    # print(f'全局优化总耗时: {time.time() - start_time:.2f}s')
    # print(f"加速度序列: {a_sequence}")
    # print(f"优化后的俯仰角序列（角度制）: {np.rad2deg(theta_sequence)}")
    return theta_sequence, phi0


def optimize_upper_bounds(a_sequence, phi0):
    """
    动态调整无人机角度优化变量的上界 保持顺序结构的向量化版本
    :param a_sequence: 加速度序列 (T, 3)
    :param phi0: 无人机初始水平夹角 (n,)
    :return: 调整后的上下界列表 [(lower, upper), ...]
    输出顺序：[(t0_uav0), (t0_uav1)...(t0_uavn), (t1_uav0)...]
    """
    T, _ = a_sequence.shape
    n = len(phi0)

    # 预计算方向向量 (n,2)
    uav_dirs = np.column_stack([np.cos(phi0), np.sin(phi0)])

    # 计算加速度方向 (T,2)
    a_horizontal = a_sequence[:, :2]
    a_norms = np.linalg.norm(a_horizontal, axis=1, keepdims=True)
    zero_acc_mask = a_norms.squeeze() < 1e-6
    a_dirs = np.divide(a_horizontal, a_norms, where=~zero_acc_mask[:, None])

    # 计算夹角矩阵 (T,n)
    cos_theta = np.einsum('td,nd->tn', a_dirs, uav_dirs)  # 保持t维度在前
    theta_deg = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))

    # 非线性上界计算（二次函数）
    angle_crisis = 90
    delta = np.clip(theta_deg - angle_crisis, 0, 180 - angle_crisis)  # 限制delta范围
    nonlinear_factor = (delta ** 2) * 10 / (angle_crisis ** 2)  # 二次增长因子
    upper = 70 + nonlinear_factor
    upper = np.minimum(upper, 80)  # 确保不超过80
    upper[zero_acc_mask, :] = 70  # 处理零加速度情况

    # 按时间步优先顺序展平
    upper_flat = upper.reshape(-1)  # 默认C顺序：t0_uav0, t0_uav1...t1_uav0...
    # 生成最终边界列表
    lower = np.deg2rad(25)
    bounds = [(lower, np.deg2rad(u)) for u in upper_flat]

    return bounds

def cal_quadpose(load_pos, theta_list, phi_list, n, L):
    """根据载荷位置以及两个角度计算四轴的位置 n个行向量
       return : dim = n x 3"""
    D = np.array([Spher2Cart_unit(theta_list[i], phi_list[i]) for i in range(n)]).T  # calculate D
    quad_pos = np.tile(load_pos, (n, 1)).T
    quad_pos = L * D + quad_pos
    return quad_pos.T


def formation_error(A, B, offset, p_l, p_f):
    """
    计算编队协同误差
    Args:
        A: 邻接矩阵（无向图固对称）
        B: 牵制矩阵
        offset: 编队形状偏移（由n个行向量组成）
        p_l: 领导者位置向量（1个行向量）
        p_f: 跟随者位置向量（n个行向量）
    Returns:
        error: 编队协同误差（n个行向量）
    """
    n_agent = len(p_f)
    error = np.zeros_like(p_f)
    for i in range(n_agent):
        for j in range(n_agent):
            error[i, :] += A[i, j] * (offset[i, :] - p_f[i, :] - offset[j, :] + p_f[j, :] )
        error[i, :] += B[i] * (offset[i, :] + p_l - p_f[i, :])
    return error

def calError_Form(A, B, p_l, p_f, theta_list, phi_list,  L, n):
    """根据俯仰角和水平角计算四轴的协同期望位置误差 n个行向量"""
    offset = cal_quadpose([0,0,0], theta_list, phi_list, n, L)
    error = formation_error(A, B, offset, p_l, p_f)
    return error

def caldError_Form(A, B, v_f):
    """计算四轴的协同期望速度误差 n个行向量"""
    error = formation_error(A, B, np.zeros_like(v_f), [0,0,0], v_f)
    return error

def demo_global():
    # 生成加速度序列 (T=2, n=4)
    n = 4
    a_sequence = np.array([
        [1, 0, 0],
        # [1, 1, 1],
        # [1, 1, 1],
        # [1, 0, 0],
        # [0, 0, 0],
        # [0, 0, 0],
    ])
    T = len(a_sequence)
    ml = 1
    # 全局优化
    print("-------------" * 5)
    print(f"进行全局优化，无人机数量: {n}, 离散时间步数: {T}")
    theta_sequence, phi0 = global_optimize_theta_sequence(n, a_sequence, ml)
    print(f'theta_opt(deg): {np.rad2deg(theta_sequence)}')
    print("-------------"*5)

def fit_circle(points):
    # 输入：点集，格式为 [[x1,y1], [x2,y2], ...]
    # 输出：圆心(a, b) 和半径r
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    A = np.vstack([x, y, np.ones(len(x))]).T
    B = -(x ** 2 + y ** 2)
    D, E, F = np.linalg.lstsq(A, B, rcond=None)[0]
    a, b = -D / 2, -E / 2
    r = np.sqrt(a ** 2 + b ** 2 - F)
    return (a, b), r

def demp_plot_circle_and_points():
    """绘制点和拟合的圆"""
    points = [[0, 2],
              # [2, 0],
              [-2.3, 0],
              [0, -2],
              ]
    center, radius = fit_circle(points)
    print(f"拟合结果: 圆心 ({center[0]:.2f}, {center[1]:.2f}), 半径 {radius:.2f}")
    # 生成圆上的点
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = center[0] + radius * np.cos(theta)
    circle_y = center[1] + radius * np.sin(theta)

    # 提取输入点的坐标
    x_points = [p[0] for p in points]
    y_points = [p[1] for p in points]

    # 绘制图形
    plt.figure(figsize=(8, 8))
    plt.scatter(x_points, y_points, color='red', label='points')
    plt.plot(circle_x, circle_y, color='blue', label='circle')
    plt.scatter([center[0]], [center[1]], color='green', marker='x', s=100, label='center')

    # 设置坐标轴范围和标签
    plt.axis('equal')
    plt.xlim(min(x_points + [center[0] - radius]) - 1, max(x_points + [center[0] + radius]) + 1)
    plt.ylim(min(y_points + [center[1] - radius]) - 1, max(y_points + [center[1] + radius]) + 1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('fit circle')
    plt.legend()
    plt.grid(True)
    plt.show()


def estimate_payload_position(drone_positions, L):
    # 转换为numpy数组
    p = np.array(drone_positions)
    n = p.shape[0]
    # 构建A矩阵和b向量
    A = p[1:] - p[0]
    b = 0.5 * (np.linalg.norm(p[1:], axis=1) ** 2 - np.linalg.norm(p[0]) ** 2)
    # 最小二乘解
    q = np.linalg.lstsq(A, b, rcond=None)[0]
    # 非线性优化修正
    from scipy.optimize import least_squares
    def residuals(q):
        return [np.linalg.norm(p_i - q) - L for p_i in p]

    result = least_squares(residuals, q, method='lm')
    return result.x

def demo_est_load():
    drones_3d = [
        [2, 3, 5],
        [5, 7, 6],
        [4, 2, 7],
        [1, 8, 4]
    ]
    L = 5.0
    q_estimated = estimate_payload_position(drones_3d, L)
    print(f"Estimated payload position: {q_estimated}")

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # demp_plot_circle_and_points()
    # demo_est_load()
    # demo_global()
    # [1.18367421e+00 4.07287754e-02 3.41368272e+01]
    # [3.00013307e+00 2.25154864e-02 2.17705628e+01]
    import matplotlib.pyplot as plt
    a_vec = np.array([[2, 0, 9.8],
                      [9.99999586, 9.99999975, 9.54623434],
                      [3.00013307e+00 ,2.25154864e-02 ,2.17705628e+01]])
    # a_vec = limit_vector_magnitude(a_vec, 5.0)
    a_vec = a_vec[0]
    n = 4
    ml = 1
    theta, phi = update_opt_phi_theta(n, a_vec, ml)
    # print(_f'theta(deg): {np.rad2deg(theta)}')
    # print(_f'phi(deg): {np.rad2deg(phi)}')
    load_pos = [0, 0, 0]
    pos_mat = cal_quadpose(load_pos, theta, phi, n, 1.5)
    D = np.array([Spher2Cart_unit(theta[i], phi[i]) for i in range(n)]).T  # calculate D
    t_vec = phi_theta2_tension(D, a_vec, 9.81, ml)
    print(f'acc: {a_vec}')
    print(f"opt tension: {t_vec}")  # 优化后的拉力分配
    theta_ori =  np.deg2rad([62, 70, 70, 70])  # [70] * 4
    D_ori = np.array([Spher2Cart_unit(theta_ori[i], phi[i]) for i in range(n)]).T  # calculate D
    t_ori = phi_theta2_tension(D_ori, a_vec, 9.81, ml)
    print(f"original tension: {t_ori}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(4):
        x_cartesian, y_cartesian, z_cartesian = np.dot(t_vec[i], Spher2Cart_unit(theta[i], phi[i]))
        # x_cartesian, y_cartesian, z_cartesian = np.dot(t_vec[i], pos_mat[i, :])
        ax.quiver(load_pos[0], load_pos[1], load_pos[2], x_cartesian, y_cartesian, z_cartesian, color='r', length=1)
    # 绘制单位球体
    # 生成球面上的点
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, 0.5 * np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='b', alpha=0.2)
    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 显示横纵坐标
    ax.set_zlim([0, 2])
    plt.show()
