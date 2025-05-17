import numpy as np
import matplotlib.pyplot as plt
from opt_Tension import compute_D,phi_theta2_tension

__all__ = [
    "calculate_power_and_energy_multi_drones",
    "plot_power_and_energy_curves",
    "limit_vector_magnitude",
    "plot_position_curves",
    "plot_error_curves",
    "plot_3d_trajectory",
    "plot_tension_curve",
    "plot_theta_curve",
    "plot_angle_curves",
    "plot_normtension_curve"
]


def calculate_power_and_energy_multi_drones(data_seq, n, T, dt,  k_m = 0.01, L=0.25,k_T=0.1, k_P=1e-4):
    """
    计算 n 架四旋翼无人机基于升力和扭矩序列的功率和总能量消耗

    参数:
    data_seq : array-like, 升力与扭矩序列 (n*T, 4)，每 n 行为一个时间步的 n 架无人机数据
    n : int, 无人机数量
    T : int, 时间步数
    L : float, 四旋翼臂长 (m)
    k_m : float, 扭矩与推力系数比值 k_tau / k_T
    k_T : float, 推力系数 (N/rpm^2)，默认值 0.1
    k_P : float, 功率系数 (W/rpm^3)，默认值 1e-4
    dt : float, 时间步长 (s)，默认值 0.1

    返回:
    energies : array, 每架无人机的总能量消耗 (n,)
    P_total_multi : array, 每架无人机每个时间步的总功率 (T, n)
    """
    # 转换为 numpy 数组
    data_seq = np.array(data_seq)
    # 检查输入维度
    if data_seq.shape != (n * T, 4):
        raise ValueError(f"data_seq 的形状必须是 ({n * T}, 4)，当前为 {data_seq.shape}")
    # 重塑数据为 (T, n, 4)，每架无人机一个时间序列
    data_reshaped = data_seq.reshape(T, n, 4)
    # 初始化推力数组 (T, n, 4)，每架无人机 4 个螺旋桨
    T_multi = np.zeros((T, n, 4))
    # 计算每架无人机的推力
    for t in range(T):
        for i in range(n):
            F = data_reshaped[t, i, 0]  # 升力
            tau_x = data_reshaped[t, i, 1]
            tau_y = data_reshaped[t, i, 2]
            tau_z = data_reshaped[t, i, 3]

            T_multi[t, i, 0] = F / 4 + tau_y / (2 * L) + tau_z / (4 * k_m)  # T1
            T_multi[t, i, 1] = F / 4 + tau_x / (2 * L) - tau_z / (4 * k_m)  # T2
            T_multi[t, i, 2] = F / 4 - tau_y / (2 * L) + tau_z / (4 * k_m)  # T3
            T_multi[t, i, 3] = F / 4 - tau_x / (2 * L) - tau_z / (4 * k_m)  # T4
    # 检查推力是否为负
    if np.any(T_multi < 0):
        print("警告：部分推力值为负，可能数据异常或模型不适用")
        T_multi = np.clip(T_multi, 0, None)
    # 计算转速 omega_i (T, n, 4)
    omega_multi = np.sqrt(T_multi / k_T)
    # 计算功率 P_i (T, n, 4)
    P_multi = k_P * omega_multi ** 3
    # 每架无人机的总功率 (T, n)
    P_total_multi = np.sum(P_multi, axis=2)
    # 计算每架无人机的总能量 (n,)
    energies = np.trapz(P_total_multi, dx=dt, axis=0)
    return energies, P_total_multi


def plot_power_and_energy_curves(t, P_total_multi, dt):
    """
    绘制 n 架无人机的功率和累积能量随时间变化的子图

    参数:
    t : array-like, 时间序列 (T,)
    P_total_multi : array-like, 每架无人机的功率序列 (T, n)
    dt : float, 时间步长 (s)
    """
    n = P_total_multi.shape[1]
    T = P_total_multi.shape[0]

    # 计算累积能量
    cumulative_energy = np.zeros((T, n))
    for i in range(n):
        for j in range(T):
            cumulative_energy[j, i] = np.trapz(P_total_multi[:j + 1, i], dx=dt)

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # 子图 1：功率曲线
    for i in range(n):
        ax1.plot(t, P_total_multi[:, i], label=f'Drone {i + 1}')
    ax1.set_ylabel('Power (W)')
    ax1.set_title('Power Consumption Over Time for Multiple Drones')
    ax1.grid(True)
    ax1.legend()

    # 子图 2：累积能量曲线
    for i in range(n):
        last_y = cumulative_energy[-1, i]
        ax2.plot(t, cumulative_energy[:, i], label=f'Drone{i + 1}-{last_y:.4f}')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Cumulative Energy (J)')
    average_energy = np.mean(cumulative_energy[-1, :])
    ax2.set_title(f'Cumulative Energy Consumption Over Time\n(Average: {average_energy:.5f} J)')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    # plt.show()


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



def plot_position_curves(store_t_list, store_posd_list, store_pos_list, para):
    figs = []
    for j in range(para.n):
        fig1, ax1 = plt.subplots(3, 1, figsize=(10, 8))

        # 绘制位置变化曲线
        ax1[0].plot(store_t_list, store_posd_list[j::para.n, 0], label=f'posd_x_{j}',linestyle='--')
        ax1[1].plot(store_t_list, store_posd_list[j::para.n, 1], label=f'posd_y_{j}',linestyle='--')
        ax1[2].plot(store_t_list, store_posd_list[j::para.n, 2], label=f'posd_z_{j}',linestyle='--')
        ax1[0].plot(store_t_list, store_pos_list[j::para.n, 0], label=f'pos_x_{j}')
        ax1[1].plot(store_t_list, store_pos_list[j::para.n, 1], label=f'pos_y_{j}')
        ax1[2].plot(store_t_list, store_pos_list[j::para.n, 2], label=f'pos_z_{j}')
        ax1[0].set_ylabel('X Position')
        ax1[0].legend()
        ax1[0].grid(True)
        ax1[1].set_ylabel('Y Position')
        ax1[1].legend()
        ax1[1].grid(True)
        ax1[2].set_ylabel('Z Position')
        ax1[2].legend()
        ax1[2].grid(True)
        fig1.suptitle(f'Position Quadrotors {j}')
        ax1[2].set_xlabel('Time Step')
        # Ensure y-axis range is at least 0.1
        for ax_idx in range(3):
            y_min, y_max = ax1[ax_idx].get_ylim()
            if y_max - y_min < 0.1:
                y_center = (y_max + y_min) / 2
                ax1[ax_idx].set_ylim(y_center - 0.05, y_center + 0.05)
        figs.append(fig1)  # 将每个子图添加到列表中
    return figs


def plot_error_curves(store_t_list, store_error_list, para):
    figs = []
    for j in range(para.n):
        fig2, ax2 = plt.subplots(3, 1, figsize=(10, 8))

        # 绘制误差变化曲线
        ax2[0].plot(store_t_list, store_error_list[j::para.n, 0], label=f'error_x_{j}')
        ax2[1].plot(store_t_list, store_error_list[j::para.n, 1], label=f'error_y_{j}')
        ax2[2].plot(store_t_list, store_error_list[j::para.n, 2], label=f'error_z_{j}')
        ax2[0].set_ylabel('X Error')
        ax2[0].legend()
        ax2[0].grid(True)
        ax2[1].set_ylabel('Y Error')
        ax2[1].legend()
        ax2[1].grid(True)
        ax2[2].set_ylabel('Z Error')
        ax2[2].legend()
        ax2[2].grid(True)
        fig2.suptitle(f'Error Quadrotors {j}')
        ax2[2].set_xlabel('Time Step')
        figs.append(fig2)  # 将每个子图添加到列表中
    return figs


def plot_3d_trajectory(store_posd_list, store_pos_list, store_load_list, para, posd_detail=False):
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')
    for j in range(para.n):
        if posd_detail is True:
            ax3d.plot(store_posd_list[j::para.n, 0], store_posd_list[j::para.n, 1], store_posd_list[j::para.n, 2],
                    label=f'posd_{j}',linestyle='--')
        ax3d.plot(store_pos_list[j::para.n, 0], store_pos_list[j::para.n, 1], store_pos_list[j::para.n, 2],
                label=f'pos_{j}')
    ax3d.plot(store_load_list[:, 0], store_load_list[:, 1], store_load_list[:, 2], label='pos_load')
    ax3d.legend()
    ax3d.grid(True)
    fig3d.suptitle('3D Trajectory')
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    return fig3d, ax3d


def plot_tension_curve(store_t_list, store_t_force_list, store_tension_vec_list, eso, para, detail=False):
    figs = []
    fig, ax1 = plt.subplots()
    for j in range(para.n):
        ax1.plot(store_t_list, store_t_force_list[j::para.n], label=f'Tension_{j}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Tension Force')
    ax1.set_title(f'Tension Force vs Time')
    ax1.grid(True)
    ax1.legend()
    if detail is False:
        return figs.append(fig)
    for j in range(para.n):
        fig2, ax2 = plt.subplots(3, 1, figsize=(10, 8))
        # 绘制拉力分量曲线
        sigma_hat_array = np.array(eso[j].sigma_hat_list)  # 将 list 转换为 NumPy 数组
        ax2[0].plot(store_t_list, store_tension_vec_list[j::para.n, 0], label=f'Real_x_{j}')
        ax2[0].plot(store_t_list[1:], para.m_load * sigma_hat_array[:, 0], label=f'Est_x_{j}', linestyle='--')
        ax2[1].plot(store_t_list, store_tension_vec_list[j::para.n, 1], label=f'Tension_y_{j}')
        ax2[1].plot(store_t_list[1:], para.m_load * sigma_hat_array[:, 1], label=f'Est_y_{j}', linestyle='--')
        ax2[2].plot(store_t_list, store_tension_vec_list[j::para.n, 2], label=f'Tension_z_{j}')
        ax2[2].plot(store_t_list[1:], para.m_load * sigma_hat_array[:, 2], label=f'Est_z_{j}', linestyle='--')
        ax2[0].set_ylabel('X')
        ax2[0].legend()
        ax2[0].grid(True)
        ax2[1].set_ylabel('Y')
        ax2[1].legend()
        ax2[1].grid(True)
        ax2[2].set_ylabel('Z')
        ax2[2].legend()
        ax2[2].grid(True)
        fig2.suptitle(f'Quad_{j} Tension')
        ax2[2].set_xlabel('Time')
        figs.append(fig2)  # 将每个子图添加到列表中

    figs.append(fig)  # 将图形添加到列表中
    return figs


def plot_normtension_curve(store_t_list, store_norm_tension_list, para):
    figs = []
    fig, ax1 = plt.subplots()
    for j in range(para.n):
        ax1.plot(store_t_list, store_norm_tension_list[:,j], label=f'Tension_norm_{j}', linestyle='--')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Tension Force')
    ax1.set_title(f'Norm Tension Force vs Time')
    ax1.grid(True)
    ax1.legend()
    return figs.append(fig)


def plot_theta_curve(store_t_list, store_theta_list, store_theta_des_list, para):
    figs = []
    fig4, ax4 = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    for j in range(para.n):
        ax4.plot(store_t_list, store_theta_list[:, j], label=f'Theta_{j}', color=colors[j])
        ax4.plot(store_t_list, store_theta_des_list[:, j], label=f'Theta_des_{j}', linestyle='--', color=colors[j])
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Theta')
    ax4.set_title(f'Theta vs Time')
    ax4.grid(True)
    ax4.legend()


    figs.append(fig4)  # 将图形添加到列表中
    return figs

def plot_angle_curves(store_t_list, store_ang_list, para):
    figs = []
    for i in range(para.n):
        # 创建新的图窗
        fig, ax = plt.subplots(3, 1, figsize=(10, 8))

        # 提取当前无人机的所有时间步数据
        drone_data = store_ang_list[i::para.n, :]  # 关键索引技巧
        # 绘制三个子图
        ax[0].plot(store_t_list, drone_data[:, 0], 'b', label='Actual Roll')
        ax[0].plot(store_t_list, drone_data[:, 3], 'r--', label='Desired Roll')
        ax[0].set_ylabel('Roll (deg)')
        ax[0].legend()
        ax[0].grid(True)
        ax[1].plot(store_t_list, drone_data[:, 1], 'b', label='Actual Pitch')
        ax[1].plot(store_t_list, drone_data[:, 4], 'r--', label='Desired Pitch')
        ax[1].set_ylabel('Pitch (deg)')
        ax[1].legend()
        ax[1].grid(True)
        ax[2].plot(store_t_list, drone_data[:, 2], 'b', label='Actual Yaw')
        ax[2].plot(store_t_list, drone_data[:, 5], 'r--', label='Desired Yaw')
        ax[2].set_ylabel('Yaw (deg)')
        ax[2].set_xlabel('Time')
        ax[2].legend()
        ax[2].grid(True)
        # Ensure y-axis range is at least 0.1
        for ax_idx in range(3):
            y_min, y_max = ax[ax_idx].get_ylim()
            if y_max - y_min < 0.1:
                y_center = (y_max + y_min) / 2
                ax[ax_idx].set_ylim(y_center - 0.05, y_center + 0.05)
        fig.suptitle(f'Quad_{i} Attitude Angle(deg)')
        figs.append(fig)

    return figs

