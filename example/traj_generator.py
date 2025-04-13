import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

class MinimumSnapTrajectory:
    def __init__(self, waypoints, poly_order=5, max_cont_deriv=4, dim=2, ave_v=[1,1,1],
                 max_acceleration = None, time_alloc= None ):
        """
        初始化轨迹生成器:将每两个航路点之间的路径段视为一个多项式轨迹段，通过最小化snap来平滑轨迹
        参数:
        waypoints : np.array (N, 2/3) - 航路点坐标 (x, y) 或者 (x, y, z)
        poly_order : int - 多项式段阶数 (推荐奇数，如5阶多项式)
        max_cont_deriv : int - 需要连续的导数阶数 (通常设为4以保证snap连续)
        """
        self.waypoints = waypoints
        self.n_segments = len(waypoints) - 1 #TODO: 自定义轨迹段数分配
        self.poly_order = poly_order
        self.max_cont_deriv = max_cont_deriv
        self.dim = dim  # x和y维度
        self.t_sum = 1
        self.max_acceleration = max_acceleration #TODO: 添加最大加速度约束失败
        # 时间分配（均匀分配，可自定义）
        if time_alloc is None:
            time_alloc = self.cal_time_alloc(waypoints, ave_v)  # 根据距离和速度计算时间节点
            self.t_sum = time_alloc[-1]
            self.time_alloc = time_alloc/ self.t_sum  # 归一化时间为[0, 1]
            print(f"航路点时间节点: {time_alloc}")
        else:
            self.t_sum = time_alloc[-1]
            self.time_alloc = time_alloc / self.t_sum  # 归一化时间为[0, 1]
        # 每个多项式段的系数矩阵 (n_segments x (poly_order+1) x dim)
        self.coeffs = None

    def generate_trajectory(self):
        """ 生成最小Snap轨迹
        Returns:  每一段轨迹的多项式对应的时间为t_i \in [0, delta_t_i],长度为对应的时间间隔长度
        coeffs: np.array - 降幂多项式系数矩阵 (n_segments x (poly_order+1) x dim)"""
        # 构造优化问题
        start_time = time.time()
        x0 = self._initial_guess()
        constraints = self._build_constraints()
        result = minimize(self._cost_function, x0,
                          constraints=constraints,
                          method='SLSQP',
                          options={'maxiter': 1000})
        if not result.success:
            raise RuntimeError("优化失败: " + result.message)
        print(f'轨迹规划总耗时: {time.time() - start_time:.2f}s')
        # 提取系数
        self.coeffs = self._unpack_coefficients(result.x)
        return self.coeffs

    def _cost_function(self, x):
        """ 轨迹平滑度代价函数 """
        coeffs = self._unpack_coefficients(x)
        cost = 0.0
        for seg in range(self.n_segments):
            T = self.time_alloc[seg + 1] - self.time_alloc[seg]
            # 获取四阶导数（snap）系数，形状为 (n_coeffs_snap, dim)
            snap_coeffs = self._get_derivative_coefficients(coeffs[seg], 4)
            # 累加多维度积分结果
            cost += self._integrate_square(snap_coeffs, T)
        return cost


    def _position_constraint(self, seg, dim, t_frac):
        """ 位置约束辅助函数 """
        def constraint(x):
            coeffs = self._unpack_coefficients(x)
            # 计算段内相对时间（0或段持续时间）
            if t_frac == 0:
                t = 0.0
            else:
                t = self.time_alloc[seg + 1] - self.time_alloc[seg]  # 段的持续时间
            poly = coeffs[seg, :, dim]
            pos = self._eval_derivative(poly, 0, t)  # 0阶导数即位置
            target = self.waypoints[seg + t_frac, dim]
            return pos - target
        return constraint

    def _continuity_constraint(self, wp, deriv, dim):
        """ 修正后的连接点连续性约束 """
        def constraint(x):
            coeffs = self._unpack_coefficients(x)
            # 前一段的结束时间点
            prev_seg = wp - 1
            t_prev = self.time_alloc[wp] - self.time_alloc[wp - 1]  # 前一段持续时间
            # 当前段的开始时间点
            curr_seg = wp
            t_curr = 0.0
            # 计算两段在连接点的deriv阶导数值
            prev_val = self._eval_derivative(coeffs[prev_seg, :, dim], deriv, t_prev)
            curr_val = self._eval_derivative(coeffs[curr_seg, :, dim], deriv, t_curr)
            return prev_val - curr_val  # 约束差值应为0
        return constraint

    def _eval_derivative(self, coeffs, deriv, t):
        """ 支持处理一维（单维度）或二维（多维度）系数数组 """
        if coeffs.ndim == 1:
            # 单一维度处理
            d_coeffs = np.polyder(coeffs, deriv)
            return np.polyval(d_coeffs, t)
        else:
            # 多维度处理
            values = []
            for dim in range(coeffs.shape[1]):
                d_coeffs = np.polyder(coeffs[:, dim], deriv)
                values.append(np.polyval(d_coeffs, t))
            return np.array(values)

    def _integrate_square(self, coeffs, T):
        """ 计算多维度多项式系数的平方积分（适配二三维输入） """
        total_cost = 0.0
        # 遍历每个维度 (x, y, z)
        for dim in range(coeffs.shape[1]):
            # 提取单维度系数（一维数组）
            single_dim_coeffs = coeffs[:, dim]
            # 计算平方多项式 (coeffs^2)
            squared_coeffs = np.polymul(single_dim_coeffs, single_dim_coeffs)
            # 积分多项式
            integral_coeffs = np.polyint(squared_coeffs)
            # 在时间T处求值并累加
            total_cost += np.polyval(integral_coeffs, T)
        return total_cost

    def _initial_guess(self):
        """ 初始猜测（线性插值） """
        return np.zeros(self.n_segments * (self.poly_order + 1) * self.dim)

    def _unpack_coefficients(self, x):
        """ 将优化变量解包为系数矩阵 """
        return x.reshape(self.n_segments, self.poly_order + 1, self.dim)

    def _get_derivative_coefficients(self, coeffs, deriv):
        """ 生成多维度导数系数（二维数组） """
        d_coeffs = []
        for dim in range(self.dim):
            single_dim_coeffs = coeffs[:, dim]
            d_coeffs.append(np.polyder(single_dim_coeffs, deriv))
        # 列堆叠形成 (n_coeffs, dim) 的二维数组
        return np.column_stack(d_coeffs)

    def plot_trajectory(self, n_points=100):
        """ 可视化轨迹 """
        t_total = np.linspace(self.time_alloc[0], self.time_alloc[-1], n_points)
        # t_total = np.linspace(0, 1, n_points)
        trajectory = []
        # 计算每个时间点的轨迹位置
        for t in t_total:
            seg = np.searchsorted(self.time_alloc, t) - 1
            seg = min(max(seg, 0), self.n_segments - 1)
            t_seg = t - self.time_alloc[seg]
            # pos = [self._eval_derivative(self.coeffs[seg, :, dim], 0, t_seg)
            #        for dim in range(self.dim)]
            pos = [np.polyval(self.coeffs[seg, :, dim], t_seg)
                   for dim in range(self.dim)]
            trajectory.append(pos)

        trajectory = np.array(trajectory)

        plt.figure(figsize=(10, 6))
        if self.dim == 2:
            plt.plot(self.waypoints[:, 0], self.waypoints[:, 1], 'ro', label='waypoints', markersize=8)
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='trajectory')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
        elif self.dim == 3:
            ax = plt.axes(projection='3d')  # 创建3D坐标轴
            ax.scatter3D(self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2], c='r', label='waypoints', s=50)
            ax.plot3D(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', label='trajectory')
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            # 设置 z 轴的范围，确保最小尺度不小于 0.1
            z_min, z_max = ax.get_zlim()
            if z_max - z_min < 0.1:
                z_center = (z_max + z_min) / 2
                ax.set_zlim(z_center - 0.05, z_center + 0.05)

    def _build_constraints(self):
        """ 构造约束条件 航路点位置约束、中间点max_cont_deriv阶连续约束、起点终点零速度加速度约束"""
        constraints = []
        # 航路点位置约束（必须严格通过）
        for seg in range(self.n_segments):
            for dim in range(self.dim):
                # 当前段的起点必须等于航路点
                con_start = {'type': 'eq',
                             'fun': self._position_constraint(seg, dim, 0)}
                # 当前段的终点必须等于下一航路点
                con_end = {'type': 'eq',
                           'fun': self._position_constraint(seg, dim, 1)}
                constraints.extend([con_start, con_end])

        # 连接点处的导数连续性约束（默认C⁴连续）max_cont_deriv
        for wp in range(1, self.n_segments):
            for deriv in range(1, self.max_cont_deriv + 1):
                for dim in range(self.dim):
                    con = {'type': 'eq',
                           'fun': self._continuity_constraint(wp, deriv, dim)}
                    constraints.append(con)

        # 边界条件：起点和终点的速度、加速度为零
        for dim in range(self.dim):
            # 起点速度约束
            con_start_vel = {'type': 'eq',
                             'fun': self._boundary_deriv_constraint(0, 1, dim)}
            # 起点加速度约束
            con_start_acc = {'type': 'eq',
                             'fun': self._boundary_deriv_constraint(0, 2, dim)}
            # 终点速度约束
            con_end_vel = {'type': 'eq',
                           'fun': self._boundary_deriv_constraint(-1, 1, dim)}
            # 终点加速度约束
            con_end_acc = {'type': 'eq',
                           'fun': self._boundary_deriv_constraint(-1, 2, dim)}
            constraints.extend([con_start_vel,  #con_start_acc,
                                con_end_vel, con_end_acc])
        # 添加最大加速度约束  不可行
        # if self.max_acceleration is not None:
        #     for seg in range(self.n_segments):
        #         for dim in range(self.dim):
        #             con_max_acc = {'type': 'ineq',
        #                            'fun': self._max_acceleration_constraint(seg, dim)}
        #             constraints.append(con_max_acc)
        return constraints

    def _max_acceleration_constraint(self, seg, dim):
        #TODO 未添加 """ 最大加速度约束  """
        def constraint(x):
            coeffs = self._unpack_coefficients(x)
            T = self.time_alloc[seg + 1] - self.time_alloc[seg]
            # 计算加速度的导数系数
            acc_coeffs = self._get_derivative_coefficients(coeffs[seg], 2)  # 2阶导数即加速度
            # 在段内多个时间点评估加速度
            t_points = np.linspace(0, T, num=100)  # 100个时间点
            max_acc = np.max(np.abs(self._eval_derivative(acc_coeffs, 0, t_points)))
            return self.max_acceleration - max_acc  # 确保加速度小于最大值
        return constraint

    def _boundary_deriv_constraint(self, seg_idx, deriv, dim):
        """ 边界导数约束 """
        def constraint(x):
            coeffs = self._unpack_coefficients(x)
            if seg_idx == -1:  # 最后一段的结束点
                seg = self.n_segments - 1
                t = self.time_alloc[-1] - self.time_alloc[-2]
            else:  # 第一段的起始点
                seg = seg_idx
                t = 0.0
            return self._eval_derivative(coeffs[seg, :, dim], deriv, t)
        return constraint

    def cal_trajectory_order(self, t_ori, _deriv):
        """ 在任意时间t(原始时间)计算轨迹位置或高阶状态 """
        if np.isscalar(t_ori):
            t_ori = np.array([t_ori])  # 将标量转换为一维数组
        # 检查时间范围
        if np.any(t_ori > self.t_sum):
            t_ori = np.where(t_ori > self.t_sum, self.t_sum, t_ori)  # 超出时间范围的均保持最后时刻的状态
        # 归一化时间
        t_norm = t_ori / self.t_sum
        # 初始化结果列表
        results = []
        for t_val in t_norm:
            seg = np.searchsorted(self.time_alloc, t_val) - 1
            seg = min(max(seg, 0), self.n_segments - 1)
            t_seg = t_val - self.time_alloc[seg]
            # 评估每个维度并收集结果
            result = np.array([self._eval_derivative(self.coeffs[seg, :, dim], _deriv, t_seg)
                               for dim in range(self.dim)])
            results.append(result)
        # 将结果转换为二维数组
        return np.array(results) if len(results) > 1 else results[0]

    def plot_derivatives(self, deriv_order=1, n_points=100):
        """ 可视化指定阶导数随时间变化 """
        t_total = np.linspace(self.time_alloc[0], self.time_alloc[-1], n_points)
        derivatives = []
        for t in t_total:
            seg = np.searchsorted(self.time_alloc, t) - 1
            seg = min(max(seg, 0), self.n_segments - 1)
            t_seg = t - self.time_alloc[seg]  # 段内相对时间
            deriv = [self._eval_derivative(self.coeffs[seg, :, dim], deriv_order, t_seg)
                     for dim in range(self.dim)]
            derivatives.append(deriv)
        derivatives = np.array(derivatives)
        if self.time_alloc is not None:
            t_total *= self.t_sum
        plt.figure(figsize=(10, 4))
        plt.plot(t_total, derivatives[:, 0], label='X-derivatives')
        plt.plot(t_total, derivatives[:, 1], label='Y-derivatives')
        if self.dim == 3:
            plt.plot(t_total, derivatives[:, 2], label='Z-derivatives')
        plt.title(f'{deriv_order}-derivatives curves')
        plt.xlabel('time')
        plt.ylabel(f'{deriv_order}-derivatives value')
        plt.legend()
        plt.grid(True)

    def cal_time_alloc(self, waypoints, average_speed):
        """
        根据航路点和平均速度计算时间节点
        参数:
        waypoints : np.array (N, 3) - 航路点坐标 (x, y, z)
        average_speed : np.array (3,) - 平均速度向量 (vx, vy, vz)
        返回:
        time_nodes : np.array - 时间节点数组
        """
        # 计算速度的模长
        speed_magnitude = np.linalg.norm(average_speed)
        if speed_magnitude == 0:
            raise ValueError("平均速度不能为零向量")
        # 计算航路点之间的距离
        distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)  # 计算相邻航路点之间的距离
        # 计算每段的时间
        times = distances / speed_magnitude  # 每段的时间
        # 计算时间节点，起始时间为0
        time_nodes = np.zeros(len(times) + 1)  # 包含起始点
        time_nodes[1:] = np.cumsum(times)  # 使用cumsum进行累加
        return time_nodes


# 使用示例
if __name__ == "__main__":
    # 定义航路点 (N x 3)
    waypoints = np.array([
        # [0.0, 0.0, 1.0],
        [0, 0, 2],
        [0.5, 0, 2],
        [1, 0, 2],
        [3, 0, 2],
        [3, 2, 2],
        [3, 4, 2]
    ])
    traj_gen1 = MinimumSnapTrajectory(waypoints, poly_order=5, dim=3, ave_v=[1, 1, 1])
    traj_gen1.coeffs = np.load('traj3.npy')
    traj_gen1.plot_trajectory()
    plt.show()

    # t_alloc = np.array([0.0, 4.4, 7, 10., 12.0])
    # 创建轨迹生成器
    traj_gen = MinimumSnapTrajectory(waypoints, poly_order=5, dim=3,ave_v=[1,1,1],
                                     # time_alloc=t_alloc,
                                     # max_acceleration=[5,5,5],
                                     )
    # t = traj_gen.cal_time_alloc(waypoints, [1, 1, 1])
    try:
        coeffs = traj_gen.generate_trajectory()
        # np.save('traj3.npy', coeffs)  # 文件扩展名为 .npy
        print("轨迹生成成功！")
        # 可视化
        traj_gen.plot_trajectory()
        # 可视化轨迹导数变化
        traj_gen.plot_derivatives(deriv_order=2)
        traj_gen.plot_derivatives(deriv_order=1)
        traj_gen.plot_derivatives(deriv_order=0)
        # 验证终点状态
        _deriv = 0
        end_vel = traj_gen._eval_derivative(coeffs[-1, :, :], _deriv, traj_gen.time_alloc[-1] - traj_gen.time_alloc[-2])
        print(f"终点{_deriv}阶状态:{end_vel}")
        p = traj_gen.cal_trajectory_order(5, 2)
        print(p)
        plt.show()
    except RuntimeError as e:
        print("轨迹生成错误:", str(e))