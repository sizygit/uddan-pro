import numpy as np


class CascadeAttitudeCtl:
    def __init__(self, Kp1, Kd1, Kp2, Ki2, J):
        """
        so3 姿态控制器 角速度环PD控制 + 角加速度环PI控制
        'High performance full attitude control of a quadrotor on SO(3)'
        """
        self.Kp1 = Kp1
        self.Kd1 = Kd1
        self.Kp2 = Kp2
        self.Ki2 = Ki2
        self.J = J
        self.integral_error_omegadot = np.zeros(3)
        self.prev_error_omega = np.zeros(3)
        self.prev_omega = np.zeros(3)  # calculate the derivative of omega
        self.prev_time = 0

    def rotation_to_logmap(self, R):
        """
        将旋转矩阵转换为 so(3) 的对数映射 SO3 -> so3
        :param R: 旋转矩阵
        :return: so(3) 向量
        """
        cos_theta = (np.trace(R) - 1) / 2
        cos_theta = np.clip(cos_theta, -1, 1)  # 防止数值误差导致 arccos 报错
        theta = np.arccos(cos_theta)
        if np.isclose(theta, 0):  # 防止数值误差导致除零错误
            return np.zeros(3)
        elif np.isclose(theta, np.pi):
            M = (R - np.transpose(R)) / 2
            w = np.array([M[2, 1], M[0, 2], M[1, 0]])
            if np.linalg.norm(w) == 0:
                w = np.array([1, 0, 0])
            else:
                w = w / np.linalg.norm(w)
            return np.pi * w
        else:
            # log(R) = theta / (2 * sin(theta)) * (R - R^T)
            omega_mat = (theta / (2 * np.sin(theta))) * (R - np.transpose(R))
            return np.array([omega_mat[2, 1], omega_mat[0, 2], omega_mat[1, 0]])

    def cal_control_torque(self, R_e, omega, dt, k1=[20, 20, 3]):
        """
        计算控制扭矩

        :param R_e: 旋转矩阵SO3误差
        :param omega: 当前的角速度
        :param omega_dot: 当前的角加速度
        :param dt: 控制间隔时间
        :return: 控制扭矩
        """
        # Angular velocity  control loop  - PD
        omega_d = k1 * self.rotation_to_logmap(R_e)
        omega_e = omega_d - omega
        omega_dot_d = self.Kp1 * omega_e + self.Kd1 * (omega_e - self.prev_error_omega) / dt
        # Angular acceleration control loop - PI
        omega_dot = (omega - self.prev_omega) / dt  # there's no feedback omega_dot
        omega_e_dot = omega_dot_d - omega_dot
        self.integral_error_omegadot += omega_e_dot * dt
        torque = (self.Kp2 * omega_e_dot + self.Ki2 * self.integral_error_omegadot
                  + np.cross(omega, self.J @ omega))
        # print(_f'w:{omega} wd:{omega_dot}')
        # print(_f'w_des:{omega_d} wd_des:{omega_dot_d} torque:{torque}')
        self.prev_error_omega = omega_e
        self.prev_omega = omega
        return torque
