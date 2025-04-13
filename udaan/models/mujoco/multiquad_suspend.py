import numpy as np
import copy
import time
import mujoco_viewer

from ..mujoco import MujocoModel, mujoco
from .. import base
from ... import utils
from ...traj import trajectory_generator
import enum
from ... import manif
import sys

sys.path.append("..")  # Adds higher directory (expamle) to python modules path.
from example.ESO import ExtendedStateObserver
from example.so3Attitude import CascadeAttitudeCtl


class MulitQuadCS(base.BaseModel):
    """
    个人编写的多个四旋翼飞行器与悬挂载荷系统的控制模型
    """

    class INPUT_TYPE(enum.Enum):
        FORCE = 1  # thrust [N] (scalar), torque [Nm] (3x1) : (4x1)

    class State(object):
        """
        State class for the multi-quadrotors with cable-suspended payload system.
        """

        class Entity(object):
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class Quadrotor(Entity):
            def __init__(self, **kwargs):
                # 定义实例变量
                self.rotation = np.eye(3)
                self.angular_velocity = np.zeros(3)
                self.position = np.zeros(3)
                self.velocity = np.zeros(3)
                self.acc = np.zeros(3)
                self.integral = np.zeros(3)
                self.q = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
                super().__init__(**kwargs)

        class Cable(Entity):
            def __init__(self, L, **kwargs):
                # 定义实例变量
                self.q = np.array([0.0, 0.0, 1.0])
                self.omega = np.zeros(3)
                self.dq = np.zeros(3)
                self.length = L
                self.tension = np.zeros(3)
                self.t_froce = 0.0
                super().__init__(**kwargs)

        def __init__(self, nQ: int, L):
            self.num_quad = nQ  # number of quadrotors
            self.quads = [self.Quadrotor() for _ in range(nQ)]
            self.cables = [self.Cable(L) for _ in range(nQ)]
            self.load_pos = np.zeros(3)
            self.load_vel = np.zeros(3)
            self.load_acc = np.zeros(3)


        def reset(self):
            for quad in self.quads:
                quad.reset()
            for cable in self.cables:
                cable.reset()
            self.load_pos = np.zeros(3)
            self.load_vel = np.zeros(3)

    def __init__(self, paraStruct, **kwargs):
        if "render" in kwargs:
            self._mj_render = kwargs["render"]
        else:
            self._mj_render = False
        if "num_quad" in kwargs:
            self.nQ = kwargs["num_quad"]
        else:
            raise ValueError("Number of quadrotors not provided")

        super().__init__(**kwargs)  # init base.BaseModel()
        self.state = self.State(nQ=self.nQ, L=paraStruct.l)  # state object
        self.eso = [ExtendedStateObserver(paraStruct.quad_poso[j], [0., 0., 0.],
                                          # [0.] * 3, [0.] * 3, [0.] * 3, [0.] * 3, [0.] * 3, [0.] * 3,
                                          [1.7, 1.7, 5.], [1.7, 1.7, 3.], [12., 12., 20.],  # 32
                                          [1.6, 1.6, 1], [1., 1.7, 2.0], [10., 10., 14.], # [1.6, 1.6, 0.2], [1., 1.7, 2.0], [10., 10., 10.],
                                          0.5, 0.1)
                    for j in range(paraStruct.n)]  # init extern state observer
        self.so3Ctl = [CascadeAttitudeCtl([2.1, 2.1, 2.1], [1.7, 1.7, 1.3],
                                          np.diag(paraStruct.diaginertia) @ [0.3, 0.3, 0.3],
                                          np.diag(paraStruct.diaginertia) @ [0.01, 0.01, 0.01],
                                          np.diag(paraStruct.diaginertia)) for _ in range(paraStruct.n)]
        self.Flift_vec_store = np.zeros(3 * self.nQ)
        self.mj_dt = paraStruct.dt
        # system parameters
        self.mQ = np.array([paraStruct.m_quad] * self.nQ)
        self.mL = paraStruct.m_load
        self._inertia_matrix = np.diag(paraStruct.diaginertia)
        self._min_thrust = paraStruct.ctrlrange_dict['thrust'][0]
        self._max_thrust = paraStruct.ctrlrange_dict['thrust'][1]
        self._min_torque = np.array([paraStruct.ctrlrange_dict['Mx'][0], paraStruct.ctrlrange_dict['My'][0],
                                     paraStruct.ctrlrange_dict['Mz'][0]])
        self._max_torque = np.array([paraStruct.ctrlrange_dict['Mx'][1], paraStruct.ctrlrange_dict['My'][1],
                                     paraStruct.ctrlrange_dict['Mz'][1]])
        # self._prop_min_force = 0.0
        # self._prop_max_force = 10.0
        # self._input_type = self.INPUT_TYPE.FORCE
        # 拼接推力和扭矩的最大最小值，并扁平化
        self._wrench_min = np.concatenate([[self._min_thrust], self._min_torque])
        self._wrench_max = np.concatenate([[self._max_thrust], self._max_torque])
        self._feasible_min_input = np.matlib.repmat(self._wrench_min, self.nQ, 1).flatten()
        self._feasible_max_input = np.matlib.repmat(self._wrench_max, self.nQ, 1).flatten()
        self._step_freq = 100.0
        self._step_iter = max(1, int(1.0 / self._step_freq / self.sim_timestep))

        # mujoco model param handling
        self._attitude_zoh = False
        if "model_xml_path" in kwargs:
            self._mjMdl = MujocoModel(model_path=kwargs["model_xml_path"],
                                      render=self._mj_render, pwdpath=True)
        else:
            self._mjMdl = MujocoModel(model_path="multi%d_quad_pointmass.xml" % (self.nQ),
                                      render=self._mj_render)  # get MjModel and mujoco_viewer.MujocoViewer
        if "attitude_zoh" in kwargs:  # TODO: ?????
            self._attitude_zoh = kwargs["attitude_zoh"]
        # 使用lambda函数与列表推导式来定义对应数据的索引序列
        # MjData.qpos = [pos1,quat1,...] MjData.qvel=[vel1, omega1,...]
        self._mj_qpos_id = lambda quad_id: [7 * quad_id + j for j in range(3)]
        self._mj_qquat_id = lambda quad_id: [7 * quad_id + 3 + j for j in range(4)]
        self._mj_qvel_id = lambda quad_id: [6 * quad_id + j for j in range(3)]
        self._mj_qomega_id = lambda quad_id: [6 * quad_id + j + 3 for j in range(3)]

        self._mj_loadpos_id = [i for i in range(self.nQ * 7, self.nQ * 7 + 3)]
        self._mj_loadquat_id = [i for i in range(self.nQ * 7 + 3, self.nQ * 7 + 7)]
        self._mj_loadvel_id = [i for i in range(self.nQ * 6, self.nQ * 6 + 3)]
        self._mj_loadomega_id = [i for i in range(self.nQ * 6 + 3, self.nQ * 6 + 6)]
        # self._mj_cable_tendon_index = 0
        self._ctrl_index = 0

        self._mjDt = 1.0 / 100.0  # mujoco timestep 10 ms
        self._step_iter = int(self.sim_timestep / self._mjDt)
        self._nFrames = 1
        if self._attitude_zoh:
            self._step_iter, self._nFrames = self._nFrames, self._step_iter

    def step_data(self, u_clamped):
        """step the simulation"""
        for _ in range(self._step_iter):
            # set control
            self._mjMdl.data.ctrl[self._ctrl_index:self._ctrl_index +
                                                   4 * self.nQ] = u_clamped
            self._mjMdl._step_mujoco_simulation(self._nFrames)  # advance the simulation by a specified number of frames
            self._query_latest_state()  # Retrieve the latest state from the simulation

        return

    def reset(self, **kwargs):
        """reset state and time"""
        self.t = 0.0
        self._mjMdl.reset()
        self._query_latest_state()
        return

    def _query_latest_state(self):  # query update state from mujoco
        self.t = self._mjMdl.data.time
        for i in range(self.nQ):  # 使用deepcopy避免浅拷贝（引用赋值）
            self.state.quads[i].position = copy.deepcopy(self._mjMdl.data.qpos[7 * i:7 * i +
                                                                                     3])
            _quat = copy.deepcopy(self._mjMdl.data.qpos[7 * i + 3:7 * i + 7])
            self.state.quads[i].q = _quat
            self.state.quads[i].rotation = self._mjMdl._quat2rot(_quat)  # rotation matrix
            self.state.quads[i].velocity = copy.deepcopy(self._mjMdl.data.qvel[6 * i:6 * i +
                                                                                     3])
            self.state.quads[i].angular_velocity = copy.deepcopy(self._mjMdl.data.qvel[6 * i +
                                                                                       3:6 * i + 6])
            self.state.quads[i].acc = copy.deepcopy(self._mjMdl.data.sensordata[4+i*3 : 4+i*3+3])
        self.state.load_pos = copy.deepcopy(self._mjMdl.data.qpos[7 * self.nQ:
                                                                  7 * self.nQ + 3])
        self.state.load_vel = copy.deepcopy(self._mjMdl.data.qvel[6 * self.nQ:
                                                                  6 * self.nQ + 3])
        self.state.load_acc = copy.deepcopy(self._mjMdl.data.sensordata[4+3 * self.nQ :4+3 * self.nQ +3])
        for i in range(self.nQ):
            # T = |T| .* q  q: unit vector load -> quad
            p =  self.state.quads[i].position - self.state.load_pos
            self.state.cables[i].length = np.linalg.norm(p)
            self.state.cables[i].q = p / self.state.cables[i].length
            self.state.cables[i].dq = (self.state.load_vel -
                                       self.state.quads[i].velocity)
            self.state.cables[i].omega = np.cross(self.state.cables[i].q,
                                                  self.state.cables[i].dq)
            self.state.cables[i].t_froce = copy.deepcopy(self._mjMdl.data.sensordata[i])
            self.state.cables[i].tension = copy.deepcopy(self._mjMdl.data.sensordata
                                                         [i] * self.state.cables[i].q)

    def quad_position_control_mpc(self, posd_error, f_margin, fromationUsed=False):
        thrust_vec = np.zeros(3 * self.nQ)
        for i in range(self.nQ):
            kp = np.array([2.6, 2.6, 6.1])
            kd = np.array([4, 4, 5.9])
            if fromationUsed is True:
                ex = posd_error[i, :]  # 提取期望位置误差
                ev = posd_error[i + 4, :]  # 提取期望速度误差
            else:
                ex = posd_error[i] - self.state.quads[i].position
                ev = np.zeros(3) - self.state.quads[i].velocity
            Fpd = kp * ex + kd * ev
            Fff = self.mQ[i] * (self._g * self._e3)

            A_nor = self.Flift_vec_store[3 * i:3 * i + 3] / self.mQ[i] - np.array([0, 0, self._g])
            _, _, Disturbance_hat = self.eso[i].update(self.state.quads[i].position, self.state.quads[i].velocity,
                                                       A_nor,
                                                       dt=self.mj_dt)
            # print(_f'Disturbance_hat: {np.linalg.norm(Disturbance_hat):.2f}*{Disturbance_hat/np.linalg.norm(Disturbance_hat)}   q: {self.state.cables[0].q}')
            tmp_F = Fpd + Fff + f_margin[i] - Disturbance_hat * self.mQ[i]
            thrust_vec[3 * i:3 * i + 3] = tmp_F
            # thrust_vec[3 * i:3 * i + 3] = self.limit_vector_angle(tmp_F, 35)  # 期望的推力向量
            # print(_f'ex:{ex}  exp_f:{Fpd - Disturbance_hat * self.mQ[i]}')
        return thrust_vec


    def quad_position_control(self, posd_error, f_margin,fromationUsed=False):
        """quadrotor position control """
        thrust_vec = np.zeros(3 * self.nQ)
        for i in range(self.nQ):
            kp = np.array([2.6, 2.6, 10.1])  #2.6 2.6 6.1
            kd = np.array([5, 5, 5.9])  # 4, 4, 5.9
            ki = np.array([1.8, 1.8, 1.8])
            if fromationUsed is True:
                ex = posd_error[i, :]  # 提取期望位置误差
                # ev = posd_error[i+4, :]  # 提取期望速度误差
                ev = -self.state.quads[i].velocity
            else:
                ex = posd_error[i] - self.state.quads[i].position
                ev = -self.state.quads[i].velocity
            self.state.quads[i].integral += ex * self.mj_dt
            # 应用积分限幅
            self.state.quads[i].integral = np.clip(
                self.state.quads[i].integral,
                -100,
                100
            )
            Fpd = kp * ex + kd * ev #+ ki * self.state.quads[i].integral
            # print(f'q{i} integ: {self.state.quads[i].integral}')
            Fff = self.mQ[i] * (self._g * self._e3)

            A_nor = self.Flift_vec_store[3 * i:3 * i + 3] / self.mQ[i] - np.array([0, 0, self._g])
            _, _, Disturbance_hat = self.eso[i].update(self.state.quads[i].position, self.state.quads[i].velocity,
                                                       A_nor,
                                                       dt=self.mj_dt)
            # print(_f'Disturbance_hat: {np.linalg.norm(Disturbance_hat):.2f}*{Disturbance_hat/np.linalg.norm(Disturbance_hat)}   q: {self.state.cables[0].q}')
            tmp_F = Fpd + Fff + f_margin[i]  - Disturbance_hat * self.mQ[i]
            # tmp_F = Fpd + Fff + f_margin[i] + self.state.cables[i].tension #TODO 用真实的拉力
            # tmp_F += 0.9 * self.state.cables[i].tension
            thrust_vec[3 * i:3 * i + 3] = tmp_F
            # 抗饱和处理：当推力饱和时冻结积分
            # if np.linalg.norm(tmp_F) > self.max_thrust:
            #     self.integral_errors[i] -= ex * self.mj_dt  # 回退本次积分更新
            # thrust_vec[3 * i:3 * i + 3] = self.limit_vector_angle(tmp_F, 35)  # 期望的推力向量
            # print(_f'ex:{ex}  exp_f:{Fpd - Disturbance_hat * self.mQ[i]}')
        return thrust_vec

    def compute_attitude_control_old(self, thrust_force_all):
        """原论文采用的SO3 controller"""
        u_vec = np.zeros(4 * self.nQ)  # _f tau1 tau2 tau3
        for i in range(self.nQ):
            thrust_force = thrust_force_all[3 * i:3 * i + 3]
            norm_thrust = np.linalg.norm(thrust_force)  # norm of thrust force
            b1d = np.array([1.0, 0.0, 0.0])
            b3c = thrust_force / norm_thrust  # Unit vector of the required thrust direction
            b3_b1d = np.cross(b3c, b1d)
            norm_b3_b1d = np.linalg.norm(b3_b1d)
            b1c = (-1 / norm_b3_b1d) * np.cross(b3c, b3_b1d)
            b2c = np.cross(b3c, b1c)
            Rd = np.hstack([
                np.expand_dims(b1c, axis=1),
                np.expand_dims(b2c, axis=1),
                np.expand_dims(b3c, axis=1),
            ])
            R = self.state.quads[i].rotation
            # Rd = self.limit_euler_angles(Rd, 35)  # 限制欧拉角
            Omega = self.state.quads[i].angular_velocity
            Omegad = np.zeros(3)  # TODO add differential flatness
            dOmegad = np.zeros(3)  # TODO add differential flatness

            # attitude control
            tmp = 0.5 * (Rd.T @ R - R.T @ Rd)
            eR = np.array([tmp[2, 1], tmp[0, 2], tmp[1, 0]])  # vee-map
            eOmega = Omega - R.T @ Rd @ Omegad

            kR = np.array([1.4, 1.4, 1.35])
            kOm = np.array([0.35, 0.35, 0.225])

            M = -kR * eR - kOm * eOmega + np.cross(Omega,
                                                   self._inertia_matrix @ Omega)
            M += (-1 * self._inertia_matrix
                  @ (manif.hat(Omega) @ R.T @ Rd @ Omegad - R.T @ Rd @ dOmegad))
            # ignoring this for since Omegad is zero
            f = thrust_force.dot(R[:, 2])  # 推力向量在机体坐标系的z轴方向分量
            u_clamped = np.clip(np.hstack([f, M]), self._feasible_min_input[4 * i:4 * i + 4],
                                self._feasible_max_input[4 * i:4 * i + 4])
            u_vec[4 * i:4 * i + 4] = u_clamped
            self.Flift_vec_store[3 * i:3 * i + 3] = u_clamped[0] * self.state.quads[i].rotation[:, 2]  # 提供给ESO
            # print(_f'_f:{_f:.2f} M:{M}')
        return u_vec

    def compute_attitude_control_cascade(self, thrust_force_all, k1=[12, 12, 3]):
        """ 1 ： SO3 controller"""
        u_vec = np.zeros(4 * self.nQ)  # _f tau1 tau2 tau3
        angle_vec  = np.zeros([self.nQ, 6])
        for i in range(self.nQ):
            thrust_force = thrust_force_all[3 * i:3 * i + 3]
            norm_thrust = np.linalg.norm(thrust_force)  # norm of thrust force
            b1d = np.array([1.0, 0.0, 0.0])
            b3c = thrust_force / norm_thrust  # Unit vector of the required thrust direction
            b3_b1d = np.cross(b3c, b1d)
            norm_b3_b1d = np.linalg.norm(b3_b1d)
            b1c = (-1 / norm_b3_b1d) * np.cross(b3c, b3_b1d)
            b2c = np.cross(b3c, b1c)
            Rd = np.hstack([
                np.expand_dims(b1c, axis=1),
                np.expand_dims(b2c, axis=1),
                np.expand_dims(b3c, axis=1),
            ])
            R = self.state.quads[i].rotation
            # Rd = self.limit_euler_angles(Rd, 35)  # 限制欧拉角
            Omega = self.state.quads[i].angular_velocity
            Re = Rd @ R.T
            angle_vec[i, :] = np.hstack([self.rot_to_euler(R), self.rot_to_euler(Rd)])  # 姿态角误差：deg
            M = self.so3Ctl[i].cal_control_torque(Re, Omega, self.mj_dt)
            f = thrust_force.dot(R[:, 2])  # 推力向量在机体坐标系的z轴方向分量
            u_clamped = np.clip(np.hstack([f, M]), self._feasible_min_input[4 * i:4 * i + 4],
                                self._feasible_max_input[4 * i:4 * i + 4])
            u_vec[4 * i:4 * i + 4] = u_clamped
            self.Flift_vec_store[3 * i:3 * i + 3] = u_clamped[0] * self.state.quads[i].rotation[:, 2]
            # print(f'f:{f:.2f} M:{M}')
        return u_vec, angle_vec

    def simulate(self, tf, **kwargs):
        """在类里面实现完整的仿真过程，建议在主函数单独构建，而不是调用该方法"""
        self.reset(**kwargs)
        start_t = time.time_ns()
        cam = self._mjMdl.viewer.cam
        cam.distance = 10
        while self.t < tf:
            u = [8, 0, 0, 0] * self.nQ
            u_clamped = np.clip(u, self._feasible_min_input,
                                self._feasible_max_input)
            self.step_data(u)
            # add tracking marker
            # if self._mjMdl.render:
            #     for j in range(self.nQ):
            #         self._mjMdl.add_arrow_at(
            #             p = self._mjMdl.data.qpos[self._mj_qpos_id(j)],  # position
            #             R = self.state.quads[j].rotation,  # rotation matrix
            #             s = self.state.quads[j].velocity,  # scale
            #             # label="quad%d" % j,  # label
            #             color=[1.0, 1.0, 0.0, 1.0],  # color
            #          )
        end_t = time.time_ns()
        print("Took (%.4f)s for simulating (%.4f)s" %
              (float(end_t - start_t) * 1e-9, self.t))
        pass

    def limit_vector_angle(self, vector, max_angle_deg):
        # 将向量转换为 numpy 数组
        vector = np.array(vector)
        # 计算向量的模长
        norm_vector = np.linalg.norm(vector)
        # 确保向量不为零向量
        if norm_vector == 0:
            return vector
        # 单位化向量
        unit_vector = vector / norm_vector
        # z 轴的单位向量
        z_axis = np.array([0, 0, 1])
        # 计算向量与 z 轴的夹角（弧度）
        angle = np.arccos(np.dot(unit_vector, z_axis))
        # 角度范围（弧度）
        max_angle = np.deg2rad(max_angle_deg)
        # 检查当前角度是否超出范围
        if angle > max_angle:
            # 计算需要保留的 z 轴分量
            target_z = np.cos(np.clip(angle, 0, max_angle)) * norm_vector
            # 计算当前 z 轴分量
            current_z = vector[2]
            # 缩放因子
            scale_factor = target_z / current_z
            # 调整向量
            adjusted_vector = vector * np.array([scale_factor, scale_factor, 1])
            return adjusted_vector
        else:
            return vector

    def limit_euler_angles(self, R, roll_max=10, pitch_max=10):
        q = self._mjMdl._rotation2quat(R)
        roll, pitch, yaw = np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)), np.arcsin(
            2 * (q[0] * q[2] - q[3] * q[1])), np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                                                         1 - 2 * (q[2] ** 2 + q[3] ** 2))
        print('roll:', np.rad2deg(roll), 'pitch:', np.rad2deg(pitch), 'yaw:', np.rad2deg(yaw))
        # return R
        roll = np.clip(roll, -np.deg2rad(roll_max), np.deg2rad(roll_max))
        pitch = np.clip(pitch, -np.deg2rad(pitch_max), np.deg2rad(pitch_max))
        q = np.array([
            np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
                yaw / 2),
            np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
                yaw / 2),
            np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
                yaw / 2),
            np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
                yaw / 2)
        ])
        return self._mjMdl._quat2rot(q)

    def rot_to_euler(self, R):
        # 俯仰角 θ        若需从旋转矩阵中提取欧拉角（ZYX顺序）： ?????????
        theta = np.arcsin(-R[2, 0])
        # 处理万向节死锁（θ = ±90°）
        if np.abs(np.cos(theta)) < 1e-6:
            phi = 0.0  # 横滚角设为 0
            psi = np.arctan2(-R[0, 1], R[1, 1])  # 偏航角与横滚角耦合
        else:
            # 横滚角
            phi = np.arctan2(R[2, 1], R[2, 2])  # 横滚角
            # 偏航角
            psi = np.arctan2(R[1, 0], R[0, 0])  # 偏航角
        return np.degrees(phi), np.degrees(theta), np.degrees(psi)



class DirectionAwareController:
    def __init__(self, n_drones):
        """
        改进版方向感知控制器
        :param n_drones: 无人机数量
        """
        self.n = n_drones
        self.phi = np.linspace(0, 2 * np.pi, n_drones, endpoint=False)  # 方位角分布 [n,]
        self.Kp_base = 3.0  # 基础比例增益
        self.Kd_ratio = 0.7  # 阻尼比
        self.eta = 0.3  # 方位调制系数

        # 预计算旋转矩阵 [n, 2, 2]
        self.theta_error_last = np.zeros(self.n)
        self.R = np.array([[[np.cos(phi), -np.sin(phi)],
                            [np.sin(phi), np.cos(phi)]] for phi in self.phi])

    def compute_all_torques(self, theta_d, theta_fd, _dt):
        """
        批量计算所有无人机的XY平面扭矩 (Z轴扭矩始终为0)
        参数:
            theta_errors: 俯仰角误差数组 [n,] (rad)
            theta_dot_errors: 俯仰角速度误差数组 [n,] (rad/s)
        返回:
            torques: 扭矩数组 [n, 3] (x,y,z扭矩)，其中z分量全为0
        """
        theta_errors = theta_d - theta_fd
        theta_dot_errors = (theta_errors - self.theta_error_last) / _dt
        # 计算方位相关增益矩阵 [n, 2, 2]
        cos2_phi = np.cos(self.phi) ** 2
        sin2_phi = np.sin(self.phi) ** 2
        Kx = self.Kp_base * (1 + self.eta * cos2_phi)
        Ky = self.Kp_base * (1 + self.eta * sin2_phi)
        K = np.zeros((self.n, 2, 2))
        K[:, 0, 0] = Kx
        K[:, 1, 1] = Ky

        # 计算阻尼矩阵 [n, 2, 2]
        D = self.Kd_ratio * K

        # 构建误差向量 [n, 2]
        errors = np.column_stack([np.zeros(self.n), theta_errors])
        error_dots = np.column_stack([np.zeros(self.n), theta_dot_errors])

        # 批量计算扭矩 (使用einsum优化矩阵乘法) [n, 2]
        torques_xy = np.einsum('nij,nj->ni', K, errors) + np.einsum('nij,nj->ni', D, error_dots)

        # 转换到全局坐标系 [n, 2]
        global_torques = np.einsum('nij,nj->ni', self.R, torques_xy)
        self.theta_error_last = theta_errors

        # 添加z轴零扭矩 [n, 3]
        return np.column_stack([global_torques, np.zeros(self.n)])

    def update_parameters(self, Kp_base=None, eta=None):
        """动态更新控制参数"""
        if Kp_base is not None:
            self.Kp_base = Kp_base
        if eta is not None:
            self.eta = eta




class LoadCentricController:
    def __init__(self, n_drones, L):
        """
        基于负载坐标系的扭矩控制器
        :param n_drones: 无人机数量
        :param L: 系绳长度（单位：米）
        """
        self.n = n_drones
        self.L = L
        self.phi = np.linspace(0, 2 * np.pi, n_drones, endpoint=False)  # 方位分布 [n,]
        self.theta_error_last = np.zeros(self.n)

        # 控制参数
        self.Kp = 3.0  # 基础比例增益
        self.Kd = 0.0  # 基础微分增益
        self.tension_min = 5.0  # 最小张力(N)

    def compute_f(self, theta_d, theta_fd, _dt):
        """
        计算保持构型所需的无人机扭矩（机体系）
        参数:
            theta_errors: 俯仰角误差 [n,] (rad)
            theta_dot_errors: 俯仰角速度误差 [n,] (rad/s)
            current_theta: 当前俯仰角 [n,] (rad)

        返回:

        """
        theta_errors = theta_d - theta_fd
        theta_dot_errors = (theta_errors - self.theta_error_last) / _dt
        # 计算单位方向向量 [n,3]
        sin_theta = np.sin(theta_fd)
        cos_theta = np.cos(theta_fd)
        sin_phi = np.sin(self.phi)
        cos_phi = np.cos(self.phi)

        # u_vectors = np.column_stack([
        #     sin_theta * cos_phi,
        #     sin_theta * sin_phi,
        #     cos_theta
        # ])  # [n,3]

        # 计算切向力方向（误差修正方向）[n,3]
        # 沿球面切线方向：∂p/∂θ
        tangent_vectors = np.column_stack([
            -sin_theta * cos_phi,
            -sin_theta * sin_phi,
            cos_theta
        ])  # [n,3]

        # 计算所需力矢量（负载坐标系）[n,3]
        F_magnitudes = self.Kp * theta_errors + self.Kd * theta_dot_errors
        print(f'theta_errors: {theta_errors} theta_dot_errors: {theta_dot_errors} F_magnitudes: {F_magnitudes}')
        F_vectors = F_magnitudes[:, None] * tangent_vectors  # [n,3]
        self.theta_error_last = theta_errors

        # # 转换为机体系扭矩（假设无人机姿态水平）
        # # 对于标准多旋翼，x扭矩对应俯仰，y扭矩对应横滚
        # body_torques = np.column_stack([
        #     -F_vectors[:, 1],  # 横滚扭矩（对应y力分量）
        #     F_vectors[:, 0],  # 俯仰扭矩（对应x力分量）
        #     np.zeros(self.n)  # 偏航扭矩保持为零
        # ])
        #
        # # 张力约束处理
        # for i in range(self.n):
        #     if np.linalg.norm(F_vectors[i]) > self.tension_min:
        #         F_vectors[i] = F_vectors[i] / np.linalg.norm(F_vectors[i]) * self.tension_min
        #         body_torques[i, :2] = [-F_vectors[i, 1], F_vectors[i, 0]]

        return F_vectors

