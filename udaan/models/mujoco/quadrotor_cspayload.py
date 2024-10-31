import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.linalg import expm
import time
import copy
import enum

from ..mujoco import MujocoModel, mujoco
from .. import base
from ... import utils
from ...traj import trajectory_generator


class QuadrotorCSPayload(base.QuadrotorCSPayload):
    class MODEL_TYPE(enum.Enum):
        LINKS = 0
        TENDON = 1  # https://mujoco.readthedocs.io/en/stable/XMLreference.html#default-tendon
        CABLE = 2  # https://mujoco.readthedocs.io/en/latest/XMLreference.html?highlight=Cable#body-j-composite

    def __init__(self, **kwargs):
        if "render" in kwargs:
            self._mj_render = kwargs["render"]
        else:
            self._mj_render = False
        kwargs["render"] = False
        super().__init__(render=False)  # disable vfx in base

        # mujoco model param handling
        self._mjMdl = None
        self._mj_quad_index = None
        self._mj_payload_index = None
        self._mj_cable_index = None

        self._model_type = self.MODEL_TYPE.LINKS

        self._attitude_zoh = False
        if "attitude_zoh" in kwargs:
            self._attitude_zoh = kwargs["attitude_zoh"]
        if "model" in kwargs:
            if kwargs["model"] == "tendon":
                self._model_type = self.MODEL_TYPE.TENDON
            elif kwargs["model"] == "cable":
                self._model_type = self.MODEL_TYPE.CABLE
                utils.printc_fail(
                    "Note, cable model has not been validated yet")
            elif kwargs["model"] == "links":
                self._model_type = self.MODEL_TYPE.LINKS
                utils.printc_fail(
                    "Note, links model has not been extensively tested yet")
            else:
                raise ValueError(
                    "Invalid model type provided for mujoco quadrotor_cspayload"
                )

        # Initialize mujoco model
        # different modeling choices for cable
        # ------------------------------------
        if self._model_type == self.MODEL_TYPE.TENDON:
            """tendon model (tries to create a tendon with 0 stiffness
            and 0 damping, not great for large payload mass)
            """
            self.__init_tendon_model()
        elif self._model_type == self.MODEL_TYPE.LINKS:
            """create n rigid cylinder links with ball joints and small damping
            working well but slower to simulate
            """
            self.__init_nlink_model()
        elif self._model_type == self.MODEL_TYPE.CABLE:
            """uses mujoco2.3 composite cable model
            issue with the equality constraint between the cable and the payload
            """
            self.__init_composite_model()

        if self._input_type == base.QuadrotorCSPayload.INPUT_TYPE.CMD_PROP_FORCES:
            self._ctrl_index = 4
        else:
            self._ctrl_index = 0

        self._mjDt = 1.0 / 100.0
        self._step_iter = int(self.sim_timestep / self._mjDt)
        self._nFrames = 1
        if self._attitude_zoh:
            self._step_iter, self._nFrames = self._nFrames, self._step_iter

        # TODO make it configurable or read from xml
        self.mQ = 0.75  # kg
        self.mL = 0.15  # kg
        self.l = 1.0  # m
        if self.verbose:
            utils.printc_ok("Mujoco model loaded")
        return

    def __init_tendon_model(self):
        # 初始化 MuJoCo 模型，指定 XML 文件路径和渲染选项
        self._mjMdl = MujocoModel(model_path="quad_payload_mj.xml",
                                  render=self._mj_render)
        self._mj_quad_pos_index = 0
        self._mj_quad_quat_index = 3
        self._mj_quad_vel_index = 0
        self._mj_quad_omega_index = 3

        self._mj_payload_pos_index = 7
        self._mj_payload_quat_index = 10
        self._mj_payload_vel_index = 6
        self._mj_payload_omega_index = 9

        self._mj_cable_tendon_index = 0

        self._mj_quad_body_index = 1
        self._mj_payload_body_index = 2
        return

    def __init_nlink_model(self):
        self._mjMdl = MujocoModel(model_path="quad_flxcbl_mj.xml",
                                  render=self._mj_render)
        self._mj_quad_pos_index = 0
        self._mj_quad_quat_index = 3
        self._mj_quad_vel_index = 0
        self._mj_quad_omega_index = 3

        self._mj_payload_body_index = 27
        self._mj_data__payload_pos_prev = np.zeros(3)
        self._mj_data__payload_vel_not_initialized = False
        return

    def __init_composite_model(self):
        self._mjMdl = MujocoModel(model_path="quad_cable_mj.xml",
                                  render=self._mj_render)
        self._mj_quad_pos_index = 0
        self._mj_quad_quat_index = 3
        self._mj_quad_vel_index = 0
        self._mj_quad_omega_index = 3

        self._mj_payload_pos_index = 167
        self._mj_payload_quat_index = 170
        self._mj_payload_vel_index = 126
        self._mj_payload_omega_index = 129
        return

    @base.QuadrotorCSPayload.qrotor_mass.setter
    def qrotor_mass(self, value):
        super(QuadrotorCSPayload, self.__class__).qrotor_mass.fset(self, value)
        self._mjMdl.model.body_mass[self._mj_quad_body_index] = value  # set the quad's mass
        return

    @base.QuadrotorCSPayload.qrotor_inertia.setter
    def qrotor_inertia(self, value):
        super(QuadrotorCSPayload, self.__class__).qrotor_inertia.fset(self, value)
        if value.ndim == 2:
            value = np.diag(value)  # get the elements of the main diagonal
        self._mjMdl.model.body_inertia[self._mj_quad_body_index] = value
        return

    @base.QuadrotorCSPayload.cable_length.setter
    def cable_length(self, value):
        super(QuadrotorCSPayload, self.__class__).cable_length.fset(self, value)
        if self._model_type == self.MODEL_TYPE.TENDON:
            self._mjMdl.model.tendon_length0[self._mj_cable_tendon_index] = value  # 设置腱的初始长度
            self._mjMdl.model.tendon_limited[self._mj_cable_tendon_index] = True  # 将腱设置为有限长度
            self._mjMdl.model.tendon_range[self._mj_cable_tendon_index][1] = value  # 设置腱的长度范围
            self._mjMdl.model.tendon_lengthspring[self._mj_cable_tendon_index] = value  # 设置腱的弹簧长度
            mujoco.mj_tendon(self._mjMdl.model, self._mjMdl.data)  # 更新 MuJoCo 模型中的腱属性
        return

    @base.QuadrotorCSPayload.payload_mass.setter
    def payload_mass(self, value):
        super(QuadrotorCSPayload,
              self.__class__).payload_mass.fset(self, value)
        self._mjMdl.model.body_mass[self._mj_payload_body_index] = value
        return

    def reset(self, **kwargs):
        """reset state and time"""
        self.t = 0.0
        self._mjMdl.reset()
        super().reset(**kwargs)

        if self._model_type == self.MODEL_TYPE.TENDON:
            self.__reset_tendon_model()
        elif self._model_type == self.MODEL_TYPE.CABLE:
            self.__reset_composite_model()
        elif self._model_type == self.MODEL_TYPE.LINKS:
            self.__reset_nlink_model()

        self._query_latest_state()
        return

    def __reset_tendon_model(self):
        quat_Q = np.array([1.0, 0.0, 0.0, 0.0])  # quadrotor attitude (w, x, y, z)
        _qQ = utils.sp_rot.from_matrix(self.state.orientation).as_quat()  # 将当前姿态旋转矩阵转换为四元数
        quat_Q = np.array([_qQ[3], _qQ[0], _qQ[1], _qQ[2]])

        self._mjMdl.data.qpos[self._mj_quad_pos_index:self._mj_quad_pos_index +
                                                      3] = self.state.position
        self._mjMdl.data.qpos[self._mj_quad_quat_index:self._mj_quad_quat_index +
                                                       4] = quat_Q
        self._mjMdl.data.qvel[self._mj_quad_vel_index:self._mj_quad_vel_index +
                                                      3] = self.state.velocity
        self._mjMdl.data.qvel[self._mj_quad_omega_index:self._mj_quad_omega_index +
                                                        3] = self.state.angular_velocity
        self._mjMdl.data.qpos[self._mj_payload_pos_index:self._mj_payload_pos_index +
                                                         3] = self.state.payload_position
        self._mjMdl.data.qpos[self._mj_payload_quat_index:self._mj_payload_quat_index + 4] = np.array([
            1.0, 0.0, 0.0, 0.0])  # point mass payload, this is obsolete
        self._mjMdl.data.qvel[self._mj_payload_vel_index:self._mj_payload_vel_index +
                                                         3] = self.state.payload_velocity  # payload velocity
        self._mjMdl.data.qvel[self._mj_payload_omega_index:self._mj_payload_omega_index + 3] = np.zeros(
            3)  # payload angular velocity, always zero
        return

    def __reset_nlink_model(self):
        self._mjMdl.data.qpos[self._mj_quad_pos_index:self._mj_quad_pos_index +
                                                      3] = self.state.position
        return

    def __reset_composite_model(self):
        import warnings
        warnings.warn("Composite model not implemented yet")
        # raise NotImplementedError
        return

    def step(self, u):
        for _ in range(self._step_iter):
            # clamp input
            if self._input_type == base.QuadrotorCSPayload.INPUT_TYPE.CMD_PROP_FORCES:
                u_clamped = np.clip(u, self._prop_min_force, self._prop_max_force)
            else:
                thrust, torque = self._parse_input(u)
                u_clamped = np.append(thrust, torque)
            # set control and step simulation
            self._mjMdl.data.ctrl[self._ctrl_index: self._ctrl_index + 4] = u_clamped
            self._mjMdl._step_mujoco_simulation(self._nFrames)
            # update state required to be in the loop for attitude control
            self._query_latest_state()
            # add tracking marker
            if self._mjMdl.render:
                if (self._input_type == base.QuadrotorCSPayload.INPUT_TYPE.CMD_PROP_FORCES):
                    for i in range(4):
                        self._mjMdl.add_arrow_at(
                            self._mjMdl.data.site_xpos[i],  # position
                            self._mjMdl.data.site_xmat[i],  # rotation matrix
                            s=[0.005, 0.005, 0.25 * float(u_clamped[i])],  # scale
                            label="f%d" % i,  # label
                            color=[1.0, 1.0, 0.0, 1.0],  # color
                        )
        return

    def simulate(self, tf, r, **kwargs):
        # reset and update render(base class)
        self.reset(**kwargs)
        self.update_render()

        timevals = []
        quad_pos = []
        quad_vel = []
        quad_acc = []
        quad_q = []
        load_pos = []
        load_vel = []
        T_pred = []
        Tq = []
        Tl = []
        U = []
        v = np.random.uniform(3, 3)
        traj = trajectory_generator(r, v, tf, (self.mL + self.mQ) * 9.81)
        start_t = time.time_ns()
        # simulate
        while self._mjMdl.data.time < tf - 0.0001:
            k = int(100 * self._mjMdl.data.time)  # 计算当前时间步的索引k
            if self._input_type == QuadrotorCSPayload.INPUT_TYPE.CMD_PROP_FORCES:
                raise NotImplementedError("TODO: Implement")
                u = self._prop_controller.compute(self.t, self.state)
            else:
                u = self._payload_controller.compute(self.t, traj[k, :], self.state) # 计算控制输入
            self.step(u)  # 执行一步仿真
            timevals.append(self._mjMdl.data.time)
            quad_pos.append(self.state.position.copy())
            quad_vel.append(self.state.velocity.copy())
            quad_q.append(self.state.q.copy())
            quad_acc.append(self.state.angular_velocity.copy())
            load_pos.append(self.state.payload_position.copy())
            load_vel.append(self.state.payload_velocity.copy())
            # T_pred.append(force_pred)
            Tq.append(self._mjMdl.data.sensordata[0:3].copy())
            Tl.append(self._mjMdl.data.sensordata[3:6].copy())
            U.append(u.copy())
        data = np.hstack((np.array(quad_pos), np.array(quad_vel), np.array(quad_q),
                          np.array(quad_acc), np.array(Tl), np.array(U))) # 将仿真数据整合到一个数组中
        end_t = time.time_ns()
        time_taken = (end_t - start_t) * 1e-9
        print("Took (%.4f)s for simulating (%.4f)s" % (time_taken, self.t))

        _, ((ax1), (ax2), (ax3)) = plt.subplots(3, 1)
        ax1.plot(np.array(timevals), np.array(load_pos)[:, 0])
        ax1.plot(np.array(timevals), traj[1:, 0])
        ax1.set_title("Posx")
        ax2.plot(np.array(timevals), np.array(load_pos)[:, 1])
        ax2.plot(np.array(timevals), traj[1:, 1])
        # # ax2.plot(data[:,20])
        # ax2.plot(np.array(Tq)[:,1])
        ax2.set_title("Posy")
        ax3.plot(np.array(timevals), np.array(load_pos)[:, 2])
        ax3.plot(np.array(timevals), traj[1:, 2])
        # # ax3.plot(data[:,21])
        # ax3.plot(np.array(Tq)[:,2])
        ax3.set_title("Posz")

        _, ((ax1), (ax2), (ax3)) = plt.subplots(3, 1)
        ax1.plot(np.array(timevals), data[:, 3])
        ax1.plot(np.array(timevals), traj[1:, 0])
        ax1.set_title("Velx")
        ax2.plot(np.array(timevals), data[:, 4])
        ax2.plot(np.array(timevals), traj[1:, 1])
        ax2.set_title("Vely")
        ax3.plot(np.array(timevals), data[:, 5])
        ax3.plot(np.array(timevals), traj[1:, 2])
        ax3.set_title("Velz")
        _, ((ax1), (ax2), (ax3)) = plt.subplots(3, 1)
        ax1.plot(np.array(timevals), data[:, 10])
        # ax1.plot(np.array(timevals),traj[1:,0])
        ax1.set_title("Accx")
        ax2.plot(np.array(timevals), data[:, 11])
        # ax2.plot(np.array(timevals),traj[1:,1])
        ax2.set_title("Accy")
        ax3.plot(np.array(timevals), data[:, 12])
        # ax3.plot(np.array(timevals),traj[1:,2])
        ax3.set_title("Accz")
        _, ax = plt.subplots()
        plt.title('Reference trajectory')
        ax = plt.axes(projection="3d")
        ax.plot3D(np.array(load_pos)[:, 0], np.array(load_pos)[:, 1], np.array(load_pos)[:, 2], label='meas')
        ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2], label='ref')
        ax.set_xlabel("x[m]")
        ax.set_ylabel("y[m]")
        ax.set_zlabel("z[m]")
        _, ax = plt.subplots()
        plt.title('Reference trajectory')
        ax.plot(np.array(load_pos)[:, 0], np.array(load_pos)[:, 1], label='meas')
        ax.plot(traj[:, 0], traj[:, 1], label='ref')
        ax.set_xlabel("x[m]")
        ax.set_ylabel("y[m]")
        plt.show()
        return

    def _query_latest_state(self):
        """ NOTE: it is very important to copy the data from mujoco data"""
        # ------------------------------------------------------------
        if self._model_type == self.MODEL_TYPE.TENDON:
            self.__query_state_tendon_model()
        elif self._model_type == self.MODEL_TYPE.CABLE:
            self.__query_state_composite_model()
        elif self._model_type == self.MODEL_TYPE.LINKS:
            self.__query_state_nlink_model()

        # update quadrotor state
        # q here is quadrotor quaternion
        q = copy.deepcopy(self._mj_data__quad_quat)
        self.state.position = copy.deepcopy(self._mj_data__quad_pos)
        self.state.velocity = copy.deepcopy(self._mj_data__quad_vel)
        self.state.orientation = self._mjMdl._quat2rot(q)
        self.state.angular_velocity = copy.deepcopy(self._mj_data__quad_angvel)
        self.state.payload_position = copy.deepcopy(self._mj_data__payload_pos)
        self.state.payload_velocity = copy.deepcopy(self._mj_data__payload_vel)

        p = self.state.payload_position - self.state.position
        self.state.cable_attitude = p / np.linalg.norm(p)
        dq = self.state.payload_velocity - self.state.velocity
        self.state.cable_ang_velocity = np.cross(self.state.cable_attitude, dq)
        return

    def __query_state_tendon_model(self):
        self._mj_data__quad_pos = self._mjMdl.data.qpos[
                                  self._mj_quad_pos_index:self._mj_quad_pos_index + 3]
        self._mj_data__quad_quat = self._mjMdl.data.qpos[
                                   self._mj_quad_quat_index:self._mj_quad_quat_index + 4]
        self._mj_data__quad_vel = self._mjMdl.data.qvel[
                                  self._mj_quad_vel_index:self._mj_quad_vel_index + 3]
        self._mj_data__quad_angvel = self._mjMdl.data.qvel[
                                     self._mj_quad_omega_index:self._mj_quad_omega_index + 3]

        self._mj_data__payload_pos = self._mjMdl.data.qpos[
                                     self._mj_payload_pos_index:self._mj_payload_pos_index + 3]
        self._mj_data__payload_vel = self._mjMdl.data.qvel[
                                     self._mj_payload_vel_index:self._mj_payload_vel_index + 3]
        return

    def __query_state_composite_model(self):
        self._mj_data__quad_pos = self._mjMdl.data.qpos[
                                  self._mj_quad_pos_index:self._mj_quad_pos_index + 3]
        self._mj_data__quad_quat = self._mjMdl.data.qpos[
                                   self._mj_quad_quat_index:self._mj_quad_quat_index + 4]
        self._mj_data__quad_vel = self._mjMdl.data.qvel[
                                  self._mj_quad_vel_index:self._mj_quad_vel_index + 3]
        self._mj_data__quad_angvel = self._mjMdl.data.qvel[
                                     self._mj_quad_omega_index:self._mj_quad_omega_index + 3]

        self._mj_data__payload_pos = self._mjMdl.data.qpos[
                                     self._mj_payload_pos_index:self._mj_payload_pos_index + 3]
        self._mj_data__payload_quat = self._mjMdl.data.qpos[
                                      self._mj_payload_quat_index:self._mj_payload_quat_index + 4]
        self._mj_data__payload_vel = self._mjMdl.data.qvel[
                                     self._mj_payload_vel_index:self._mj_payload_vel_index + 3]
        self._mj_data__payload_angvel = self._mjMdl.data.qvel[
                                        self._mj_payload_omega_index:self._mj_payload_omega_index + 3]
        return

    def __query_state_nlink_model(self):
        self._mj_data__quad_pos = self._mjMdl.data.qpos[
                                  self._mj_quad_pos_index:self._mj_quad_pos_index + 3]
        self._mj_data__quad_quat = self._mjMdl.data.qpos[
                                   self._mj_quad_quat_index:self._mj_quad_quat_index + 4]
        self._mj_data__quad_vel = self._mjMdl.data.qvel[
                                  self._mj_quad_vel_index:self._mj_quad_vel_index + 3]
        self._mj_data__quad_angvel = self._mjMdl.data.qvel[
                                     self._mj_quad_omega_index:self._mj_quad_omega_index + 3]

        self._mj_data__payload_pos = self._mjMdl.data.xpos[
            self._mj_payload_body_index]
        self._mj_data__payload_vel = (
                                             self._mj_data__payload_pos -
                                             self._mj_data__payload_pos_prev) / self._mjMdl.model.opt.timestep
        if not self._mj_data__payload_vel_not_initialized:
            self._mj_data__payload_vel = np.zeros(3)
            self._mj_data__payload_vel_not_initialized = True
        self._mj_data__payload_pos_prev = self._mj_data__payload_pos
        return
