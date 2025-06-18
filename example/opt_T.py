import sys

import mujoco.viewer
import numpy as np
import time
import os
from quad_mjxml import multi_xml
from udaan.models.mujoco.multiquad_suspend import MulitQuadCS, LoadCentricController
import matplotlib.pyplot as plt
from ESO import ExtendedStateObserver
import opt_Tension as oT
import traj_generator as tra
from plot_utility import *



class ParaStruct:
    def __init__(self):
        """ 定义参数 """
        self.m_quad = 0.75
        self.diaginertia = [0.0053, 0.0049, 0.0098]
        self.m_load = 1
        self.diaginertia_load = [0.00015, 0.00015, 0.00015]
        self.ctrlrange_dict = {'thrust': [0.0, 40.0], 'Mx': [-3.0, 3.0], 'My': [-3.0, 3.0], 'Mz': [-3.0, 3.0]}
        self.quad_poso = [  # [0.0, 0.0, 2.0]
            [0.6, 0.0, 0.8], [0.0, 0.6, 0.8], [-0.6, 0.0, 0.8], [0.0, -0.6, 0.8]
            # [0.4223, 0.4223, 0.8],[-0.4223, 0.4223, 0.8],[-0.4223, -0.4223, 0.8],[0.4223, -0.4223, 0.8]
            # [-0.6, 0.0, 1.8], [0.6, 0.0, 1.8]
        ]
        self.n = np.size(self.quad_poso, 0)
        self.load_pos0 = [0.0, 0.0, 0.0]
        self.l = 1.5
        self.dt = 0.02
        self.A = 0.4 * np.array([[0, 0, 1, 0],
                           [0, 0, 0, 1],
                           [1, 0, 0, 0],
                           [0, 1, 0, 0]])
        # self.A = np.zeros((self.n, self.n))  0.4
        self.B = 0.6 * np.array([1] * self.n)  #TODO: 协同误差还是单体误差  0.5

        self.posKp = np.array([4.6, 4.6, 18.1])  #2.6 2.6 6.1
        self.posKd = np.array([5, 5, 8.9])  # 4, 4, 5.9
        self.posKi = np.array([0.08, 0.08, 0.08])

        self.so3Kp1 = np.array([2.1, 2.1, 2.1])
        self.so3Kd1 = np.array([1.7, 1.7, 1.3])
        self.so3Kp2 = np.diag(self.diaginertia) @ [0.3, 0.3, 0.3]
        self.so3Ki2 = np.diag(self.diaginertia) @ [0.01, 0.01, 0.01]


para = ParaStruct()
waypoints = np.array([
    # para.load_pos0,
    # [0, 0, 2],[1, 0, 2],[3, 2, 3],[3, 4, 3],
    [0, 0, 2],[0.5, 0, 2],[1, 0, 2],[3, 0, 2],[3, 2, 2],[3, 4, 2]
])
# 创建轨迹生成器
traj_gen = tra.MinimumSnapTrajectory(waypoints, poly_order=5, dim=3, ave_v=[1,1,1])
traj_gen.coeffs = np.load('traj3.npy')
# traj_gen.plot_trajectory()
# plt.show()
# 使用示例
model_path = f'./generate{para.n}.xml'
mdl = MulitQuadCS(para, num_quad=para.n, model_xml_path=model_path,
                  render=False)
sim_T = 12
if mdl.render:
    start_t = time.time_ns()
    cam = mdl._mjMdl.viewer.cam
    cam.distance = 15

mdl._query_latest_state()  # 更新初始状态
np.set_printoptions(suppress=True)  # ban e
step = 0

store_u_list = np.zeros((para.n, 4)) # 保存控制输入的二维矩阵
store_ang_list = np.zeros((para.n, 6)) # 保存姿态角(feedback+desired)的二维矩阵
store_pos_list = np.array(para.quad_poso).reshape(-1, 3)  # 保存实际位置序列的二维矩阵
store_vel_list = 0 * np.array(para.quad_poso).reshape(-1, 3)  # 保存实际速度序列的二维矩阵
store_acc_list = 0 * np.array(para.quad_poso).reshape(-1, 3)  # 保存实际加速度序列的二维矩阵
store_posd_list = np.array(para.quad_poso).reshape(-1, 3)  # 保存期望位置序列的二维矩阵
store_error_list = np.zeros_like(para.quad_poso).reshape(-1, 3)  # 保存位置误差的二维矩阵
store_tension_vec_list = np.zeros_like(para.quad_poso).reshape(-1, 3)  # 保存拉力向量的二维矩阵
store_load_list = np.array(para.load_pos0).reshape(-1, 3)  # 保存载荷位置序列的二维矩阵
store_loadvel_list =  0 * np.array(para.load_pos0).reshape(-1, 3)  # 保存载荷速度序列的二维矩阵
store_loadacc_list = 0 * np.array(para.load_pos0).reshape(-1, 3) # 保存载荷加速度序列的二维矩阵
store_t_force_list = [0] * para.n  # 保存拉力模长的列表
store_theta_list = np.zeros((1, para.n))  # 保存俯仰角度的列表
store_theta_des_list = np.zeros((1, para.n))  # 保存期望俯仰角度的列表
store_phi_list= np.zeros((1, para.n))  # 保存偏航角度的列表
store_norm_tension_list = np.zeros((1, para.n))
store_t_list = [0]

import MPC_base as _mpc
loadctl = _mpc.mpcPosctl(8,0.2,
          Q=np.diag([50, 50, 50, 1, 1, 1]),
          R=np.diag([0.5, 0.5, 1.5]),
          S=np.diag([6.3, 6.3, 2.3]))
# loadctl = _mpc.mpcPosctlInc(8,0.2,
#           Q=np.diag([50, 50, 50, 1, 1, 1]),
#           # R=np.diag([0.5, 0.5, 1.5]),
#           S=np.diag([6.3, 6.3, 2.3]))
# last_mpc_u = [0,0, 9.8]
thetactl = LoadCentricController(para.n, para.l)
mdl._query_latest_state()

"""" 仿真步进部分  """
while mdl.t < sim_T:
    step = step + 1
    t_start = 1.5
    if mdl.t < t_start:  # 起飞阶段
        t_tra = 0
    else:
        t_tra = mdl.t - t_start
    pos_temp = [np.array(mdl.state.quads[j].position) for j in range(para.n)]  # 获取四轴的位置储存为矩阵形式
    pos_mat = np.vstack(pos_temp)  # 将四轴位置矩阵的列表纵向堆叠
    vel_temp = [np.array(mdl.state.quads[j].velocity) for j in range(para.n)]  # 获取四轴的位置储存为矩阵形式
    vel_mat = np.vstack(vel_temp)  # 将四轴速度矩阵的列表纵向堆叠
    # pos_d = generate_pos(para.n, para.quad_poso, mdl.t)
    # pos_d = para.quad_poso + np.vstack([[0, 2.0, 0.0]] * para.n)  # 根据时间计算期望位置
    getx_load = lambda t: 0 + 0.5 * (t ** 2) * (t > 1 and t < 3) + 4.5 * (t >= 3)
    gety_load = lambda t: 0 + 0.5 * (t ** 2) * (t > 1 and t < 3) + 4.5 * (t >= 3)
    pos_load_d = np.array(traj_gen.cal_trajectory_order(t_tra, 0))  # 根据时间计算期望载荷位置
    # pos_load_d = generate_pos(1, [para.load_pos0], mdl.t).reshape(-1)
    # pos_load_d = para.load_pos0 + np.array([getx_load(mdl.t), gety_load(mdl.t), 0.0])
    # print(_f"load pos_d: {pos_load_d}")

    T = loadctl._N
    # getacc = lambda tt: [[1.0, 1.0, 0] * (t > 1 and t < 3) + [0.0, .0, .0] * (t <= 1 or t >= 3) for t in tt]
    # acc_L = np.array(getacc(np.linspace(mdl.t, mdl.t + (T - 1) * para.dt, T)))
    load_pos_est = oT.estimate_payload_position(pos_temp, para.l) # 估计载荷位置
    loadRef = _mpc.generate_pos_vel_mpc(t_tra, T, loadctl._dt, load_pos_est, mdl.state.load_vel,
                                        lambda t : traj_gen.cal_trajectory_order(t, 1),
                                        lambda t : traj_gen.cal_trajectory_order(t, 0))
    exp_acc = loadctl.solve(ref_states=loadRef.flatten())

    # # 计算时间序列
    # times = np.linspace(t_tra,
    #                     t_tra + (T - 1) * para.dt,
    #                     T)
    # # 用 traj_gen 输出标称加速度作为期望加速度
    # exp_acc = np.array(traj_gen.cal_trajectory_order(times, 2)) + np.array([0, 0, 9.8])  # 加上重力加速度

    print(f'exp acc: {exp_acc[0:]}')
    print(f' current load pos: {mdl.state.load_pos} vel: {mdl.state.load_vel} acc: {mdl.state.load_acc}')
    print(f' reference load pose: {loadRef[1, 0:3]} vel: {loadRef[1, 3:6]}')

    # acc_L = traj_gen.cal_trajectory_order(np.linspace(mdl.t, mdl.t + (T - 1) * para.dt, T), 2)
    acc_L = exp_acc
    # tmp_acc = exp_acc - np.array([0,0, 9.8])
    # acc_L = oT.limit_vector_magnitude(tmp_acc, 5.0)
    # acc_L = acc_L + np.array([0,0, 9.8])
    # print(f"acc_L: {acc_L[0]} ori:{exp_acc[0]} ")
    # acc_L = exp_acc  # 不进行限幅 在mpc内约束

    theta_sequen, phi =oT.global_optimize_theta_sequence(para.n, acc_L, para.m_load)
    # pos_load_d = np.array([0,0,3])
    if mdl.t < t_start:
        theta_d = np.deg2rad([70, 70, 70, 70])  # 起飞时先固定队形
    else:
        theta_d = theta_sequen[0]
    # pos_load_d =(mdl.t<3) * np.array([0, 0, 2.5]) +   (mdl.t>=3) * np.array([3, 3, 2.5])# todo: test 飞一个点
    # theta_d = np.deg2rad([70, 70, 70, 70])  # todo: test 先保持固定队形测试
    store_theta_des_list = np.append(store_theta_des_list, theta_d.reshape(-1, 4), axis=0)
    theta_cur, phi_cu = oT.Cart2Spher_qunit([mdl.state.cables[j].q for j in range(para.n)]) #mdl.state.load_pos_est)
    store_theta_list = np.append(store_theta_list, theta_cur.reshape(-1, 4), axis=0)
    # theta, phi = oT.update_opt_phi_theta(para.n, acc_L, para.m_load)

    # pos_d = pos_load_d + [0,0, para.l] # single quad
    pos_d = oT.cal_quadpose(pos_load_d, theta_d, phi, para.n, para.l)
    # alpha = np.clip(np.linalg.norm(theta_d - theta_cur, 1) % 1.4,0,0.5)
    # pos_load_filered = (1.0 - alpha) * pos_load_d +  (alpha)* load_pos_est
    pos_load_filered = pos_load_d
    formation_pos_error = oT.calError_Form(para.A, para.B, pos_load_filered, pos_mat, theta_d, phi, para.l,
                                           para.n)  # 根据载荷位置计算编队协同位置误差
    formation_vel_error = oT.caldError_Form(para.A, para.B, vel_mat)  # 计算编队协同速度误差
    formation_error = np.vstack([formation_pos_error, formation_vel_error])  # 将位置误差和速度误差纵向堆叠

    store_posd_list = np.append(store_posd_list, np.array(pos_d).reshape(-1, 3), axis=0)  # 保存期望位置
    for j in range(para.n):
        store_pos_list = np.append(store_pos_list, mdl.state.quads[j].position.reshape(-1, 3), axis=0)  # 保存实际位置
        store_vel_list = np.append(store_vel_list, mdl.state.quads[j].velocity.reshape(-1, 3), axis=0)  # 保存实际速度
        store_acc_list  = np.append(store_acc_list, mdl.state.quads[j].acc.reshape(-1, 3), axis=0)  # 保存实际速度
        # 保存张力  *-1 得到无人机受到的张力方向  error
        store_tension_vec_list = np.append(store_tension_vec_list, -1 * mdl.state.cables[j].tension.reshape(-1, 3),
                                           axis=0)
        store_t_force_list.append(mdl.state.cables[j].t_froce)
    store_t_list = np.append(store_t_list, mdl.t)
    store_load_list = np.append(store_load_list, mdl.state.load_pos.reshape(-1, 3), axis=0)  # 保存载荷位置
    store_loadvel_list = np.append(store_loadvel_list, mdl.state.load_vel.reshape(-1, 3), axis=0)  # 保存载荷速度
    store_loadacc_list = np.append(store_loadacc_list, mdl.state.load_acc.reshape(-1, 3), axis=0)  # 保存载荷加速度

    D = oT.compute_D(theta_d.reshape(-1, 4), phi)
    acc_cal = (store_loadvel_list[-1] -  store_loadvel_list[-2]) / para.dt
    store_norm_tension = oT.phi_theta2_tension(D, acc_cal + np.array([0,0,9.8]), 9.81, para.m_load)  # 计算差分得到的载荷加速度对应的标称拉力
    # last_mpc_u = acc_cal + np.array([0,0,9.8])
    store_norm_tension_list = np.append(store_norm_tension_list, store_norm_tension.reshape(-1, 4), axis=0)
    # store_error_list = np.append(store_error_list, formation_error[0:para.n:], axis=0)

    ################ control loop   ################
    # f_theta = thetactl.compute_f(theta_d, theta_cur, 0.02)  # 计算俯仰角补偿力

    t_margin = 0.3 * para.m_load * oT.cal_quadpose([0, 0, 0], theta_d, phi, para.n, 1)  # 期望构型的单位向量D
    t_compensite = np.array([para.m_quad, para.m_quad, 0]) * np.tile(acc_L[0], (para.n, 1))

    thrust_vec = mdl.quad_position_control(formation_error, 0*t_compensite, fromationUsed=True,
                                           kp=para.posKp, kd=para.posKd, ki=para.posKi)  # 计算期望位置控制得到的推力
    # thrust_vec += f_theta.reshape(-1)  #TODO 加入俯仰角补偿力
    # thrust_vec = mdl.quad_position_control([pos_d], 0*t_margin.reshape(-1), fromationUsed=False)  # 单无人机测试

    u, angle_vec = mdl.compute_attitude_control_cascade(thrust_vec)  # 计算姿态控制得到升力和扭矩
    # print(f"angle_vec(deg): {angle_vec}")
    mdl.step_data(u)  # MulitQuadCS类里面自定义的步进函数
    print(f"step {step}, time {mdl.t} \n")

    store_u_list = np.append(store_u_list, u.reshape(-1, 4), axis=0)  # 保存实际位置
    store_ang_list = np.append(store_ang_list, angle_vec.reshape(-1, 6), axis=0)  # 保存姿态角信息
"""" 绘制图像部分  """
# time_steps = np.arange(np.size(store_posd_list, 0))
# 绘制拉力变化曲线  ori-0.398
figs_tension = plot_tension_curve(store_t_list, store_t_force_list, store_tension_vec_list,
                                  mdl.eso, para, detail=False)
# 绘制标称拉力曲线
plot_normtension_curve(store_t_list, store_norm_tension_list, para)

# 绘制控制输入对应的能耗曲线
# energies, P_total_multi = calculate_power_and_energy_multi_drones(store_u_list, para.n, len(store_t_list), 0.02)
# plot_power_and_energy_curves(store_t_list, P_total_multi,0.02)
# 创建俯仰角度和期望俯仰角度的图形
fig0 = plot_theta_curve(store_t_list, store_theta_list, store_theta_des_list, para)

# 绘制载荷pva
fig_load = plot_load_curve(store_t_list,store_load_list,store_loadvel_list,store_loadacc_list)
# 绘制三维轨迹
# fig3d, ax3d = plot_3d_trajectory(store_posd_list, store_pos_list,store_load_list, para)

# 绘制多项式轨迹以及实际载荷轨迹
# ax_load = traj_gen.plot_trajectory()
# ax_load.plot(store_load_list[:, 0], store_load_list[:, 1], store_load_list[:, 2], label='pos_load')
# ax_load.legend()
# ax_load.set_zlim(0, 3)
# ax_load.grid(True)

# 绘制位置变化曲线
# figs_pos = plot_position_curves(store_t_list, store_posd_list, store_pos_list,para)
figs_pos = plot_position_curves(store_t_list, store_vel_list, store_acc_list, para)

# 绘制姿态角度变化曲线
figs_angle = plot_angle_curves(store_t_list, store_ang_list, para)

# 绘制误差变化曲线
# error_figs = plot_error_curves(store_t_list, store_error_list, para)
plt.show()

from scipy.io import savemat
from datetime import datetime
save_data = input("是否需要保存数据？(y/n): ").strip().lower() # 提问是否需要保存数据
if save_data == 'y':
    # 使用当前时间生成文件名
    file_name = datetime.now().strftime("UAV_%m%d-%H-%M.mat")
    # 保存数据
    savemat(file_name, {
        'pos_uav': store_pos_list,
        'vel_uav': store_vel_list,
        'acc_uav': store_acc_list,
        'vel_load': store_loadvel_list,
        'pos_load': store_load_list,
        'acc_load': store_loadacc_list,
        'tension_vel': store_tension_vec_list,
        't': store_t_list,
        'theta': store_theta_list,
        'theta_des': store_theta_des_list,
        'u': store_u_list,
        'ang': store_ang_list
    })
    print(f"数据已保存到文件: {file_name}")


