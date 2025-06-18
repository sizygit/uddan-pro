import time

import casadi as ca
import casadi.tools as ca_tools

import numpy as np


class mpcPosctl():
    """
    Nonlinear MPC for pos control(casadi library)
    """

    def __init__(self, N, dt, Q = None, R = None, S = None):
        """
        Nonlinear MPC for quadrotor control
        """
        # Time constant
        # self._T = T
        self._dt = dt
        # self._N = int(self._T / self._dt)  # control horizon
        self._N = N  # control horizon
        # Gravity
        self._gz = 9.81
        # Quadrotor constant
        self._w_max_yaw = 6.0
        self._w_max_xy = 6.0
        self._thrust_min = 2.0
        self._thrust_max = 20.0
        self._max_acc = 2  # max acceleration（姿态跟踪器跟踪效果较低，因此该值会比真实的最大加速度还大）
        # state dimension (px, py, pz, vx, vy, vz) #osition and linear velocity
        self._s_dim = 6
        # action dimensions (exp_acc)
        self._u_dim = 3
        # cost matrix for tracking the goal point
        if Q is None:
            self._Q_state = np.diag([
                100, 100, 100,  # delta_x, delta_y, delta_z
                10, 10, 10])
        else:
            self._Q_state = Q
        if R is None:
            # cost matrix for the action
            self._Q_u = np.diag([0.1, 0.1, 0.1])
        else:
            self._Q_u = R
        # 控制量变化惩罚的权重矩阵
        if S is None:
            self._S = np.diag([0.3, 0.3, 0.3])  # 示例值，按需调整
        else:
            self._S = S
        self._delta_u_max = [1.0 * dt for _ in range(self._u_dim)]

        # cost matrix for tracking the pendulum motion
        # self._Q_pen = np.diag([
        #     0, 100, 100,  # delta_x, delta_y, delta_z
        #     0, 10, 10])  # delta_vx, delta_vy, delta_vz
        self._initDynamics()

    def _initDynamics(self):
        # # # # # # # # # # # # # # # # # # #
        # ---------- Input States -----------
        # # # # # # # # # # # # # # # # # # #
        px, py, pz = ca.SX.sym('px'), ca.SX.sym('py'), ca.SX.sym('pz')
        vx, vy, vz = ca.SX.sym('vx'), ca.SX.sym('vy'), ca.SX.sym('vz')
        # -- conctenated vector
        self._x = ca.vertcat(px, py, pz, vx, vy, vz)
        # # # # # # # # # # # # # # # # # # #
        # --------- Control Command ------------
        # # # # # # # # # # # # # # # # # # #
        self._disturbance = ca.SX.sym('disturbance', 3)
        ux, uy, uz = ca.SX.sym('ux'), ca.SX.sym('uy'), ca.SX.sym('uz')
        self._u = ca.vertcat(ux, uy, uz)
        # # # # # # # # # # # # # # # # # # #
        # --------- System Dynamics ---------
        # # # # # # # # # # # # # # # # # # #
        x_dot = ca.vertcat(vx, vy, vz, ux+self._disturbance[0],
                           uy+self._disturbance[0],
                           uz - self._gz + self._disturbance[2])
        self._f = ca.Function('f', [self._x, self._u, self._disturbance], [x_dot], ['x', 'u', 'd'], ['x_dot'])
        # # RK4-integration for xdot
        F = self.sys_dynamics(self._dt)  # create F's map
        fMap = F.map(self._N)  #TODO  parallel

        # # # # # # # # # # # # # # #
        # ---- loss function --------
        # # # # # # # # # # # # # # #
        # placeholder for the quadratic cost function
        Delta_s = ca.SX.sym("Delta_s", self._s_dim)
        Delta_p = ca.SX.sym("Delta_p", self._s_dim)
        Delta_u = ca.SX.sym("Delta_u", self._u_dim)

        #
        cost_s = Delta_s.T @ self._Q_state @ Delta_s
        # cost_gap = Delta_p.T @ self._Q_pen @ Delta_p
        cost_u = Delta_u.T @ self._Q_u @ Delta_u
        f_cost_goal = ca.Function('cost_goal', [Delta_s], [cost_s])
        # f_cost_gap = ca.Function('cost_gap', [Delta_p], [cost_gap])
        f_cost_u = ca.Function('cost_u', [Delta_u], [cost_u])
        # 新增：定义控制量变化的成本函数
        Delta_u_change = ca.SX.sym("Delta_u_change", self._u_dim)
        cost_u_change = Delta_u_change.T @ self._S @ Delta_u_change
        f_cost_u_change = ca.Function('cost_u_change', [Delta_u_change], [cost_u_change])

        #
        # # # # # # # # # # # # # # # # # # # #
        # # ---- Non-linear Optimization single-shooting-----
        # # # # # # # # # # # # # # # # # # # #
        #  P:Current state and Expected trajectory state
        P = ca.SX.sym("P", self._s_dim * (self._N + 1) + self._u_dim)
        X = ca.SX.sym("X", self._s_dim, self._N + 1)
        U = ca.SX.sym("U", self._u_dim, self._N)
        self._disturbance = P[-self._u_dim:] # measured disturbance
        X[:, 0] = P[:self._s_dim]  # initial condition
        for j in range(self._N):
            # X[:, j+1] = F(X[:, j], U[:, j] + [0, 0, self._gz], self._disturbance)  # RK4-integration
            X[:, j + 1] = F(X[:, j], U[:, j], self._disturbance)  # RK4-integration
        # X[:, 1:] =fMap(X[:, :self._N], U)  map是并行计算的，会损失每一列之间的逻辑关系

        self.nlp_x = ca.reshape(U, -1, 1)  # nlp variables
        self.nlp_x0 = np.zeros([self._u_dim * self._N, 1])  # initial guess of nlp variables
        self.lbw = []  # lower bound of the variables, lbw <= nlp_x
        self.ubw = []  # upper bound of the variables, nlp_x <= ubw
        #
        self.mpc_obj = 0  # objective
        self.nlp_g = []  # constraint functions
        self.lbg = []  # lower bound of constrait functions, lbg < g
        self.ubg = []  # upper bound of constrait functions, g < ubg

        u_min = [-1*self._max_acc, -1*self._max_acc , -1*self._max_acc + self._gz]  * self._N
        u_max = [ self._max_acc, self._max_acc , 1.2*self._max_acc+ self._gz] * self._N
        # x_min = [-ca.inf for _ in range(self._s_dim)]
        # x_max = [+ca.inf for _ in range(self._s_dim)]
        # g_min = [0 for _ in range(self._s_dim)]
        # g_max = [0 for _ in range(self._s_dim)]
        #

        # "Lift" initial conditions
        # self.nlp_x += [X[:, 0]]
        # self.nlp_x0 += self._quad_s0
        self.lbw += u_min
        self.ubw += u_max
        # self.lbw += x_min
        # self.ubw += x_max

        # # starting point.
        # self.nlp_g += [X[:, 0] - P[0:self._s_dim]]
        # self.lbg += g_min
        # self.ubg += g_max

        for k in range(self._N):
            # cost for tracking the goal position
            delta_s_k = (X[:, k + 1] - P[self._s_dim * (k+1) : self._s_dim * (k + 2)])
            cost_state_k = f_cost_goal(delta_s_k)
            delta_u_k = U[:, k] - [0, 0, self._gz]
            cost_u_k = f_cost_u(delta_u_k)
            self.mpc_obj += cost_state_k + cost_u_k #+ cost_gap_k
            # 原有的Δu成本项可保留（用于平滑控制量）
            if k < self._N - 1:
                delta_u_change = U[:, k + 1] - U[:, k]
                cost_change = f_cost_u_change(delta_u_change)
                self.mpc_obj += cost_change
            # Add equality constraint
            # self.nlp_g += [X_next[:, k] - X[:, k + 1]]
            # self.lbg += g_min
            # self.ubg += g_max

        for k in range(self._N):
            # ... 其他成本项计算 ...
            if k < self._N - 1:
                delta_u_change = U[:, k + 1] - U[:, k]
                # 添加Δu的硬约束：-delta_u_max ≤ Δu ≤ delta_u_max
                for i in range(self._u_dim):
                    self.nlp_g.append(delta_u_change[i])
                    self.lbg.append(-self._delta_u_max[i])  # 下界
                    self.ubg.append(self._delta_u_max[i])  # 上界

        # nlp objective
        nlp_dict = {'f': self.mpc_obj,
                    'x': self.nlp_x,
                    'p': P,
                    'g': ca.vertcat(*self.nlp_g)}

        # # # # # # # # # # # # # # # # # # #
        # -- ipopt
        # # # # # # # # # # # # # # # # # # #
        ipopt_options = {
            'verbose': False,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0,
            "print_time": False
        }
        self.solver = ca.nlpsol("solver", "ipopt", nlp_dict, ipopt_options)
        # # jit (just-in-time compilation)
        # print("Generating shared library........")
        # cname = self.solver.generate_dependencies("mpc_v1.c")
        # system('gcc -fPIC -shared -O3 ' + cname + ' -o ' + self.so_path) # -O3

        # # # reload compiled mpc
        # print(self.so_path)
        # self.solver = ca.nlpsol("solver", "ipopt", self.so_path, ipopt_options)

    def solve(self, ref_states, _disturbance=None):
        # # # # # # # # # # # # # # # #
        # -------- solve NLP ---------
        # # # # # # # # # # # # # # # #
        #
        if _disturbance is None:
            disturbance = np.zeros(self._u_dim)
        else:
            disturbance = np.array(_disturbance)
        p_ = np.hstack((ref_states, disturbance))
        start_time = time.time()
        self.sol = self.solver(
            x0=self.nlp_x0,
            lbx=self.lbw,
            ubx=self.ubw,
            p=p_,
            lbg=self.lbg,
            ubg=self.ubg)
        #
        sol_x0 = self.sol['x'].full()  # full() get a array
        opt_u = sol_x0.reshape(-1, self._u_dim) # row vector is a input
        # Warm initialization
        self.nlp_x0 = sol_x0
        # x0_array = np.reshape(sol_x0[:-self._s_dim], newshape=(-1, self._s_dim + self._u_dim))
        print(f"Time elapsed: {time.time() - start_time} opt u(exp acc): {opt_u[0]}", )
        # print(f"distrubance: {disturbance}")
        # return optimal action
        return opt_u

    def sys_dynamics(self, dt):
        """step Discrete dynamics of the quadrotor with RK4"""
        M = 4  # refinement
        DT = dt / M
        X0 = ca.SX.sym("X", self._s_dim)
        U = ca.SX.sym("U", self._u_dim)
        D = ca.SX.sym("U", self._u_dim)
        # #
        X = X0
        for _ in range(M):
            # --------- RK4------------
            k1 = DT * self._f(X, U, D)
            k2 = DT * self._f(X + 0.5 * k1, U, D)
            k3 = DT * self._f(X + 0.5 * k2, U, D)
            k4 = DT * self._f(X + k3, U, D)
            #
            X = X + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            # Fold
        F = ca.Function('F', [X0, U, D], [X], ['x','u', 'd'],['x_next'])  # get a ca function
        return F


class mpcPosctlInc:
    """
    增量式非线性MPC:
    决策变量为 Δu 序列，uₖ = uₖ₋₁ + Δuₖ
    """

    def __init__(self, N, dt, Q=None, R=None, S=None):
        self._N = N
        self._dt = dt
        self._gz = 9.8
        self._s_dim = 6
        self._u_dim = 3
        # 状态&动作权重
        self._Q_state = Q if Q is not None else np.diag([100, 100, 100,10, 10, 10])
        # self._Q_u = R if R is not None else np.diag([0.1, 0.1, 0.1])
        # Δu 权重
        self._S = S if S is not None else np.diag([0.3, 0.3, 0.3])
        # Δu 上下界
        max_du = 13.0*dt
        lb = [-max_du,-max_du,-max_du]*self._N
        ub = [ max_du,max_du,max_du]*self._N
        self.lbw, self.ubw = lb, ub
        self.nlp_g  = []
        self._initDynamics()

    def _initDynamics(self):
        # 符号变量
        px,py,pz,vx,vy,vz = ca.SX.sym('px'), ca.SX.sym('py'), ca.SX.sym('pz'),ca.SX.sym('vx'), ca.SX.sym('vy'), ca.SX.sym('vz')
        self._x = ca.vertcat(px,py,pz,vx,vy,vz)
        ux,uy,uz = ca.SX.sym('ux'),ca.SX.sym('uy'),ca.SX.sym('uz')
        self._u = ca.vertcat(ux,uy,uz)
        d = ca.SX.sym('d',3)
        # 连续动力学
        x_dot = ca.vertcat(vx,vy,vz,
                           ux + d[0],
                           uy + d[1],
                           uz - self._gz + d[2])
        f = ca.Function('f',[self._x,self._u,d],[x_dot])
        # RK4 一步离散
        def _rk4(dt):
            X0 = ca.SX.sym('X0',6)
            U0 = ca.SX.sym('U0',3)
            D0 = ca.SX.sym('D0',3)
            M=4; DT=dt/M; X=X0
            for _ in range(M):
                k1 = DT * f(X,U0,D0)
                k2 = DT * f(X+0.5*k1,U0,D0)
                k3 = DT * f(X+0.5*k2,U0,D0)
                k4 = DT * f(X+k3,U0,D0)
                X = X + (k1+2*k2+2*k3+k4)/6
            return ca.Function('F',[X0,U0,D0],[X])
        F = _rk4(self._dt)

        # 定义参数向量 P = [x0,(N)个参考,state, last_u₋₁, disturbance]
        # 改为 N 步参考
        total_ref = self._s_dim * self._N
        P = ca.SX.sym('P', self._s_dim + total_ref + self._u_dim + self._u_dim)

        x0 = P[0: self._s_dim]
        ref_all = ca.reshape(
            P[self._s_dim: self._s_dim + total_ref],
            self._s_dim,
            self._N
        ) # (6 * N)
        # 提取 last_u 和 disturbance
        last_u = P[self._s_dim + total_ref: self._s_dim + total_ref + self._u_dim]
        disturbance = P[self._s_dim + total_ref + self._u_dim:]

        # 决策变量 ΔU
        dU = ca.SX.sym('dU', self._u_dim, self._N)
        self.nlp_x = ca.reshape(dU, -1, 1)

        # 重构 U 序列 & 状态轨迹 X
        U_seq = []
        u_prev = last_u
        for k in range(self._N):
            u_k = u_prev + dU[:,k]
            U_seq.append(u_k)
            u_prev = u_k

        # X = ca.SX(self._s_dim, self._N+1)
        # X[:,0] = x0
        X_list = [x0]
        for k in range(self._N):
            x_next = F(X_list[-1], U_seq[k], disturbance)
            X_list.append(x_next)

        X = ca.hcat(X_list)

        # 成本只计算 1..N 因为索引0对应的初始状态x0
        Delta_s = X[:,1:] - ref_all
        cost = 0
        for k in range(self._N):
            cost += Delta_s[:,k].T @ self._Q_state @ Delta_s[:,k]
            # cost += (U_seq[k] - ca.vertcat(0,0,self._gz)).T @ self._Q_u @ (U_seq[k] - ca.vertcat(0,0,self._gz))
            cost += dU[:,k].T @ self._S @ dU[:,k]

        nlp = {'x': self.nlp_x, 'p': P, 'f': cost}
        self.solver = ca.nlpsol('solver','ipopt',nlp,{'ipopt.print_level':0,'print_time':False})

    def solve(self, ref_states, last_u, disturbance=None):
        last_u = np.array(last_u).reshape(self._u_dim, 1)
        if disturbance is None:
            disturbance = np.zeros(self._u_dim)
        # 只传 N 步参考
        p_vec = np.hstack((ref_states.flatten(), last_u.flatten(), disturbance))

        x0 = np.zeros((self._u_dim * self._N, 1))
        start_time = time.time()
        sol = self.solver(x0=x0, lbx=self.lbw, ubx=self.ubw, p=p_vec)

        du_opt = sol['x'].full()
        # u_seq = np.hstack((last_u, du_opt))
        u_seq = np.vstack((last_u.reshape(-1, self._u_dim), du_opt.reshape(-1, self._u_dim)))
        u_opt = np.cumsum(u_seq, axis=0)[1:,:] # 每一行均为一个步长的控制量 （N，u_dim）
        print(f"Time elapsed: {time.time() - start_time} opt u(exp acc): {u_opt[0]}", )
        return u_opt



def generate_pos_vel_mpc(t, N, dt, cur_p, cur_v, t2vel, t2pos):
    """  current state + ref state"""
    t_seq = np.linspace(t, t + (N - 1) * dt, N)
    pv_list = []
    pv_list.append(np.hstack((cur_p, cur_v)))
    # getxy_load = lambda t: 0 + 0.5 * (t ** 2) * ((t > 1) & (t < 3)) + 4.5 * (t >= 3)
    # getxy_load = lambda t: 0
    for ti in t_seq:
        p_goal = t2pos(ti)
        v_goal = t2vel(ti)
        pv_list.append([p_goal[0], p_goal[1], p_goal[2], v_goal[0], v_goal[1], v_goal[2]])
    return np.array(pv_list)  # size(N+1,6)


if  __name__ == "__main__":
    # s= mpcPosctl(10, 0.1, S=np.diag([0, 0, 0]))
    # s.solve([0,0,0,0,0,0] + [1,1,1,0,0,0] * 10)
    # s.solve([0, 0, 0, 0, 0, 0] + [2, 2, 2, 0, 0, 0] * 10)

    N = 5
    dt = 1

    temp = mpcPosctlInc(N, dt, S=np.diag([0, 0, 0]))
    # ref = np.array([ 0.75604862 ,-0.01457808 , 1.98391634 , 0.48314085 ,-0.02969889  ,0.0,
    #         2.4832781  ,-0.06157046 , 2.         , 1.18851701 , 0.02818885 , 0.,
    #         2.91289846 ,-0.02129768 , 2.         , 0.91992865 , 0.18702312 , 0.])
    x0 = [0 ,0 , 0 ,    0  ,0  ,0.0]
    goal = [
        # 1  , 0 , 2.         , 1.0 , 0.0 , 0.,
        #     1.5 , 0 , 2.         , 0 , 0.0 , 0.,
            2 , 2 , 2.         , 0 , 0.0 , 0.
            ]
    ref = np.array([x0+goal*N])
    u_seq = temp.solve(ref, [0, 0, 9.8])
    print(u_seq)



    import matplotlib.pyplot as plt
    # 仿真 forward 模拟
    x = x0.copy()
    X_list = [x.copy()]
    for u in u_seq:
        a = u - np.array([0, 0, 9.8])
        x_dot = np.zeros(6)
        x_dot[0:3] = x[3:6]
        x_dot[3:6] = a
        x += x_dot * dt
        X_list.append(x.copy())

    X_array = np.array(X_list)

    # 绘图
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X_array[:,0], X_array[:,1], X_array[:,2], 'b.-', label='Trajectory')
    ax.scatter(goal[0], goal[1], goal[2], c='r', label='Target')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(); plt.title('Incremental MPC Trajectory'); plt.tight_layout(); plt.show()
