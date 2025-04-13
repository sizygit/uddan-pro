import numpy as np

"""  二阶系统的扩展状态观测器  x1=P x2=V"""
""" x1_dot = x_2
    x2_dot = sigma + bu
    sigma_dot = _f(...)      """


class ExtendedStateObserver:
    def __init__(self, x10, x20,
                 beta11, beta12, beta13, beta21, beta22, beta23,
                 alpha, delta):
        """ e1 = x1_hat - x1
            e2 = x2_hat - x2
            x1_hat_dot = x2_hat - beta11 * e1 - beta21 * fal(e2)
            x2_hat_dot = u + sigma_hat - beta22 * fal(e1)
            sigma_hat_dot = - beta13 * fal(e1) """
        self.beta11 = np.array(beta11)
        self.beta12 = np.array(beta12)
        self.beta13 = np.array(beta13)
        self.beta21 = np.array(beta21)
        self.beta22 = np.array(beta22)
        self.beta23 = np.array(beta23)
        self.alpha = alpha
        self.delta = delta
        self.x1_hat = np.array(x10)
        self.x2_hat = np.array(x20)
        self.sigma_hat = np.zeros_like(x10)
        self.y_list = []
        self.x1_hat_list = []
        self.sigma_hat_list = []

    # def update(self, y1, u, dt):
    #     e1 = self.x1_hat - y1
    #     fal_e1 = self.fal(e1, self.alpha, self.delta)
    #     self.x1_hat += dt * (self.x2_hat - self.beta11 * fal_e1)
    #     self.x2_hat += dt * (u + self.sigma_hat - self.beta12 * fal_e1)
    #     self.sigma_hat += dt * (-self.beta13 * fal_e1)
    #     self.y_list.append(np.copy(y1))
    #     self.x1_hat_list.append(np.copy(self.x1_hat))
    #     self.sigma_hat_list.append(np.copy(self.sigma_hat))
    #     return self.x1_hat, self.x2_hat, self.sigma_hat

    def update(self, y1, y2, u, dt):
        e1 = self.x1_hat - y1
        e2 = self.x2_hat - y2
        fal_e1 = self.fal(e1, self.alpha, self.delta)
        fal_e2 = self.fal(e2, self.alpha, self.delta)
        self.x1_hat += dt * (self.x2_hat      - self.beta11 * fal_e1 - self.beta21 * fal_e2)
        self.x2_hat += dt * (u+self.sigma_hat - self.beta12 * fal_e1 - self.beta22 * fal_e2)
        self.sigma_hat +=  dt * (-self.beta13 * fal_e1 - self.beta23 * fal_e2)
        self.y_list.append(np.copy(y1))
        self.x1_hat_list.append(np.copy(self.x1_hat))
        self.sigma_hat_list.append(np.copy(self.sigma_hat))
        return self.x1_hat, self.x2_hat, self.sigma_hat

    def fal(self, e, alpha, delta):
        abse = np.abs(e)
        output = np.zeros_like(e)
        # 使用布尔索引来处理不同的条件
        output[abse <= delta] = e[abse <= delta] / delta ** (1 - alpha)
        output[abse > delta] = abse[abse > delta] ** alpha * np.sign(e[abse > delta])
        return output

    def plot(self):
        import matplotlib.pyplot as plt
        n = len(self.y_list)  # 获取数据组数，假设self.y_list和self.x1_hat_list长度相同
        assert n == len(self.x1_hat_list), "The lengths of y_list and x1_hat_list should be the same"

        # 创建包含3个子图的图形，figsize可以根据需要调整图形大小
        fig, axes = plt.subplots(3, 1, figsize=(8, 12))

        # 提取self.y_list和self.x1_hat_list中每组数据的对应元素
        y_list_elem1 = [data[0] for data in self.y_list]
        y_list_elem2 = [data[1] for data in self.y_list]
        y_list_elem3 = [data[2] for data in self.y_list]
        x1_hat_list_elem1 = [data[0] for data in self.x1_hat_list]
        x1_hat_list_elem2 = [data[1] for data in self.x1_hat_list]
        x1_hat_list_elem3 = [data[2] for data in self.x1_hat_list]
        sigmax_hat_list = [data[0] for data in self.sigma_hat_list]
        sigmay_hat_list = [data[1] for data in self.sigma_hat_list]
        sigmaz_hat_list = [data[2] for data in self.sigma_hat_list]

        # 在第一个子图中绘制self.y_list和self.x1_hat_list的第一个元素对应的曲线
        axes[0].plot(np.arange(n), y_list_elem1, label='x')
        axes[0].plot(np.arange(n), x1_hat_list_elem1, label='x_hat')
        axes[0].plot(np.arange(n), sigmax_hat_list, label='sigma_x_hat')
        axes[0].set_title('axes x')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True)

        # 在第二个子图中绘制self.y_list和self.x1_hat_list的第二个元素对应的曲线
        axes[1].plot(np.arange(n), y_list_elem2, label='y')
        axes[1].plot(np.arange(n), x1_hat_list_elem2, label='y_hat')
        axes[1].plot(np.arange(n), sigmay_hat_list, label='sigma_y_hat')
        axes[1].set_title('axes y')
        axes[1].set_ylabel('Value')
        axes[1].legend()
        axes[1].grid(True)

        # 在第三个子图中绘制self.y_list和self.x1_hat_list的第三个元素对应的曲线
        axes[2].plot(np.arange(n), y_list_elem3, label='z')
        axes[2].plot(np.arange(n), x1_hat_list_elem3, label='z_hat')
        axes[2].plot(np.arange(n), sigmaz_hat_list, label='sigma_z_hat')
        axes[2].set_title('axes z')
        axes[2].set_xlabel('Index')
        axes[2].set_ylabel('Value')
        axes[2].legend()
        axes[2].grid(True)

        # plt.show()


if __name__ == '__main__':
    observer = ExtendedStateObserver([0.,0.,0.],[0.,0.,0.],
                                     [0.1]*3, [0.1]*3, [0.1]*3,
                                     0.7, 0.1)
    observer.update([1.0]*3, [.0]*3, 0.1)
    flist= []
    import matplotlib.pyplot as plt
    for j in np.linspace(-10, 10, 100):
        s = observer.fal(j, observer.alpha, 0)
        flist.append(s)
    plt.plot(np.linspace(0, 10, 100), flist)
    plt.show()