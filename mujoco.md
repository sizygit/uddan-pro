# 笔记

# mujoco相关

​		mujoco**可以以原生MJCF**格式以及流行但功能更有限的**URDF**格式加载 XML 模型文件 ([XML 参考](https://mujoco.readthedocs.io/en/stable/XMLreference.html) )。mujucot提供了非常详细的[文档说明](https://mujoco.readthedocs.io/en/stable/overview.html)，可以帮助我们进行代码编写，关于底层的c++代码有着详细的说明。

​		mujoco的仿真过程可参考[文档](https://mujoco.readthedocs.io/en/stable/programming/simulation.html)，其讲解了mujoco实现的重要步骤，以及仿真计算相关的细节，包括正向、逆向动力学计算、积分器、雅可比矩阵等概念，在必要的时候可进行参考。

​		使用python进行mujoco的仿真通常需要加载xml模型文件、步进仿真、渲染几个步骤，我们可以使用`mujoco_viewer`第三方库来实现更简单的渲染，如下为一个最简单的示例：

```python
model = mujoco.MjModel.from_xml_path("../udaan/models/assets/mjcf/multi2_quad_pointmass.xml") # 加载模型
data = mujoco.MjData(model) # 定义变量存储模型数据
viewer = mujoco_viewer.MujocoViewer(model, data)
while data.time < 5:
  mujoco.mj_step(model, data) # 利用data里面关于控制力的分量进行一步仿真计算，并更新模型数据
  viewer.render() # 渲染
```

​		也可以使用mujoco自带的`mujoco.viewer.launch_passive`与`sync()`方法来进行步进与渲染,mujoco.viewer提供的API会创建一个Interactive GUI viewer，其内部创建了诸多子进程来实现实时的渲染：

```python
model = mujoco.MjModel.from_xml_string(model_xml)
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch_passive(model, data)
viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD  # mjvOption:https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjvoption 想要显示的坐标系
viewer.cam.lookat = [0.0, 0.0, 0.0]  # mjvCamera:https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjvcamera
n = 2
while data.time < 5:
    for _ in range(n): # n substeps for a render step
        mujoco.mj_step(model, data)
        data.ctrl[0] = 0.1
    if viewer.is_running():
        viewer.sync() # 进行渲染的更新
        time.sleep(model.opt.timestep * n)

```

​		对于mujoco的渲染过程，主要包括抽象化的场景数据 [mjvScene](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjvscene)和图形渲染相关两部分组成，目前发现python对渲染后的环境添加新的几何体存在着bug，只能显示其label，主要是因为对于mjvScene的gemos成员是一个元组，无法添加新的geoms的相关属性。

MuJoCo关于坐标系与旋转的定义参考[文档](https://mujoco.readthedocs.io/en/stable/modeling.html#frame-orientations)，通常按照xml文件定义的顺序，状态量通常为:

```cpp
x = (mjData.time, mjData.qpos, mjData.qvel, mjData.act)
```

MuJoCo 的控制向量指定模型中定义的执行器的控制信号 (mjData.ctrl)，或直接应用关节空间 (mjData.qfrc_applied) 或笛卡尔空间 (mjData.xfrc_applied) 中指定的力和扭矩，其维度与xml定义`actuator`数量相关。

```cpp
u = (mjData.ctrl, mjData.qfrc_applied, mjData.xfrc_applied)
```

​		对于tendon定义的柔性约束，通常可以通过[mjData.efc_force](https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjdata)得到，或者通过设置[sensor/tendonlimitfrc](https://mujoco.readthedocs.io/en/latest/XMLreference.html#sensor-tendonlimitfrc)来读取mjData.sensotData得到，这两种方式得到的均为力的标量大小。







# 代码结构相关

1. 相应文件夹下使用`__init__.py` 文件作为包识别标志

project/
│
├── udaan/
│   ├── models/
│   │   ├── mujoco/
│   │   │   ├── \__init\__.py
│   │   │   ├── quadrotor.py
│   │   │   ├── quadrotor_cspayload.py
│   │   │   ├── multi_quad_cs_pointmass.py
│   │   │   ├── quadrotor_comparison.py
│   │   ├── \__init\__.py
│   ├── \__init\__.py
├── other_folder/
│   ├── some_script.py

`__init__.py` 文件在 Python 包中有以下几个主要作用：  

- 标识包：表明包含它的目录是一个 Python 包。没有这个文件，Python 解释器不会将该目录识别为包。  

- 初始化包：当包被导入时，`__init__.py`文件中的代码会被执行。可以在这里进行包的初始化工作，比如导入子模块、设置包级别的变量等。

- 控制导入行为：可以在 `__init__.py` 文件中定义 `__all__` 列表来控制 from package import * 语句的行为。

- 组织代码：可以在`__init__.py` 文件中导入包内的模块，使得包的使用更加方便和直观。 

    

  



$$
\begin{gathered}
D=[d_1, ..., d_n]  \\ 
d_i = [sin\theta_icos\phi_i,sin\theta_isin\phi_i,cos\theta_i]^T \\
t=D^T(DD^T)^{-1}F \\
\frac{\partial ||t||^2_2}{\partial d} 

\end{gathered}
$$
