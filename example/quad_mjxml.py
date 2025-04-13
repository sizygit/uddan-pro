import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString


class multi_xml(object):
    def __init__(self, filepath="./"):
        # 解析 XML 文件
        self.filepath = filepath
        self.tree = None
        self.root = None
        self.root_mujoco = None

    def read_cs_xml(self, file):
        self.tree = ET.parse(self.filepath + file)
        self.root = self.tree.getroot()
        # 查找 name 属性为 quadrotor0 的 <body> 元素
        body_info = {}
        # 表示在 XML 文档中查找所有 <body> 元素，不论它们位于 XML 树的哪个层级。"." 表示从当前根节点开始查找，"//" 表示递归查找所有子节点。
        worlbody = self.root.findall(".//body")
        for j in range(len(worlbody)):
            body_info['name'] = worlbody[j].attrib.get('name')
            body_info['pos0'] = worlbody[j].attrib.get('pos')
            body_info['quat0'] = worlbody[j].attrib.get('quat')
            inertial = worlbody[j].find("inertial")
            body_info['diaginertia'] = inertial.attrib.get("diaginertia")
            body_info['mass'] = inertial.attrib.get("mass")
            for key in body_info:
                print(f"{key}:[{body_info[key]}]", end='  ')
            print("\n")

        motor = self.root.findall(".//motor")
        actuator_info = {}
        for j in range(len(motor)):
            actuator_info['site'] = motor[j].attrib.get('site')
            actuator_info['ctrl'] = motor[j].attrib.get('ctrlrange')
            print(f"{actuator_info['site']}: [{actuator_info['ctrl']}]", end="  ")
            if (j + 1) % 4 == 0:
                print("\n")

    def build_xml(self, n_quad, timestep, mass, diaginertia, mass_load,
                  diaginertia_load, ctrlrange_dict, quad_poso, load_pos0, L,
                  file='./generate1.xml'):
        self.root_mujoco = ET.Element('mujoco', model="QuadSuspend")
        self.build_root(timestep)
        self.build_body(n=n_quad, mass=mass, diaginertia=diaginertia, mass_load=mass_load,
                        diaginertia_load=diaginertia_load, quad_poso=quad_poso, load_pos0=load_pos0)
        self.build_actuator(n=n_quad, ctrlrange_dict=ctrlrange_dict)

        self.build_tendon_sensor(n=n_quad, L=L)

        # 将ElementTree对象转换为字符串
        xml_string = ET.tostring(self.root_mujoco, encoding='utf-8')
        # 使用xml.dom.minidom解析字符串并美化格式
        dom = parseString(xml_string)
        pretty_xml_string = dom.toprettyxml(indent="  ")
        # 将美化后的XML字符串保存为文件
        if file != 'None':
            with open(file, 'w', encoding='utf-8') as f:
                f.write(pretty_xml_string)
            print(f"generate the xml file: {file}")
        return pretty_xml_string

    def build_root(self, timestep):
        """ 创建mujoco根元素下与环境场景相关的基本子元素"""
        # 创建<option>元素并设置属性
        option = ET.SubElement(self.root_mujoco, 'option')
        option.set('timestep', str(timestep))
        option.set('gravity', '0 0 -9.81')
        option.set('wind', '0 0 0')
        option.set('density', '1')
        option.set('viscosity', '1e-5')

        # 创建<compiler>元素并设置属性
        compiler = ET.SubElement(self.root_mujoco, 'compiler')
        compiler.set('angle', 'radian')
        compiler.set('coordinate', 'local')
        compiler.set('inertiafromgeom', 'false')

        # 创建<visual>元素和<map>子元素并设置属性
        visual = ET.SubElement(self.root_mujoco, 'visual')
        map_element = ET.SubElement(visual, 'map')
        map_element.set('fogstart', '3.0')  # 指定雾效（fog）开始的距离
        map_element.set('fogend', '5.0')
        map_element.set('force', '0.1')
        map_element.set('znear', '0.1')  # 定义近裁剪平面（near clipping plane）

        #  创建<asset>元素用于存储模型所需要的各种资源。这些资源包括纹理（textures）、材质（materials）、网格（meshes）等，
        asset = ET.SubElement(self.root_mujoco, 'asset')
        asset_element = ET.SubElement(asset, 'texture')
        asset_element.set('type', 'skybox')  # 表明该纹理是用于天空盒（skybox）
        asset_element.set('builtin', 'gradient')  # 一个内置的渐变（gradient）纹理类型
        asset_element.set('rgb1', '1.0 1.0 1.0')  # 定义了渐变的起始颜色和结束颜色
        asset_element.set('rgb2', '0.6 0.8 1.0')
        asset_element.set('width', '256')  # 定义了纹理的尺寸
        asset_element.set('height', '256')

    def build_body(self, n, mass, diaginertia, mass_load, diaginertia_load,
                   quad_poso, load_pos0):
        worldbody = ET.SubElement(self.root_mujoco, 'worldbody')
        # 创建<geom>元素并设置属性
        geom = ET.SubElement(worldbody, 'geom')
        geom.set('name', 'ground')
        geom.set('type', 'plane')
        geom.set('size', '25.0 25.0 0.02')
        geom.set('friction', '1.0')
        geom.set('pos', '0.0 0.0 0.0')
        geom.set('rgba', '0.8 0.9 0.8 1.0')

        # 创建<light>元素并设置属性
        light = ET.SubElement(worldbody, 'light')
        light.set('directional', 'true')
        light.set('diffuse', '.9 .9 .9')
        light.set('specular', '.3 .3 .3')
        light.set('pos', '0 0 4.0')
        light.set('dir', '0 0.15 -1')

        # 创建n架无人机的body子属性
        body = {}
        for j in range(n):
            body[j] = ET.SubElement(worldbody, 'body')
            body[j].set('name', f'quadrotor{j}')
            body[j].set('pos', str(quad_poso[j][0]) + ' ' +str(quad_poso[j][1]) + ' ' +str(quad_poso[j][2]))  # 根据索引调整位置的x坐标
            body[j].set('quat', '1 0 0 0')

            # 创建<inertial>元素并设置属性
            inertial = ET.SubElement(body[j], 'inertial')
            inertial.set('pos', '0. 0. 0.')
            inertial.set('mass', str(mass))
            inertial.set('diaginertia', str(diaginertia[0]) + ' ' + str(diaginertia[1]) + ' ' + str(diaginertia[2]))

            # 创建XYZ的body子属性
            xyzbody = ET.SubElement(body[j], 'body')
            # 对loadbody进行写入相关XML内容
            xyzbody.set('name', f'xyz_axes_{j}')
            xyzbody.set('pos', '0 0 0 ')
            # X 轴 (红色)
            ET.SubElement(xyzbody, 'geom', type='capsule', size='0.01', fromto='0 0 0 0.3 0 0', rgba='1 0 0 1')
            # Y 轴 (绿色)
            ET.SubElement(xyzbody, 'geom', type='capsule', size='0.01', fromto='0 0 0 0 0.3 0', rgba='0 1 0 1')
            # Z 轴 (蓝色)
            ET.SubElement(xyzbody, 'geom', type='capsule', size='0.01', fromto='0 0 0 0 0 0.3', rgba='0 0 1 1')

            # 创建<geom>元素（quad0_geom）并设置属性
            geom_quad = ET.SubElement(body[j], 'geom')
            geom_quad.set('name', f'quad{j}_geom')
            geom_quad.set('type', 'box')
            geom_quad.set('pos', '0.0 0.0 0.0')
            geom_quad.set('quat', '1 0 0 0')
            geom_quad.set('size', '0.08 0.04 0.025')
            geom_quad.set('rgba', '0.3 0.3 0.8 1.0')

            # 创建<site>元素（quad0_end1）并设置属性
            site_end1 = ET.SubElement(body[j], 'site')
            site_end1.set('name', f'quad{j}_end')
            site_end1.set('pos', '0.0 0 0.0')
            site_end1.set('type', 'sphere')
            site_end1.set('size', '0.01')

            # 创建<joint>元素并设置属性
            joint = ET.SubElement(body[j], 'joint')
            joint.set('name', f'quad{j}_root_joint')
            joint.set('type', 'free')
            joint.set('limited', 'false')
            joint.set('pos', '0.0 0.0 0.0')
            joint.set('axis', '0.0 0.0 1.0')
            joint.set('range', '-1.0 1.0')
            joint.set('armature', '0.0')
            joint.set('damping', '0.0')
            joint.set('stiffness', '0.0')

            # 创建四个<geom>和<geom>对应的<site>元素用于转子臂和螺旋桨
            rotor_arm_geoms = []
            rotor_prop_geoms = []
            rotor_sites = []
            quat_values = ['0.92388 0 0 0.382683', '0.382683 0 0 0.92388', '-0.382683 0 0 0.92388',
                           '-0.92388 0 0 0.382683']
            pos_values = [
                '0.1414213562373095 0.14142135623730953 0.0',
                '-0.1414213562373095 0.14142135623730953 0.0',
                '-0.1414213562373095 -0.1414213562373095 0.0',
                '0.1414213562373095 -0.14142135623730953 0.0'
            ]
            for i in range(4):
                # 创建<geom>元素（quad0_rotor_arm_geom_*）并设置属性
                rotor_arm_geom = ET.SubElement(body[j], 'geom')
                rotor_arm_geom.set('name', f'quad{j}_rotor_arm_geom_{i}')
                rotor_arm_geom.set('type', 'box')
                rotor_arm_geom.set('pos', '0.0 0.0 0.0')
                rotor_arm_geom.set('quat', quat_values[i])  # 根据索引和循环变量获取不同的quat值
                rotor_arm_geom.set('size', '0.2 0.01 0.01')
                rotor_arm_geom.set('rgba', '0.1 0.1 0.5 1.0')
                rotor_arm_geoms.append(rotor_arm_geom)

                # 创建<geom>元素（quad0_rotor_prop_geom_*）并设置属性
                rotor_prop_geom = ET.SubElement(body[j], 'geom')
                rotor_prop_geom.set('name', f'quad{j}_rotor_prop_geom_{i}')
                rotor_prop_geom.set('type', 'cylinder')
                rotor_prop_geom.set('size', '0.1 0.005')
                rotor_prop_geom.set('pos', pos_values[i])  # 根据索引和循环变量获取不同的pos值
                rotor_prop_geom.set('rgba', '0.5 0.1 0.1 1.0')
                rotor_prop_geoms.append(rotor_prop_geom)

                # 创建<site>元素（quad0_site*）并设置属性
                rotor_site = ET.SubElement(body[j], 'site')
                rotor_site.set('name', f'quad{j}_site{i}')
                rotor_site.set('type', 'box')
                rotor_site.set('pos', pos_values[i])  # 根据索引和循环变量获取不同的pos值
                rotor_site.set('quat', '1 0 0 0')
                rotor_site.set('size', '0.01 0.01 0.01')
                rotor_site.set('rgba', '0.0 1 1 1.0')
                rotor_sites.append(rotor_site)

            # 创建<site>元素（quad0_thrust、quad0_Mx、quad0_My、quad0_Mz）并设置属性
            thrust_site = ET.SubElement(body[j], 'site')
            thrust_site.set('name', f'quad{j}_thrust')
            thrust_site.set('type', 'box')
            thrust_site.set('pos', '0.0 0.0 0.0')
            thrust_site.set('quat', '1 0 0 0')
            thrust_site.set('size', '0.035 0.035 0.035')
            thrust_site.set('rgba', '0.0 1 1 1.0')

            mx_site = ET.SubElement(body[j], 'site')
            mx_site.set('name', f'quad{j}_Mx')
            mx_site.set('type', 'box')
            mx_site.set('pos', '0.0 0.0 0.0')
            mx_site.set('quat', '1 0 0 0')
            mx_site.set('size', '0.06 0.035 0.025')
            mx_site.set('rgba', '0.0 1 1 1.0')

            my_site = ET.SubElement(body[j], 'site')
            my_site.set('name', f'quad{j}_My')
            my_site.set('type', 'box')
            my_site.set('pos', '0.0 0.0 0.0')
            my_site.set('quat', '1 0 0 0')
            my_site.set('size', '0.06 0.035 0.025')
            my_site.set('rgba', '0.0 1 1 1.0')

            mz_site = ET.SubElement(body[j], 'site')
            mz_site.set('name', f'quad{j}_Mz')
            mz_site.set('type', 'box')
            mz_site.set('pos', '0.0 0.0 0.0')
            mz_site.set('quat', '1 0 0 0')
            mz_site.set('size', '0.06 0.035 0.025')
            mz_site.set('rgba', '0.0 1 1 1.0')

        # 创建载荷的body子属性
        loadbody = ET.SubElement(worldbody, 'body')
        # 对loadbody进行写入相关XML内容
        loadbody.set('name', 'payload')
        loadbody.set('pos', str(load_pos0[0]) + ' ' + str(load_pos0[1]) + ' ' + str(load_pos0[2]))
        loadbody.set('quat', '1 0 0 0')

        inertial = ET.SubElement(loadbody, 'inertial')
        inertial.set('pos', '0. 0. 0.')
        inertial.set('mass', str(mass_load))
        inertial.set('diaginertia', str(diaginertia_load[0])+ ' ' + str(diaginertia_load[1]) + ' ' + str(diaginertia_load[2]))

        geom = ET.SubElement(loadbody, 'geom')
        geom.set('type', 'sphere')
        geom.set('pos', '0. 0. 0.0')
        geom.set('size', '0.05')

        site = ET.SubElement(loadbody, 'site')
        site.set('name', 'end')
        site.set('pos', '0.0 0 0.0')
        site.set('type', 'sphere')
        site.set('size', '0.01')

        joint = ET.SubElement(loadbody, 'joint')
        joint.set('type', 'free')

        # 创建XYZ的body子属性
        loadbody = ET.SubElement(worldbody, 'body')
        # 对loadbody进行写入相关XML内容
        loadbody.set('name', 'xyz_axes')
        loadbody.set('pos', '0 0 0 ')
        # X 轴 (红色)
        ET.SubElement(loadbody, 'geom', type='capsule', size='0.01', fromto='0 0 0 0.3 0 0', rgba='1 0 0 1', contype="0", conaffinity="0")
        # Y 轴 (绿色)
        ET.SubElement(loadbody, 'geom', type='capsule', size='0.01', fromto='0 0 0 0 0.3 0', rgba='0 1 0 1', contype="0", conaffinity="0")
        # Z 轴 (蓝色)
        ET.SubElement(loadbody, 'geom', type='capsule', size='0.01', fromto='0 0 0 0 0 0.3', rgba='0 0 1 1', contype="0", conaffinity="0")

    def build_actuator(self, n, ctrlrange_dict):
        actuator = ET.SubElement(self.root_mujoco, 'actuator')
        for quad_index in range(n):
            for component in ['thrust', 'Mx', 'My', 'Mz']:
                site_name = f"quad{quad_index}_{component}"
                motor = ET.SubElement(actuator, 'motor')
                motor.set('ctrllimited', 'true')
                motor.set('site', site_name)
                motor.set('ctrlrange', f"{ctrlrange_dict[component][0]} {ctrlrange_dict[component][1]}")
                if component == 'thrust':
                    motor.set('gear', '0.0 0.0 1.0 0.0 0.0 0.0')
                elif component == 'Mx':
                    motor.set('gear', '0.0 0.0 0.0 1.0 0.0 0.0')
                elif component == 'My':
                    motor.set('gear', '0.0 0.0 0.0 0.0 1.0 0.0')
                elif component == 'Mz':
                    motor.set('gear', '0.0 0.0 0.0 0.0 0.0 1.0')

    def build_tendon_sensor(self, n, L):
        tendon = ET.SubElement(self.root_mujoco, 'tendon')
        sensor = ET.SubElement(self.root_mujoco, 'sensor')

        for quad_index in range(n):
            # 生成<tendon>中的<spatial>元素
            spatial = ET.SubElement(tendon, 'spatial')
            spatial.set('name', f'tendon{quad_index}')
            spatial.set('limited', 'true')
            spatial.set('range', '0.0 ' + str(L))  # 定义了肌腱拉伸的范围
            spatial.set('width', '0.005')
            spatial.set('damping', '0.0')  # 表示阻尼系数。阻尼在物理系统中用于消耗能量，模拟像空气阻力、内部摩擦等因素。
            spatial.set('stiffness', '0.0')  # 表示刚度，它决定了肌腱抵抗拉伸的能力。刚度为 0.0 意味着肌腱在这个设置下没有抵抗拉伸的弹性力，

            # 添加<tendon>中<spatial>元素的<site>子元素
            site_quad = ET.SubElement(spatial, 'site')
            site_quad.set('site', f"quad{quad_index}_end")
            site_end = ET.SubElement(spatial, 'site')
            site_end.set('site', "end")  # 定义tendon的两个链接点的site，绳索长度即为载荷与无人机间距离

            # 生成<sensor>中的<force>元素
            force = ET.SubElement(sensor, 'tendonlimitfrc')
            force.set('tendon', f'tendon{quad_index}')
        for j in range(n):
            acc = ET.SubElement(sensor, 'accelerometer')
            acc.set('name', f'quad{j}_acc')
            acc.set( 'site', f'quad{j}_end')
        acc = ET.SubElement(sensor, 'accelerometer')
        acc.set('name', 'payload_acc')
        acc.set('site', 'end')

if __name__ == '__main__':
    s = multi_xml()
    #s.read_cs_xml('test.xml')
    s.build_xml(2, timestep=.02,
                mass=0.75, diaginertia=[0.0053, 0.0049, 0.0098], mass_load=1,
                diaginertia_load=[0.00015, 0.00015, 0.00015],
                ctrlrange_dict={
                    'thrust': [0.0, 40.0],
                    'Mx': [-3.0, 3.0],
                    'My': [-3.0, 3.0],
                    'Mz': [-3.0, 3.0]},
                quad_poso=[
                    [-0.6, 0.0, 0.8],[0.6, 0.0, 0.8]
                           ],
                load_pos0=[0.0, 0.0, 0.0], L=1.5,file='./generate2.xml'
                )
    s.build_xml(4, timestep=.02,
                mass=0.75, diaginertia=[0.0053, 0.0049, 0.0098], mass_load=1,
                diaginertia_load=[0.00015, 0.00015, 0.00015],
                ctrlrange_dict={
                    'thrust': [0.0, 40.0],
                    'Mx': [-3.0, 3.0],
                    'My': [-3.0, 3.0],
                    'Mz': [-3.0, 3.0]},
                quad_poso=[
                    [0.6, 0.0, 0.8],[0.0, 0.6, 0.8], [-0.6, 0.0, 0.8], [0.0, -0.6, 0.8]
                           ],
                load_pos0=[0.0, 0.0, 0.0], L=1.5,file='./generate4.xml'
                )
    s.build_xml(1, timestep=.02,
                mass=0.75, diaginertia=[0.0053, 0.0049, 0.0098], mass_load=1,
                diaginertia_load=[0.00015, 0.00015, 0.00015],
                ctrlrange_dict={
                    'thrust': [0.0, 40.0],
                    'Mx': [-3.0, 3.0],
                    'My': [-3.0, 3.0],
                    'Mz': [-3.0, 3.0]},
                quad_poso=[[0.0, 0.0, 1.0]
                           ],
                load_pos0=[0.0, 0.0, 0.0], L=1.5,file='./generate1.xml'
                )