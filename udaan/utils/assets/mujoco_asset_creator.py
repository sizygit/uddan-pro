#!/usr/bin/python

import xml.etree.ElementTree as ET
import math
import numpy as np
from scipy.spatial.transform import Rotation as sp_rot


class MujocoAssetCreator(object):
    """Adapted from IsaacGymEnvs

    Args:
        object (_type_): _description_
    """

    def __init__(self, name="mujoco_asset"):
        self.name = name
        self.root = ET.Element("mujoco")
        self.root.attrib["model"] = "Quadcopter"
        compiler = ET.SubElement(self.root, "compiler")
        compiler.attrib["angle"] = "degree"
        compiler.attrib["coordinate"] = "local"
        compiler.attrib["inertiafromgeom"] = "true"
        self.asset()
        self.worldbody = ET.SubElement(self.root, "worldbody")

        self.ground_plane(size=5.0)
        self.light()
        return

    def save_to(self, filename="mujoco_asset.xml", verbose=False):
        """Save the asset to a file

        Args:
            filename (str, optional): Name of the file to save to. Defaults to "mujoco_asset.xml".
        """
        # Pretty printing to Python shell for testing purposes
        from xml.dom import minidom

        xmlstr = minidom.parseString(ET.tostring(
            self.root)).toprettyxml(indent="   ")
        if verbose:
            print(xmlstr)
        with open(filename, "w") as f:
            f.write(xmlstr)
        return

    def asset(self):
        asset = ET.SubElement(self.root, "asset")
        texture = ET.SubElement(asset, "texture")
        texture.attrib["type"] = "skybox"
        texture.attrib["builtin"] = "gradient"
        texture.attrib["rgb1"] = "1.0 1.0 1.0"
        texture.attrib["rgb2"] = "0.6 0.8 1.0"
        texture.attrib["width"] = "256"
        texture.attrib["height"] = "256"
        return asset

    def fancy_asset(self):
        visual = ET.SubElement(self.root, "visual")
        map = ET.SubElement(visual, "map")
        map.attrib["fogstart"] = "3.0"
        map.attrib["fogend"] = "5.0"
        map.attrib["force"] = "0.1"
        map.attrib["znear"] = "0.1"
        asset = ET.SubElement(self.root, "asset")
        texture1 = ET.SubElement(asset, "texture")
        texture1.attrib["type"] = "skybox"
        texture1.attrib["builtin"] = "gradient"
        texture1.attrib["rgb1"] = "1.0 1.0 1.0"
        texture1.attrib["rgb2"] = "0.6 0.8 1.0"
        texture1.attrib["width"] = "127"
        texture1.attrib["height"] = "1278"
        texture2 = ET.SubElement(asset, "texture")
        texture2.attrib["name"] = "texplane"
        texture2.attrib["type"] = "2d"
        texture2.attrib["builtin"] = "checker"
        texture2.attrib["rgb1"] = "0.2 0.3 0.4"
        texture2.attrib["rgb2"] = ".1 0.15 0.2"
        texture2.attrib["width"] = "512"
        texture2.attrib["height"] = "512"

        mat1 = ET.SubElement(asset, "material")
        mat1.attrib["name"] = "MatPlane"
        mat1.attrib["texture"] = "texplane"
        mat1.attrib["reflectance"] = "0.5"
        mat1.attrib["texrepeat"] = "1 1"
        mat1.attrib["texuniform"] = "true"
        return asset

    def exclude_contact(self, body1: str, body2: str):
        contact = ET.SubElement(self.root, "contact")
        exclude = ET.SubElement(contact, "exclude")
        exclude.attrib["body1"] = body1
        exclude.attrib["body2"] = body2
        return contact

    def tendon(self, parent):
        tendon = ET.SubElement(parent, "tendon")
        return tendon

    def spatial(
        self,
        parent,
        site1,
        site2,
        range=[0.0, 1.0],
        width=0.005,
        damping=0.0,
        stiffness=0.0,
    ):
        spatial = ET.SubElement(parent, "spatial")
        spatial.attrib["limited"] = "true"
        spatial.attrib["range"] = " ".join([str(x) for x in range])
        spatial.attrib["width"] = str(width)
        spatial.attrib["damping"] = str(damping)
        spatial.attrib["stiffness"] = str(stiffness)
        s1 = ET.SubElement(spatial, "site")
        s1.attrib["site"] = site1
        s2 = ET.SubElement(spatial, "site")
        s2.attrib["site"] = site2
        return spatial

    def ground_plane(
        self,
        pos=np.array([0.0, 0, 0.0]),
        size=2,
        friction=1.0,
        rgba=[0.8, 0.9, 0.8, 1.0],
    ):
        """Create a ground plane

        Args:
            size (int, optional): Size of the ground plane. Defaults to 10.
            friction (float, optional): Friction of the ground plane. Defaults to 1.0.
            rgba (list, optional): RGBA values of the ground plane. Defaults to [0.8, 0.9, 0.8, 1.0].
        """
        geom = ET.SubElement(self.worldbody, "geom")
        geom.attrib["name"] = "ground"
        geom.attrib["type"] = "plane"
        geom.attrib["size"] = str(size) + " " + str(size) + " 0.02"
        geom.attrib["friction"] = str(friction)
        geom.attrib["pos"] = " ".join([str(x) for x in pos])
        geom.attrib["rgba"] = " ".join([str(x) for x in rgba])
        return geom

    def light(self):
        light = ET.SubElement(self.worldbody, "light")
        light.attrib["directional"] = "true"
        light.attrib["diffuse"] = ".9 .9 .9"
        light.attrib["specular"] = ".3 .3 .3"
        light.attrib["pos"] = "0 0 4.0"
        light.attrib["dir"] = "0 0.15 -1"
        return light

    def body(
            self,
            parent,
            name,
            pos=np.array([0.0, 0.0, 0.0]),
            quat=np.array([0.0, 0.0, 0.0, 1.0]),
            mass=1.0,
            inertia=np.array([1.0, 1.0, 1.0]),
            rgba=[0.8, 0.9, 0.8, 1.0],
    ):
        body = ET.SubElement(parent, "body")
        body.attrib["name"] = name
        body.attrib["pos"] = " ".join([str(x) for x in pos])
        body.attrib["quat"] = "%g %g %g %g" % (quat[3], quat[0], quat[1],
                                               quat[2])
        # body.attrib["mass"] = str(mass)
        # body.attrib["inertia"] = " ".join([str(x) for x in inertia])
        return body

    def cylinder(
        self,
        parent,
        name,
        pos=np.array([0.0, 0.0, 0.0]),
        mass=0.01,
        density=1000,
        radius=0.1,
        length=0.1,
        alpha=1.0,
        rgb=[0.5, 0.1, 0.1],
    ):
        geom = ET.SubElement(parent, "geom")
        geom.attrib["name"] = name
        geom.attrib["type"] = "cylinder"
        geom.attrib["size"] = "%g %g" % (radius, 0.5 * length)
        geom.attrib["pos"] = " ".join([str(x) for x in pos])
        geom.attrib["mass"] = str(mass)
        geom.attrib["density"] = str(density)
        geom.attrib["rgba"] = " ".join([str(x) for x in rgb + [alpha]])
        return geom

    def sphere(
        self,
        parent,
        name,
        pos=np.array([0.0, 0.0, 0.0]),
        quat=np.array([0.0, 0.0, 0.0, 1.0]),
        radius=0.1,
        mass=0.01,
        alpha=1.0,
        rgb=[0.1, 0.5, 0.1],
        density=1000,
    ):
        geom = ET.SubElement(parent, "geom")
        geom.attrib["name"] = name
        geom.attrib["type"] = "sphere"
        geom.attrib["size"] = "%g" % (radius)
        geom.attrib["mass"] = str(mass)
        geom.attrib["density"] = str(density)
        geom.attrib["rgba"] = " ".join([str(x) for x in rgb + [alpha]])
        return geom

    def box(
        self,
        parent,
        name,
        pos=np.array([0.0, 0.0, 0.0]),
        quat=np.array([0.0, 0.0, 0.0, 1.0]),
        size=np.array([0.1, 0.1, 0.1]),
        mass=0.01,
        alpha=1.0,
        rgb=[0.1, 0.1, 0.5],
        density=1000,
    ):
        geom = ET.SubElement(parent, "geom")
        geom.attrib["name"] = name
        geom.attrib["type"] = "box"
        geom.attrib["pos"] = " ".join([str(x) for x in pos])
        geom.attrib["quat"] = "%g %g %g %g" % (quat[3], quat[0], quat[1],
                                               quat[2])
        geom.attrib["size"] = " ".join([str(x) for x in size])
        geom.attrib["mass"] = str(mass)
        geom.attrib["density"] = str(density)
        geom.attrib["rgba"] = " ".join([str(x) for x in rgb + [alpha]])
        return geom

    def joint(
        self,
        parent,
        name,
        type,
        pos=np.array([0.0, 0.0, 0.0]),
        axis=np.array([0.0, 0.0, 1.0]),
        range=np.array([-1.0, 1.0]),
        damping=0.1,
        stiffness=0.0,
        armature=0.0,
        limited=False,
    ):
        joint = ET.SubElement(parent, "joint")
        joint.attrib["name"] = name
        joint.attrib["type"] = type
        if limited:
            joint.attrib["limited"] = "true"
        else:
            joint.attrib["limited"] = "false"
        joint.attrib["pos"] = " ".join([str(x) for x in pos])
        if not type == "ball":
            joint.attrib["axis"] = " ".join([str(x) for x in axis])
            joint.attrib["range"] = " ".join([str(x) for x in range])
            joint.attrib["armature"] = str(armature)
        joint.attrib["damping"] = str(damping)
        joint.attrib["stiffness"] = str(stiffness)
        return joint

    def site(
            self,
            parent,
            name,
            type="box",
            pos=np.array([0.0, 0.0, 0.0]),
            quat=np.array([0.0, 0.0, 0.0, 1.0]),
            size=np.array([0.035, 0.035, 0.035]),
            rgba=[0.1, 0.1, 0.5, 1.0],
    ):
        site = ET.SubElement(parent, "site")
        site.attrib["name"] = name
        site.attrib["type"] = type
        site.attrib["pos"] = " ".join([str(x) for x in pos])
        site.attrib["quat"] = "%g %g %g %g" % (quat[3], quat[0], quat[1],
                                               quat[2])
        site.attrib["size"] = " ".join([str(x) for x in size])
        site.attrib["rgba"] = " ".join([str(x) for x in rgba])
        return site

    def actuator(self, parent):
        actuator = ET.SubElement(parent, "actuator")
        return actuator

    def motor(self,
              parent,
              site,
              range=[],
              gear=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])):
        motor = ET.SubElement(parent, "motor")
        motor.attrib["ctrllimited"] = "true"
        motor.attrib["site"] = site
        motor.attrib["ctrlrange"] = " ".join([str(x) for x in range])
        motor.attrib["gear"] = " ".join([str(x) for x in gear])
        return motor

    def velocity(
            self,
            parent,
            site,
            range=[-1.0, 1.0],
            kv=0.1,
            gear=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ):
        velocity = ET.SubElement(parent, "velocity")
        velocity.attrib["ctrllimited"] = "true"
        velocity.attrib["site"] = site
        velocity.attrib["ctrlrange"] = " ".join([str(x) for x in range])
        velocity.attrib["gear"] = " ".join([str(x) for x in gear])
        velocity.attrib["kv"] = str(kv)
        return velocity

    def create_isaacgym_quadcopter(
        self,
        parent,
        name,
        pos,
        chassis_radius=0.1,
        chassis_thickness=0.03,
        rotor_radius=0.04,
        rotor_thickness=0.01,
        rotor_arm_radius=0.01,
    ):
        chassis = self.body(parent, name, pos)
        chassis_geom = self.cylinder(
            chassis,
            "geom",
            radius=chassis_radius,
            length=chassis_thickness,
            mass=0.75,
            density=1000,
        )
        chassis_joint = self.joint(chassis, "root_joint", "free")

        zaxis = np.array([0, 0, 1])
        rotor_arm_offset = np.array(
            [chassis_radius + 0.25 * rotor_arm_radius, 0, 0])
        pitch_joint_offset = np.array([0, 0, 0])
        rotor_offset = np.array([rotor_radius + 0.25 * rotor_arm_radius, 0, 0])

        rotor_angles = [
            0.25 * math.pi, 0.75 * math.pi, 1.25 * math.pi, 1.75 * math.pi
        ]
        for i in range(len(rotor_angles)):
            angle = rotor_angles[i]
            # print("rotor_arm_offset", rotor_arm_offset)
            # print("angle: ", angle)
            r = sp_rot.from_rotvec(zaxis * angle)
            rotor_arm_quat = r.as_quat()
            rotor_arm_pos = r.apply(rotor_arm_offset)
            # print(rotor_arm_pos)
            pitch_joint_pos = pitch_joint_offset
            rotor_pos = rotor_offset
            rotor_quat = np.array([0, 0, 0, 1])

            rotor_arm = self.body(chassis, ("rotor_arm_%d" % i), rotor_arm_pos,
                                  rotor_arm_quat)
            rotor_arm_geom = self.sphere(
                rotor_arm,
                "rotor_arm_geom_%d" % i,
                radius=rotor_arm_radius,
                mass=0.01,
                rgba=[0.1, 0.1, 0.5, 1.0],
                density=200,
            )
            rotor_pitch_joint = self.joint(
                rotor_arm,
                "rotor_pitch_joint_%d" % i,
                "hinge",
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                range=np.array([-30.0, 30.0]),
                limited=True,
            )
            rotor = self.body(rotor_arm, "rotor_%d" % i, rotor_pos, rotor_quat)
            rotor_geom = self.cylinder(
                rotor,
                "rotor_geom_%d" % i,
                radius=rotor_radius,
                length=rotor_thickness,
                mass=0.01,
                rgba=[0.2, 0.1, 0.3, 1.0],
                density=200,
            )
            rotor_joint = self.joint(
                rotor,
                "rotor_roll_%d" % i,
                "hinge",
                np.array([0.0, 0.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
                range=np.array([-30.0, 30.0]),
                limited=True,
            )
        return chassis

    def create_quadrotor(self, parent, name, pos, xtype=True):
        chassis = self.body(parent, name, pos)
        chassis_geom = self.box(
            chassis,
            "geom",
            size=np.array([0.08, 0.04, 0.025]),
            mass=0.75,
            density=1000,
            rgba=[0.3, 0.3, 0.8, 1.0],
        )
        chassis_joint = self.joint(chassis, "root_joint", "free")

        zaxis = np.array([0, 0, 1])
        l = 0.2
        rotor_arm_offset = np.array([l, 0, 0])

        rotor_angles = [
            0.25 * math.pi, 0.75 * math.pi, 1.25 * math.pi, 1.75 * math.pi
        ]
        if not xtype:
            rotor_angles = [0, 0.5 * math.pi, math.pi, 1.5 * math.pi]

        for i in range(len(rotor_angles)):
            angle = rotor_angles[i]
            r = sp_rot.from_rotvec(zaxis * angle)
            rotor_arm_quat = r.as_quat()
            rotor_prop_pos = r.apply(rotor_arm_offset)
            rotor_arm_geom = self.box(
                chassis,
                "rotor_arm_geom_%d" % i,
                quat=rotor_arm_quat,
                size=np.array([l, 0.01, 0.01]),
                mass=0.01,
            )
            rotor_prop = self.body(chassis, ("rotor_prop_%d" % i),
                                   rotor_prop_pos, np.array([0, 0, 0, 1]))
            rotor_joint = self.joint(
                rotor_prop,
                "rotor_joint_%d" % i,
                "hinge",
                pos=rotor_prop_pos,
                axis=np.array([0.0, 1.0, 0.0]),
                range=np.array([-0.001, 0.001]),
                limited=True,
                damping=1000,
                stiffness=10000,
                armature=0.0,
            )
            rotor_prop_geom = self.cylinder(rotor_prop,
                                            "rotor_prop_geom_%d" % i,
                                            radius=0.1,
                                            length=0.01,
                                            mass=0.05)

        return chassis

    def create_quadrotor0(self, parent, name, pos, **kwargs):
        xtype = kwargs["xtype"] if "xtype" in kwargs else True
        alpha = kwargs["alpha"] if "alpha" in kwargs else 1.0
        rgb = kwargs["rgb"] if "rgb" in kwargs else [0.8, 0.3, 0.3]
        unmodeled_dynamics = False
        if "unmodeled_mass" in kwargs:
            unmodeled_dynamics = True
            unmodeled_mass = kwargs["unmodeled_mass"]
        if "unmodeled_mass_loc" in kwargs:
            unmodeled_mass_loc = kwargs["unmodeled_mass_loc"]
        else:
            unmodeled_mass_loc = np.array([0.0, 0.0, 0.0])

        chassis = self.body(parent, name, pos)
        self.box(
            chassis,
            name + "_geom",
            size=np.array([0.08, 0.04, 0.025]),
            mass=0.75,
            density=1000,
            alpha=alpha,
            rgb=[0.3, 0.3, 0.8],
        )
        self.joint(chassis, name + "_root_joint", "free")

        zaxis = np.array([0, 0, 1])
        l = 0.2
        rotor_arm_offset = np.array([l, 0, 0])

        rotor_angles = [
            0.25 * math.pi, 0.75 * math.pi, 1.25 * math.pi, 1.75 * math.pi
        ]
        if not xtype:
            rotor_angles = [0, 0.5 * math.pi, math.pi, 1.5 * math.pi]

        for i in range(len(rotor_angles)):
            angle = rotor_angles[i]
            r = sp_rot.from_rotvec(zaxis * angle)
            rotor_arm_quat = r.as_quat()
            rotor_prop_pos = r.apply(rotor_arm_offset)
            self.box(chassis,
                     name + "_rotor_arm_geom_%d" % i,
                     quat=rotor_arm_quat,
                     size=np.array([l, 0.01, 0.01]),
                     mass=0.01,
                     alpha=alpha,
                     rgb=[0.1, 0.1, 0.1])
            self.cylinder(
                chassis,
                name + "_rotor_prop_geom_%d" % i,
                pos=rotor_prop_pos,
                radius=0.1,
                length=0.01,
                mass=0.05,
                rgb=rgb,
                alpha=alpha,
            )

        if unmodeled_dynamics:
            self.box(
                chassis,
                name + "_unmodeled_mass",
                size=np.array([0.025, 0.025, 0.025]),
                mass=unmodeled_mass,
                density=1000,
                alpha=alpha,
                rgb=[0.3, 0.3, 0.8],
                pos=unmodeled_mass_loc,
            )

        # Creating actuator sites
        self.site(
            chassis,
            name + "_end1",
            pos=np.array([0.0, 0.0, 0.0]),
            type="sphere",
            size=[0.01],
        )
        self.site(
            chassis,
            name + "_thrust",
            pos=np.array([0.0, 0.0, 0.0]),
            rgba=[0.0, 1, 1, alpha],
        )
        self.site(
            chassis,
            name + "_Mx",
            pos=np.array([0.0, 0.0, 0.0]),
            size=np.array([0.06, 0.035, 0.025]),
            rgba=[0.0, 1, 1, alpha],
        )
        self.site(
            chassis,
            name + "_My",
            pos=np.array([0.0, 0.0, 0.0]),
            size=np.array([0.06, 0.035, 0.025]),
            rgba=[0.0, 1, 1, alpha],
        )
        self.site(
            chassis,
            name + "_Mz",
            pos=np.array([0.0, 0.0, 0.0]),
            size=np.array([0.06, 0.035, 0.025]),
            rgba=[0.0, 1, 1, alpha],
        )
        # Creating actuators
        actuator = self.actuator(self.root)
        self.motor(
            actuator,
            site=name + "_thrust",
            range=[0.0, 30.0],
            gear=np.array([0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        )
        self.motor(
            actuator,
            site=name + "_Mx",
            range=[-3.0, 3.0],
            gear=np.array([0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        )
        self.motor(
            actuator,
            site=name + "_My",
            range=[-3.0, 3.0],
            gear=np.array([0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        )
        self.motor(
            actuator,
            site=name + "_Mz",
            range=[-3.0, 3.0],
            gear=np.array([0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        )
        return chassis, actuator

    def create_cable_payload(self, parent, name, pos, length=1, mass=0.15):
        cable = self.body(parent, name, pos)
        cable_joint = self.joint(
            cable,
            "cable_quad_joint",
            "ball",
            pos=np.array([0.0, 0.0, 0.5 * length]),
            damping=0.01,
        )
        cable_geom = self.cylinder(
            cable,
            "cable",
            radius=0.005,
            length=length,
            mass=0.1,
            density=50,
            rgba=[0.2, 0.2, 0.2, 1.0],
        )
        pyld = self.body(cable,
                         "pyld",
                         pos=np.array([0.0, 0.0, -0.5 * length]))
        pyld_geom = self.sphere(pyld,
                                "pyld",
                                radius=0.05,
                                mass=mass,
                                density=50,
                                rgba=[0.0, 0.8, 0.2, 1.0])
        pyld_joint = self.joint(
            pyld,
            "pyld_joint",
            "hinge",
            pos=np.array([0.0, 0.0, 0]),
            range=np.array([-0.001, 0.001]),
            stiffness=1000,
            damping=1000,
            armature=0.0,
        )
        return cable

    def create_flexible_cable_payload(self,
                                      parent,
                                      name,
                                      pos,
                                      N=5,
                                      length=1,
                                      mass=0.15):
        dl = length / N
        cable_mass = 0.01
        dm = cable_mass / N
        cable = self.body(parent, name + "_0", np.array([0.0, 0.0, -0.5 * dl]))
        cable_joint = self.joint(
            cable,
            "cable_quad_joint_0",
            "ball",
            pos=np.array([0.0, 0.0, 0.5 * dl]),
            damping=1e-3,
            stiffness=0,
        )
        cable_geom = self.cylinder(
            cable,
            "cable_cylinder_0",
            radius=0.005,
            length=dl,
            mass=dm,
            density=50,
            rgba=[0.2, 0.2, 0.2, 1.0],
        )
        for i in range(1, N):
            cable = self.body(cable, name + "_" + str(i),
                              np.array([0.0, 0.0, -dl]))
            cable_joint = self.joint(
                cable,
                "cable_quad_joint_" + str(i),
                "ball",
                pos=np.array([0.0, 0.0, 0.5 * dl]),
                damping=1e-3,
                stiffness=0.0,
            )
            cable_geom = self.cylinder(
                cable,
                "cable_cylinder_" + str(i),
                radius=0.005,
                length=dl,
                mass=dm,
                density=50,
                rgba=[0.2, 0.2, 0.2, 1.0],
            )
        pyld = self.body(cable, "pyld", pos=np.array([0.0, 0.0, -0.5 * dl]))
        pyld_geom = self.sphere(pyld,
                                "pyld",
                                radius=0.05,
                                mass=mass,
                                density=50,
                                rgba=[0.0, 0.8, 0.2, 1.0])
        pyld_joint = self.joint(
            pyld,
            "pyld_joint",
            "ball",
            pos=np.array([0.0, 0.0, 0]),
            range=np.array([-0.001, 0.001]),
            stiffness=1000,
            damping=1000,
            armature=0.0,
        )
        return cable
