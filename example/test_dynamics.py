import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir,'..'))
sys.path.append(parent_dir)

import udaan as U
import numpy as np

# base models
# mdl = U.models.base.Quadrotor(render=True)
# mdl.simulate(tf=10, x0=np.array([1., 1., 0.]))
# mujoco models
# mdl = U.models.mujoco.Quadrotor(render=True)
# mdl.simulate(tf=10, position=np.array([1., 1., 0.]))

# mdl = U.models.mujoco.Quadrotor(render=True, force="prop_forces", input="wrench")
# mdl.simulate(tf=10, position=np.array([-1.0, 2.0, 0.0]))

# mdl = U.models.base.QuadrotorCSPayload(render=True)
# mdl.simulate(tf=10, position=np.array([-1., 2., 0.]))

mdl = U.models.mujoco.QuadrotorCSPayload(render=True, model="tendon")
np.random.seed(0)
random_array = np.random.uniform(low=1.0, high=3.0, size=(3,))
random_array[0] = 4
random_array[1] = 0
random_array[2] = 2
mdl.simulate(tf=4, r=4, payload_position= random_array)

# mdl = U.models.mujoco.MultiQuadrotorCSPointmass(render=True, num_quadrotors=2)
# random_array = np.random.uniform(low=1.0, high=3.0, size=(3,))
# random_array[0] = 4
# random_array[1] = 0
# mdl.simulate(tf=10, payload_position=np.array([-1., 2., 0.5]))
