import numpy as np

frequency = 100
output_path = 'home/rcir/Dev/force_prediction/traj/circle.txt'

# Parameters
sample_time = 1/frequency      # seconds
duration = 20                  # seconds

r = 3                          # m
v = 2                          # m/s

# Circle Center
x0 = 0.5                
y0 = 0.5
z0 = 1

# Trajectory
traj = np.zeros((int(duration/sample_time+1),14)) #x y z u v w du dv dw psi
t = np.arange(0,duration,sample_time)
t = np.append(t, duration)

traj[:,0] = -r*np.cos(t*v/r)+x0
traj[:,1] = -r*np.sin(t*v/r)+y0
traj[:,2] = z0
traj[:,7] = v*np.sin(t*v/r)
traj[:,8] = -v*np.cos(t*v/r)
traj[:,5] = 0
traj[:,6] = v*v/r*np.cos(t*v/r)
traj[:,7] = v*v/r*np.sin(t*v/r)
traj[:,8] = 0
traj[:,9] = np.arctan2(np.sin(t*v/r),np.cos(t*v/r))

# write to txt
np.savetxt(output_path,traj,fmt='%_f')
print("circle.txt updated!")