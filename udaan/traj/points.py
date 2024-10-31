#!/usr/bin/env python3
import os
import numpy as np

frequency = 100

package_path = os.path.dirname(os.path.abspath(__file__))
output_path = package_path + '/scripts/points.txt'

# Parameters
sample_time = 1/frequency      # seconds
cycles = 2
step_interval = 5

points_matrix = np.array([[0.0,0.0,1.0,-2.4],[2.0,0.0,1.0,2.4]])

# Trajectory
duration = cycles*np.size(points_matrix,0)*step_interval

traj = np.zeros((int(duration/sample_time+1),np.size(points_matrix,1)))

t = np.arange(0,duration,sample_time)
t = np.append(t, duration)

for i in range(1,cycles+1):
    for j in range(1,np.size(points_matrix,0)+1):
        traj_start = (i-1)*np.size(points_matrix,0)*step_interval+(j-1)*step_interval
        traj_end = (i-1)*np.size(points_matrix,0)*step_interval+j*step_interval
        traj[int(traj_start/sample_time):int(traj_end/sample_time),0:np.size(points_matrix,1)] = np.tile(points_matrix[j-1,:],(int(step_interval/sample_time),1))
traj[-1,0:np.size(points_matrix,1)] = traj[-2,0:np.size(points_matrix,1)]

# Write to txt
np.savetxt(output_path,traj,fmt='%f')
print("points.txt updated!")