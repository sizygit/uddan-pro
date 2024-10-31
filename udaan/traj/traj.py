import matplotlib.pyplot as plt
import numpy as np

def trajectory_generator(r, v, tf, thrust, x0=0, y0=0, shape=1, show_traj=False):
    '''
    Generates a circular trajectory given a final time and a sampling time 
    '''
    frequency = 100
    sample_time = 1/frequency      # seconds
    duration = tf                  # seconds
    traj = np.zeros((int(duration/sample_time+1),14)) #x y z u v w
    t = np.arange(0,duration,sample_time)
    t = np.append(t, duration)
    z0 = 3

    ## circle trajectory
    if shape == 0: 
        traj[:,0] = r*np.cos(t*v/r)+x0
        traj[:,1] = r*np.sin(t*v/r)+y0
        traj[:,2] = z0
        traj[:,3] = 1
        traj[:,7] = -v*np.sin(t*v/r)
        traj[:,8] = v*np.cos(t*v/r)
        traj[:,10] = thrust
        traj[:,11] = -v/r*v*np.cos(t*v/r)
        traj[:,12] = -v/r*v*np.sin(t*v/r)

        if show_traj == True:
            _,((ax1),(ax2)) = plt.subplots(2,1)
            ax1.plot(traj[:,0],traj[:,1])
            ax2.plot(traj[:,7],traj[:,8])
            plt.show()   

    # lemniscate trajectory
    if shape == 1: 
        amp = r
        frq = v/r
        traj[:,0] = amp*np.cos(t*frq)+x0
        traj[:,1] = amp*np.sin(t*frq)*np.cos(t*frq)+y0
        traj[:,2] = z0
        traj[:,3] = 1
        traj[:,7] = -amp*frq*np.sin(t*frq)
        traj[:,8] = amp*frq*np.cos(t*2*frq)
        traj[:,9] = 0
        traj[:,10] = thrust

        if show_traj == True:
            _,((ax1),(ax2)) = plt.subplots(2,1)
            ax1.plot(traj[:,0],traj[:,1])
            ax2.plot(traj[:,7],traj[:,8])
            plt.show()

    # line trajectory
    if shape == 2: 
        amp = r
        frq = v / r 
        traj[:,0] = v * t + r
        traj[:,1] = y0
        traj[:,2] = z0
        traj[:,3] = 1
        traj[:,7] = v
        traj[:,8] = 0
        traj[:,9] = 0
        traj[:,10] = thrust

        if show_traj == True:
            _,((ax1),(ax2)) = plt.subplots(2,1)
            ax1.plot(traj[:,0],traj[:,1])
            ax2.plot(traj[:,7],traj[:,8])
            plt.show()

    ## polynomial trajectory
    if shape == 3: 

        if show_traj == True:
            _,((ax1),(ax2)) = plt.subplots(2,1)
            ax1.plot(traj[:,0],traj[:,1])
            ax2.plot(traj[:,7],traj[:,8])
            plt.show()
    return traj

def multi_trajectory_generator(x, y, v, tf, thrust, x0=0, y0=0, shape=2, show_traj=False):
    '''
    Generates a circular trajectory given a final time and a sampling time 
    '''
    frequency = 100
    sample_time = 1/frequency      # seconds
    duration = tf                  # seconds
    traj = np.zeros((int(duration/sample_time+1),14)) #x y z u v w
    t = np.arange(0,duration,sample_time)
    t = np.append(t, duration)
    z0 = 3

    ## circle trajectory
    if shape == 0: 
        traj[:,0] = x*np.cos(t*v/x)+x0
        traj[:,1] = x*np.sin(t*v/x)+y0
        traj[:,2] = z0
        traj[:,3] = 1
        traj[:,7] = -v*np.sin(t*v/x)
        traj[:,8] = v*np.cos(t*v/x)
        traj[:,10] = thrust

        if show_traj == True:
            _,((ax1),(ax2)) = plt.subplots(2,1)
            ax1.plot(traj[:,0],traj[:,1])
            ax2.plot(traj[:,7],traj[:,8])
            plt.show()   

    ## lemniscate trajectory
    if shape == 1: 
        amp = x
        frq = v/x
        traj[:,0] = amp*np.cos(t*frq)+x0
        traj[:,1] = amp*np.sin(t*frq)*np.cos(t*frq)+y0
        traj[:,2] = z0
        traj[:,3] = 1
        traj[:,7] = -amp*frq*np.sin(t*frq)
        traj[:,8] = amp*frq*np.cos(t*2*frq)
        traj[:,9] = 0
        traj[:,10] = thrust

        if show_traj == True:
            _,((ax1),(ax2)) = plt.subplots(2,1)
            ax1.plot(traj[:,0],traj[:,1])
            ax2.plot(traj[:,7],traj[:,8])
            plt.show()

    ## line trajectory
    if shape == 2: 
        traj[:,0] = v * t + x
        traj[:,1] = y
        traj[:,2] = z0
        traj[:,3] = 1
        traj[:,7] = v
        traj[:,8] = 0
        traj[:,9] = 0
        traj[:,10] = thrust

        if show_traj == True:
            _,((ax1),(ax2)) = plt.subplots(2,1)
            ax1.plot(traj[:,0],traj[:,1])
            ax2.plot(traj[:,7],traj[:,8])
            plt.show()

    ## polynomial trajectory
    if shape == 3: 

        if show_traj == True:
            _,((ax1),(ax2)) = plt.subplots(2,1)
            ax1.plot(traj[:,0],traj[:,1])
            ax2.plot(traj[:,7],traj[:,8])
            plt.show()
    return traj