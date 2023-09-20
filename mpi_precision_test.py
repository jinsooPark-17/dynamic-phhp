#!/usr/bin/env python3
import os
import time
import uuid
import numpy as np
import mpi4py
mpi4py.rc.recv_mprobe = False
import argparse
import subprocess
from mpi4py import MPI
from socket import gethostname

from collections import namedtuple
from envs.robots import AllinOne
from envs.simulation import I_Shaped_Hallway

import rospy
Pose = namedtuple("Pose", "x y yaw")
if __name__=="__main__":
    # setup argument parser
    parser = argparse.ArgumentParser(description="Test precision of move_base")
    parser.add_argument('--num_test', type=int, help="Total number of episodes to run")
    args = parser.parse_args()

    # Initialize MPI process first
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    comm.Barrier()  # wait for ALL mpi nodes to initialized

    # Setup system environment
    hostname    = gethostname()
    LOCAL_RANK  = int(os.getenv('MV2_COMM_WORLD_LOCAL_RANK'))
    ros_port    = 11311 + 20 * LOCAL_RANK
    gazebo_port = 11321 + 20 * LOCAL_RANK
    LOG_DIR     = uuid.uuid4().__str__()
    os.environ["ROS_LOG_DIR"]       = f"{os.getenv('SCRATCH')}/roslog/{LOG_DIR}"
    os.environ["ROS_HOSTNAME"]      = hostname
    os.environ["ROS_MASTER_URI"]    = f"http://{hostname}:{ros_port}"
    os.environ["GAZEBO_MASTER_URI"] = f"http://{hostname}:{gazebo_port}"
    print(f"Connecting to {os.getenv('ROS_MASTER_URI')}", flush=True)
    # launch simulation
    time.sleep( (rank // 12) * 5 ) # distribute jobs
    start_time = time.time()
    subprocess.Popen(
        ["roslaunch", "bwi_launch", "two_robot_simulation.launch", "--screen", "-p", f"{ros_port}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(30.0)    # wait for all roslaunch to started

    # define environment
    MPI_ID = f"[{gethostname().split('.')[0]}-{rank:03d}]"
    try:
        env = I_Shaped_Hallway(debug=MPI_ID)
        init_poses = [Pose(-10.0, 0.0, 0.0), Pose(0.0,   0.0, np.pi/2.)]
        goal_poses = [Pose(  0.0, 0.0, 0.0), Pose(0.0, -10.0, np.pi/2.)]
        valid = True
    except rospy.ROSInitException as e:
        valid = False
    
    # define number of episodes per MPI job
    n_available = comm.reduce(valid, op=MPI.SUM, root=0)
    if rank == 0:
        start_time = time.time()
        N_EPISODES_PER_JOB  = int(args.num_test / n_available)
        if not os.path.exists("data.csv"):
            with open("data.csv", 'w') as f:
                f.write("ttd,r,theta\n")
    else:
        N_EPISODES_PER_JOB = None
    N_EPISODES_PER_JOB = comm.bcast(N_EPISODES_PER_JOB, root=0)
    
    # Now, run episode
    sendbuf = np.zeros((N_EPISODES_PER_JOB*2, 3), dtype=np.float32)
    if valid is True:
        for i in range(N_EPISODES_PER_JOB):
            result = env.test_precision(init_poses, goal_poses, timeout=30.0)
            sendbuf[2*i:2*i+2] = result
        print(f"{MPI_ID}: Average TTD of {N_EPISODES_PER_JOB} single robot episodes: {sendbuf.mean()[0]}")
    else:
        print(f"{MPI_ID}: Simulator failed.")

    # Collect data from rank 0 MPI job
    if rank == 0:
        recvbuf = np.empty([size, N_EPISODES_PER_JOB*2], np.float32)
    comm.Gather(sendbuf, recvbuf, root=0)
    if rank == 0:
        # filter out zero row
        idx = (np.linalg.norm(recvbuf, axis=1) == 0)
        recvbuf = recvbuf[idx].reshape(-1, 3)
        # store data to data.csv file
        import pandas as pd
        df = pd.DataFrame(recvbuf)
        df.to_csv("data.csv", mode='a', index=False, header=False)
        print(f"\n\n{np.argwhere(idx is True)} simulations does not work.", flush=True)
        print(f"MPI job took total {time.time() - start_time:.2f} seconds.", flush=True)
        print(f"Average TTD of single episode: {recvbuf.mean()[0]:.2f} seconds", flush=True)

    os.system(f"rm -r $SCRATCH/roslog/{LOG_DIR}")