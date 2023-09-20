#!/usr/bin/env python3
import os
import time
import uuid
import mpi4py
mpi4py.rc.recv_mprobe = False
import subprocess
from mpi4py import MPI
from socket import gethostname

from collections import namedtuple
from envs.robots import AllinOne
from envs.simulation import L_Hallway_Single_robot

import rospy
Pose = namedtuple("Pose", "x y yaw")
if __name__=="__main__":
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
    env = L_Hallway_Single_robot(ep_timeout=30.0)
    robot1 = AllinOne(id="marvin")
    env.register_robots(robot1=robot1)
    init_pose = Pose(-10.0, 0.0, 0.0)
    valid = robot1.connected
    print(f"[{gethostname().split('.')[0]}-{rank:03d}] took {time.time()-start_time:.3f} sec until ROS reaches {rospy.Time.now().to_sec()}", flush=True)
    comm.Barrier()

    # Now, run episode
    n_test = 10
    if valid is True:
        start_time = time.time()
        ttd = [0.] * n_test
        for i in range(n_test):
            ttd[i] = env.begin([init_pose], [Pose(0., 0., 0.)])
            print(f"[{gethostname().split('.')[0]}-{rank:03d}]: {i+1} episodes took {time.time()-start_time:.2f} s", flush=True)
    else:
        ttd = [0.] * n_test
        print(f"Simulator {rank} failed.")

    n_valid = comm.reduce(valid, op=MPI.SUM, root=0)
    avg_ep_ttd = comm.reduce(sum(ttd)/n_test, op=MPI.SUM, root=0)
    if rank == 0:
        print(f"\n\nAverage TTD of single episode: {avg_ep_ttd/n_valid:.2f} seconds", flush=True)

    os.system(f"rm -r $SCRATCH/roslog/{LOG_DIR}")