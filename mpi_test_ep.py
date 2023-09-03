#!/usr/bin/env python3
import mpi4py
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI
from socket import gethostname
from collections import namedtuple
from envs.robots import Vanilla
from envs.simulation import L_Hallway_Single_robot
import numpy as np
import matplotlib.pyplot as plt

Pose = namedtuple("Pose", "x y yaw")

if __name__ == "__main__":
    # Initialize MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    # Choose robot
    env = L_Hallway_Single_robot(ep_timeout=30.0)
    idx = np.random.choice(2)
    name, init_pose = [["marvin", Pose(-10,0,0)], ["rob", Pose(0, -10, np.pi/2.)]][idx]
    robot1 = Vanilla( name )
    env.register_robots(robot1=robot1)

    # Print total number of available simulations
    valid = robot1.connected
    n_valid = comm.reduce(valid, op=MPI.SUM, root=0)
    comm.Barrier()
    if rank == 0:
        print(f"Running episode with {n_valid}/{size} simulators")
    
    # Now, run episode
    ttd = env.begin([init_pose], [Pose(0., 0., np.pi/4.)])
    if type(ttd) is float:
        print(f"[{gethostname()}-{rank:02d}] Episode TTD: {ttd:.2f} s")
    else:
        print(f"Simulator {rank+1} failed.")

    avg_ep_ttd = comm.reduce(ttd, op=MPI.SUM, root=0)

    if rank == 0:
        avg_ep_ttd = avg_ep_ttd / float(n_valid)
        print(f"\nAverage TTD of single episode: {avg_ep_ttd:.3f} seconds")