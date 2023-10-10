import os
import torch
import mpi4py
mpi4py.rc.recv_mprobe = False
import argparse

from mpi4py import MPI
from episode.policy.policy import ActorCritic
from rl.sac import ReplayBuffer, SAC

if __name__ == "__main__":
    # Define shared storage
    jobID = os.environ["SLURM_JOBID"]

    # Initialize MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Share policy
    if rank == 0:
        model = ActorCritic( n_scan=2 )
    else:
        model = None
    model = comm.bcast(model, root=0)

    x = torch.ones(1, 2*640+3)
    print(f"Rank {rank:02d}: {model(x)[0].squeeze()}")
