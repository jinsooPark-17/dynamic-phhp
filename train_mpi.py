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

    X = torch.ones(1, 2*640+3)
    # Share policy 1
    import time
    start_time = time.time()
    if rank == 0:
        model = ActorCritic( n_scan=2 )
    else:
        model = None
    model = comm.bcast(model, root=0)
    t_bcast = time.time() - start_time
    print(f"RANK {rank:02d}: {model.act(X, deterministic=True)}", flush=True)

    # Share policy 2
    comm.Barrier()
    time.sleep(1.0)
    start_time = time.time()
    model = ActorCritic( n_scan=2 )
    if rank == 0:
        print("\n")
        torch.save(model.pi.state_dict(), "test/pi")
        torch.save(model.q1.state_dict(), "test/q1")
        torch.save(model.q2.state_dict(), "test/q2")
    comm.Barrier()
    model.pi.load_state_dict( torch.load("test/pi") )
    model.q1.load_state_dict( torch.load("test/q1") )
    model.q2.load_state_dict( torch.load("test/q2") )
    t_load = time.time() - start_time
    print(f"RANK {rank:04d}: {model.act(X, deterministic=True)}", flush=True)

    t_bcast_max = comm.reduce(t_bcast, op=MPI.MAX, root=0)
    t_load_max = comm.reduce(t_load, op=MPI.MAX, root=0)
    if rank == 0:
        time.sleep(1.0)
        print(f"\tShare policy with bcast took             {t_bcast_max:.2f} seconds")
        print(f"\tShare policy with network directory took {t_load_max:.2f} seconds")
