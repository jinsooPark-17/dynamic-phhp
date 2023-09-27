import os
import time
import uuid
import numpy as np
import pandas as pd
import mpi4py
mpi4py.rc.recv_mprobe = False
import subprocess
from mpi4py import MPI

if __name__=='__main__':
    ID = uuid.uuid4()
    n_test = 30

    # Establish MPI connections first
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    comm.Barrier()
    time.sleep(1.0)

    available=True
    for n_restart in range(10): # Max restart 10 times
        os.system(f"singularity instance start {os.getenv('CONTAINER')} {ID} > /dev/null 2>&1")
        time.sleep(5.0)
        test_proc = subprocess.Popen(["singularity", "run", f"instance://{ID}", "/wait_until_stable"])
        try:
            test_proc.wait( timeout=60.0 )
        except subprocess.TimeoutExpired as e:
            print(f"Restarting {ID}...", flush=True)
            os.system(f"singularity instance stop {ID} > /dev/null 2>&1")
        else:
            break
    else:
        available=False
    max_restart = comm.reduce(n_restart, op=MPI.MAX, root=0)
    n_available = comm.reduce(available, op=MPI.SUM, root=0)
    if rank == 0:
        print(f"Running {n_available}/{size} simulations...")
        print(f"Maximum number of restart was {max_restart}.", flush=True)

    # Run episode
    storage = f"/tmp/{ID}.result"
    ep_proc = subprocess.Popen(["singularity", "run", f"instance://{ID}", 
                                "python3", "episode/measure_precision.py", storage, f"{n_test}"])
    ep_proc.wait()
    comm.Barrier()
    os.system(f"singularity instance stop {ID} > /dev/null 2>&1")

    # Read data from file storage
    with open(storage, 'rb') as f:
        data = np.load(f)
    os.remove(storage)

    # Collect data through MPI
    recv_arr = None
    if rank==0: recv_arr = np.empty((size, *data.shape))
    comm.Gather(data, recv_arr, root=0)

    # Save data as csv file
    if rank==0:
        if not os.path.exists(f"{os.getenv('WORK')}/data"):
            os.makedirs(f"{os.getenv('WORK')}/data")
        df = pd.DataFrame(recv_arr.reshape(-1,data.shape[-1]), columns=["travel_dist", "ttd", "distance_error", "angle_error"])
        df.to_csv(f"{os.getenv('WORK')}/data/movebase_precision_result.{os.getenv('SLURM_JOB_ID')}.csv")

        print(f"MoveBase precision experiment result with {size*n_test*2} episodes")
        print(df.mean(axis=0))
