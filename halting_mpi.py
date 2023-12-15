import os
import gc
import time
import uuid
import torch
import numpy as np
import random
import mpi4py
mpi4py.rc.recv_mprobe = False
import argparse
import subprocess
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from math import pi
from mpi4py import MPI
from matplotlib.patches import Rectangle
from scipy.ndimage.filters import gaussian_filter
from episode.policy.policy import ActorCritic
from rl.sac import ReplayBuffer, SAC
from torch.utils.tensorboard import SummaryWriter

def convert_sec(sec):
    mm, ss = divmod(sec, 60)
    return f'{mm:02.0f}m {ss:02.0f}s'

def launch_simulation(ID, args):
    # Initiate ROS-Gazebo simulation
    for n_restart in range(10): # Max restart 10 times
        os.system(f"singularity instance start --net --network=none {os.getenv('CONTAINER')} {ID} {args} > /dev/null 2>&1")
        time.sleep(5.0)
        test_proc = subprocess.Popen(["singularity", "run", f"instance://{ID}", "/wait_until_stable"])
        try:
            test_proc.wait( timeout=60.0 )
        except subprocess.TimeoutExpired as e:
            print(f"Restarting {ID}...", flush=True)
            os.system(f"singularity instance stop {ID} > /dev/null 2>&1")
        else:
            break
    assert n_restart < 10, f"Some simulation failed to launched after 10 trials. Stop training process."

if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_system", type=str, nargs='+', choices=["eband", "dwa", "teb"], required=True)
    parser.add_argument("--total_sample", type=int, required=True)
    parser.add_argument("--train_iter", type=int, default=50)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--detection_range", type=float, default=8.0)
    parser.add_argument("--n_sample", type=int, default=800)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--reward_constant", type=float, default=60.0, help="Constant to make successful episode reward positive.")
    args = parser.parse_args()

    # Initialize MPI
    ROOT = 0
    ID    = uuid.uuid4()
    jobID = os.getenv("SLURM_JOBID")

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    tasks_per_node = float(os.getenv('IBRUN_TASKS_PER_NODE'))   # MUST BE 19 simulators per node
    num_node = int(size / tasks_per_node)
    local_rank = int(os.getenv('MPI_LOCALRANKID') or os.getenv('MV2_COMM_WORLD_LOCAL_RANK'))

    # Define directories
    WORK_DIRECTORY  = f"{os.getenv('WORK')}/results"
    LOG_DIRECTORY   = f"{WORK_DIRECTORY}/runs/halting/{jobID}"
    MODEL_DIRECTORY = f"{WORK_DIRECTORY}/checkpoints/halting/{jobID}"
    DATA_DIRECTORY  = f"{WORK_DIRECTORY}/data/halting/{jobID}"
    TRAIN_STORAGE   = f"{DATA_DIRECTORY}/train.pt"

    # Create directory if not exists
    if not os.path.exists(LOG_DIRECTORY):
        os.makedirs(LOG_DIRECTORY)
    if not os.path.exists(MODEL_DIRECTORY):
        os.makedirs(MODEL_DIRECTORY)
    if not os.path.exists(DATA_DIRECTORY):
        os.makedirs(DATA_DIRECTORY)

    # Log hyper-parameters to tensorboard
    logger = (SummaryWriter(log_dir=LOG_DIRECTORY) if rank==ROOT else None)
    if rank == ROOT:
        param = f"Total samples: {args.total_sample}\nGPR train iteration: {args.train_iter}\n" \
                f"epsilon: {args.epsilon}\nnumber of waypoint sampling: {args.n_sample}" \
                f"Base navigation system: {', '.join(args.base_system)}"\
                f"reward: -SUM(TTD) - 60*I_collision + {args.reward_constant:.1f}"
        logger.add_text('hyper-parameters', param)

    # Launch simulation with base navigation system
    system_setup = (
        ('eband', 'eband'), ('dwa',   'dwa'), ('teb', 'teb'),
        ('eband', 'dwa'  ), ('eband', 'teb'), ('dwa', 'teb')
    )
    time.sleep(local_rank/tasks_per_node)
    if local_rank == ROOT:
        roslaunch_argument = f"local_planner1:={args.base_system[0]} local_planner2:={args.base_system[1]}"
    else:
        nav_system1, nav_system2 = system_setup[local_rank % 6]
        roslaunch_argument = f"local_planner1:={nav_system1} local_planner2:={nav_system2}"
    launch_simulation(ID, roslaunch_argument)
    comm.Barrier()

    train_x = torch.zeros((args.total_sample,4), dtype=torch.float32)
    train_y = torch.zeros((args.total_sample, ), dtype=torch.float32)
    for i in range(0, args.total_sample, num_node):
        # save train data as .pt file
        if rank == ROOT:    # if i==0, use dummy data!
            torch.save(dict(features=train_x[:max(i,1),:], observations=train_y[:max(i,1)]), TRAIN_STORAGE)
            time.sleep(1.0) # Make sure train data is saved as file
        comm.Barrier()

        # Run episode to collect data
        epsilon = (args.epsilon if local_rank==ROOT else 0.0)
        detection_range = args.detection_range # + (0. if local_rank==ROOT else random.uniform(-2., 2.))

        time.sleep(local_rank/tasks_per_node)
        ep_proc = subprocess.Popen(["singularity", "run", f"instance://{ID}", "python3", "episode/halting_episode.py",
                                    "--train_data_storage", f"{TRAIN_STORAGE}", "--train_iter", f"{args.train_iter}", "--epsilon", f"{epsilon}",
                                    "--detection_range", f"{detection_range}", "--n_sample", f"{args.n_sample}",
                                    "--timeout", "200.0", "--reward_constant", f"{args.reward_constant}",
                                    "--result_data_storage", f"/tmp/{ID}.pt"], 
                                    stderr=subprocess.DEVNULL)
        time.sleep(5.0)
        ep_proc.wait()
        del ep_proc
        gc.collect()

        # Load episode result from file
        data = torch.load(f"/tmp/{ID}.pt")

logger.add_scalar("Episode reward", ep_rew, idx+size*ep_round)

            # Load episode result from file
            data = torch.load(f"/tmp/{ID}.pt")
            s, a, ns, r, d, traj = list(map(data.get, ["state", "action", "next_state", "reward", "done", "trajectory1"]))
            ep_len = replay.store(state=s, action=a, next_state=ns, reward=r, done=d)
            ep_steps = comm.allreduce( ep_len, op=MPI.SUM )

            # Store remain distance-reward pair
            dist_reward += [[np.linalg.norm(traj[1:,:2] - traj[:-1,:2], axis=1).sum(), r.sum()]]

            # Log episode reward to tensorboard
            ep_rew_buf = (torch.empty(size) if rank==ROOT else None)
            comm.Gather(r.sum(), ep_rew_buf, root=ROOT)
            if rank == ROOT:
                for idx, ep_rew in enumerate(ep_rew_buf):
                    logger.add_scalar("Episode reward", ep_rew, idx+size*ep_round)
                ep_round += 1

            if total_steps > args.update_after:
                start_time = time.time()
                info = np.empty((ep_steps, 2), dtype=np.float32)
                for k in range(ep_steps):
                    q_info, pi_info, t_comm = sac.update_mpi(batch=replay.sample_batch(LOCAL_BATCH_SIZE), comm=comm)
                    t_network += t_comm
                    # log q1 value to tensorboard
                    info[k,0] = q_info["LossQ"]
                    info[k,1] = pi_info["LossPi"]
                t_train += time.time() - start_time

                # Now log information to tensorboard
                avg_info = (np.zeros((size, ep_steps, 2), dtype=np.float32) if rank==ROOT else None)
                comm.Gather(info, avg_info, root=ROOT)
                if rank == ROOT:
                    avg_info = avg_info.mean(axis=0)
                    for k in range(ep_steps):
                        logger.add_scalar("Loss/Q", info[k,1], total_steps+k)
                        logger.add_scalar("Loss/PI", info[k,1], total_steps+k)
            step += ep_steps
            total_steps += ep_steps

    if rank == ROOT:
        logger.close()
    os.system(f"singularity instance stop {ID} > /dev/null 2>&1")