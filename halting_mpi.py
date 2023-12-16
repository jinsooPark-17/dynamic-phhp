import os
import gc
import time
import uuid
import torch
import numpy as np
import mpi4py
import pandas as pd
mpi4py.rc.recv_mprobe = False
import argparse
import subprocess
from mpi4py import MPI
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
    parser.add_argument("--timeout", type=float, default=200.0)
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
    CSV_STORAGE     = f"{DATA_DIRECTORY}/result.csv"

    # Create directory if not exists
    if rank==ROOT:
        if not os.path.exists(LOG_DIRECTORY):
            os.makedirs(LOG_DIRECTORY)
        if not os.path.exists(MODEL_DIRECTORY):
            os.makedirs(MODEL_DIRECTORY)
        if not os.path.exists(DATA_DIRECTORY):
            os.makedirs(DATA_DIRECTORY)
    comm.Barrier()

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
                                    "--timeout", f"{args.timeout}", "--reward_constant", f"{args.reward_constant}",
                                    "--result_data_storage", f"/tmp/{ID}.pt"], 
                                    stderr=subprocess.DEVNULL)
        time.sleep(5.0)
        ep_proc.wait()
        del ep_proc
        gc.collect()

        # Load episode result from file
        data_dict = torch.load(f"/tmp/{ID}.pt")
        data = np.zeros(23, dtype=np.float32)
        data[0] = (0.0 if data_dict['mode'] == 'exploit' else 1.0)
        data[1:5] = data_dict['features']
        data[5] = data_dict['reward']
        data[6:8] = data_dict['p_bold']
        data[8:10] = data_dict['p_polite']
        data[10:12] = data_dict['waypoint']
        data[12] = data_dict['success_polite']
        data[13] = data_dict['ttd_polite']
        data[14] = data_dict['halt_time']
        data[15] = data_dict['success_bold']
        data[16] = data_dict['ttd_bold']
        data[17] = data_dict['computation_time']
        data[18] = data_dict['waypoint_mean'].detach().item()
        data[19] = data_dict['waypoint_std'].detach().item()
        data[20] = data_dict['waypoint_sampled'].item()
        data[21] = (1.0 if local_rank==ROOT else 0.0)
        data[22] = (local_rank % 6)

        # Collect data through MPI channel
        if rank==ROOT:
            data_buf = np.empty((size,data.shape[0]), dtype=np.float32)
        comm.Gather(data, data_buf, root=ROOT)

        if rank==ROOT:
            train_idx = np.where(data_buf[:,21]>0.5, True, False)
            train_x[i+num_node,:] = torch.from_numpy(data_buf[train_idx, 1:5]).float()
            train_y[i+num_node] = data_buf[train_idx, 5]

            # save new episodic data to csv file
            df = pd.DataFrame(data_buf.numpy(), columns=[
                'mode', 'x1', 'x2', 'x3', 'x4', 'reward',
                'x_bold', 'y_bold', 'x_polite', 'y_polite', 'x_waypoint', 'y_waypoint',
                'success_polite', 'TTD_polite', 'TTD_halt', 'success_bold', 'TTD_bold',
                't_compute', 'waypoint_mean', 'waypoint_std', 'waypoint_sampled', 'is_train_sample', 'env_setup'
            ])
            if not os.path.exists(CSV_STORAGE):
                df.to_csv(CSV_STORAGE, index=False, mode='w', header=True)
            else:
                df.to_csv(CSV_STORAGE, index=False, mode='a', header=False)

            # Log to tensorboard
            train_samples = data_buf[train_idx, :]

            ## train info
            logger.add_scalar("computation_time", data_buf[:,17].mean().item(), i)
            for k, (reward, ttd_bold, ttd_polite, ttd_halt) in enumerate(train_samples[:,[5,16,13,14]]):
                logger.add_scalar(f"TRAIN/reward/{'-'.join(args.base_system)}", reward, i+k)
                logger.add_scalar(f"TRAIN/TTD/{'-'.join(args.base_system)}", {"TTD_bold": ttd_bold,
                                                                              "TTD_polite": ttd_polite,
                                                                              "TTD_halt": ttd_halt}, i+k)
            ## test info
            eval_samples = data_buf[~train_idx, :]
            logger.add_scalars("TEST/reward", {'-'.join(system_setup[0]): eval_samples[np.where(eval_samples[:,22]==0, True, False), 5].mean().item(),
                                               '-'.join(system_setup[1]): eval_samples[np.where(eval_samples[:,22]==1, True, False), 5].mean().item(),
                                               '-'.join(system_setup[2]): eval_samples[np.where(eval_samples[:,22]==2, True, False), 5].mean().item(),
                                               '-'.join(system_setup[3]): eval_samples[np.where(eval_samples[:,22]==3, True, False), 5].mean().item(),
                                               '-'.join(system_setup[4]): eval_samples[np.where(eval_samples[:,22]==4, True, False), 5].mean().item(),
                                               '-'.join(system_setup[5]): eval_samples[np.where(eval_samples[:,22]==5, True, False), 5].mean().item()})
            logger.add_scalars("TEST/TTD/polite", {'-'.join(system_setup[0]): eval_samples[np.where(eval_samples[:,22]==0, True, False), 13].mean().item(),
                                                   '-'.join(system_setup[1]): eval_samples[np.where(eval_samples[:,22]==1, True, False), 13].mean().item(),
                                                   '-'.join(system_setup[2]): eval_samples[np.where(eval_samples[:,22]==2, True, False), 13].mean().item(),
                                                   '-'.join(system_setup[3]): eval_samples[np.where(eval_samples[:,22]==3, True, False), 13].mean().item(),
                                                   '-'.join(system_setup[4]): eval_samples[np.where(eval_samples[:,22]==4, True, False), 13].mean().item(),
                                                   '-'.join(system_setup[5]): eval_samples[np.where(eval_samples[:,22]==5, True, False), 13].mean().item()})
            logger.add_scalars("TEST/TTD/halt", {'-'.join(system_setup[0]): eval_samples[np.where(eval_samples[:,22]==0, True, False), 14].mean().item(),
                                                 '-'.join(system_setup[1]): eval_samples[np.where(eval_samples[:,22]==1, True, False), 14].mean().item(),
                                                 '-'.join(system_setup[2]): eval_samples[np.where(eval_samples[:,22]==2, True, False), 14].mean().item(),
                                                 '-'.join(system_setup[3]): eval_samples[np.where(eval_samples[:,22]==3, True, False), 14].mean().item(),
                                                 '-'.join(system_setup[4]): eval_samples[np.where(eval_samples[:,22]==4, True, False), 14].mean().item(),
                                                 '-'.join(system_setup[5]): eval_samples[np.where(eval_samples[:,22]==5, True, False), 14].mean().item()})
            logger.add_scalars("TEST/TTD/bold", {'-'.join(system_setup[0]): eval_samples[np.where(eval_samples[:,22]==0, True, False), 16].mean().item(),
                                                 '-'.join(system_setup[1]): eval_samples[np.where(eval_samples[:,22]==1, True, False), 16].mean().item(),
                                                 '-'.join(system_setup[2]): eval_samples[np.where(eval_samples[:,22]==2, True, False), 16].mean().item(),
                                                 '-'.join(system_setup[3]): eval_samples[np.where(eval_samples[:,22]==3, True, False), 16].mean().item(),
                                                 '-'.join(system_setup[4]): eval_samples[np.where(eval_samples[:,22]==4, True, False), 16].mean().item(),
                                                 '-'.join(system_setup[5]): eval_samples[np.where(eval_samples[:,22]==5, True, False), 16].mean().item()})
        comm.Barrier()
    os.system(f"singularity instance stop {ID} > /dev/null 2>&1")