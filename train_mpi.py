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

def launch_simulation(ID):
    # Initiate ROS-Gazebo simulation
    for n_restart in range(10): # Max restart 10 times
        os.system(f"singularity instance start --net --network=none {os.getenv('CONTAINER')} {ID} > /dev/null 2>&1")
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

def generate_heatmap(ax, data1, data2, sigma=5.0, label='baseline'):
    # data has (x, y, yaw) format
    x1, y1, _ = data1.numpy().T
    x2, y2, _ = data2.numpy().T

    extent = [[-12.5, -1.5], [-1.0, 1.0]]
    heatmap1, _, _ = np.histogram2d(x1, y1, bins=[2400, 400], range=extent, density=False)
    heatmap2, _, _ = np.histogram2d(x2, y2, bins=[2400, 400], range=extent, density=False)

    # flip (h,w) -> (w,h)
    heatmap1 = np.flip(heatmap1, axis=1).T
    heatmap2 = np.flip(heatmap2, axis=1).T

    # Apply gaussian smoothing
    heatmap1 = gaussian_filter( heatmap1, sigma=sigma )
    heatmap2 = gaussian_filter( heatmap2, sigma=sigma )

    ax.scatter(x1, y1, s=1, alpha=0.3, c='r', label='dynamic')
    ax.scatter(x2, y2, s=1, alpha=0.3, c='b', label=label)
    ax.imshow( heatmap1/heatmap1.max() - heatmap2/heatmap2.max(), extent=sum(extent,[]), cmap='seismic' )

    # show hallway structure
    ax.add_patch( Rectangle((-12.5, -0.9),11,0.1, hatch='////', fill=False) )
    ax.add_patch( Rectangle((-12.5,  0.8),11,0.1, hatch='////', fill=False) )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=5)

def test_policy(comm, size, rank):
    pass

if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--steps_per_epoch", type=int, default=10_000, required=True)
    parser.add_argument("--replay_size", type=int, default=int(3e4))
    parser.add_argument("--gamma", type=float, default=0.99, help="(default: 0.99)")
    parser.add_argument("--polyak", type=float, default=0.005, help="(default: 0.995)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.2, help="(default: 0.2)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--explore_steps", type=int, default=10000)
    parser.add_argument("--update_after", type=int, default=1000)
    parser.add_argument("--num_test_episodes", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--policy_hz", type=float, default=1.0)
    parser.add_argument("--act_dim", type=int, choices=[3,4], default=3)
    parser.add_argument("--opponents", type=str, nargs='+', choices=["vanilla", "baseline", "custom", "phhp", "dynamic"], required=True)
    args = parser.parse_args()

    # Initialize MPI
    ID    = uuid.uuid4()
    jobID = os.getenv("SLURM_JOBID")

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    ROOT = 0

    tasks_per_node = float(os.getenv('IBRUN_TASKS_PER_NODE'))
    local_rank = int(os.getenv('MPI_LOCALRANKID') or os.getenv('MV2_COMM_WORLD_LOCAL_RANK'))
    LOCAL_BATCH_SIZE = (args.batch_size // size + 1)
    LOCAL_TEST_EPISODES = (args.num_test_episodes // size + 1)
    logger = (SummaryWriter(log_dir=f"runs/{jobID}") if rank==ROOT else None)

    # Log hyper-parameters to tensorboard
    if rank == ROOT:
        if not os.path.exists(f"{os.getenv('WORK')}/checkpoints/{jobID}"): os.makedirs(f"{os.getenv('WORK')}/checkpoints/{jobID}")
        param = f"epochs: {args.epochs}\nsteps_per_epoch: {args.steps_per_epoch}\nbatch_size: {LOCAL_BATCH_SIZE*size}\n" \
                f"gamma: {args.gamma}\npolyak: {args.polyak}\nalpha: {args.alpha}\nlearning_rate: {args.lr}\nalpha: {args.alpha}\n" \
                f"explore_steps: {args.explore_steps}\nupdate_after: {args.update_after}\n" \
                f"number of test episodes: {LOCAL_TEST_EPISODES*size}\n" \
                f"policy_hz: {args.policy_hz}\naction_size: {args.act_dim}\nopponents: {', '.join(args.opponents)}"\
                f"reward: -t + 30.0 * I_success"
        logger.add_text('hyper-parameters', param)

    # Share policy
    replay = ReplayBuffer(obs_dim=640*2+3, act_dim=args.act_dim, size=int(args.replay_size/size+1))
    if rank == ROOT:
        sac = SAC(actor_critic=ActorCritic( n_scan=2, n_act=args.act_dim ), 
                gamma=args.gamma, polyak=args.polyak, lr=args.lr, alpha=args.alpha)
        ep_round = 0
    else:
        sac = None
    sac = comm.bcast(sac, root=0)

    time.sleep(local_rank/tasks_per_node)
    launch_simulation(ID)
    comm.Barrier()



    # Main loop
    total_steps = 0
    dist_reward = []
    for epoch in range(args.epochs):
        step = 0
        t_ep, t_train, t_network, t_test = 0., 0., 0., 0.
        while step < args.steps_per_epoch:
            # Save current policy in **local** /tmp directory
            if local_rank == 0:
                torch.save( sac.ac.pi.state_dict(), "/tmp/model.pt" )
            comm.Barrier()
            # Run episode to collect data
            mode = ("explore" if total_steps < args.explore_steps else "exploit")
            opponent = (random.choice(args.opponents) if rank == 0 else None)
            opponent = comm.bcast(opponent, root=ROOT)

            x1, y1, yaw1 = -12.0, random.uniform(-0.5, 0.5), random.uniform(-pi/4., pi/4.)
            gx1, gy1, gyaw1 = -2.0, 0.0, 0.0

            x2, y2, yaw2 = -2.0, random.uniform(-0.5, 0.5), random.uniform(-pi/4., pi/4.) + pi
            gx2, gy2, gyaw2 = -12.0, 0.0, pi

            start_time = time.time()
            time.sleep(local_rank/tasks_per_node)
            ep_proc = subprocess.Popen([
                "singularity", "run", f"instance://{ID}", "python3", "episode/train_episode.py",
                "--storage", f"/tmp/{ID}.pt", "--network", "/tmp/model.pt", "--mode", mode, "--opponent", opponent,
                "--init_poses", f"{x1}", f"{y1}", f"{yaw1}", "--goal_poses", f"{gx1}", f"{gy1}", f"{gyaw1}",
                "--init_poses", f"{x2}", f"{y2}", f"{yaw2}", "--goal_poses", f"{gx2}", f"{gy2}", f"{gyaw2}",
                "--timeout", "60.0", "--hz", f"{args.policy_hz}"], stderr=subprocess.DEVNULL)
            time.sleep(5.0)
            ep_proc.wait()
            del ep_proc
            gc.collect()
            t_ep += time.time() - start_time

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

        # Evaluate trained policy with test episode
        if local_rank == 0:
            torch.save( sac.ac.pi.state_dict(), "/tmp/model.pt" )
        comm.Barrier()

        if rank == ROOT:
            fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(15,3*4))

        test_ep_reward = torch.zeros(4, dtype=torch.float32)
        ttd = np.zeros((4,2), dtype=np.float32)
        for idx, opponent in enumerate( ['vanilla', 'baseline', 'phhp', 'dynamic'] ):
            data = dict(reward_dist=torch.empty(LOCAL_TEST_EPISODES,2), trajectory1=torch.empty(0,3), trajectory2=torch.empty(0,3))
            for k in range(LOCAL_TEST_EPISODES):
                x1,  y1,  yaw1  = [-12.0, 0.0, 0.0]
                gx1, gy1, gyaw1 = [ -2.0, 0.0, 0.0]
                x2,  y2,  yaw2  = [ -2.0, 0.0, pi ]
                gx2, gy2, gyaw2 = [-12.0, 0.0, pi ]
                start_time = time.time()
                time.sleep(local_rank/tasks_per_node)
                ep_proc = subprocess.Popen([
                    "singularity", "run", f"instance://{ID}", "python3", "episode/train_episode.py",
                    "--storage", f"/tmp/{ID}.pt", "--network", "/tmp/model.pt", "--mode", "evaluate", "--opponent", opponent,
                    "--init_poses", f"{x1}", f"{y1}", f"{yaw1}", "--goal_poses", f"{gx1}", f"{gy1}", f"{gyaw1}",
                    "--init_poses", f"{x2}", f"{y2}", f"{yaw2}", "--goal_poses", f"{gx2}", f"{gy2}", f"{gyaw2}",
                    "--timeout", "60.0", "--hz", f"{args.policy_hz}"], stderr=subprocess.DEVNULL)
                ep_proc.wait()
                t_test += time.time() - start_time
                # Load episode result from file
                new_data = torch.load(f"/tmp/{ID}.pt")
                for key in ['trajectory1', 'trajectory2']:
                    data[key] = torch.cat((data[key], new_data[key]))

                dist_reward += [[np.linalg.norm(new_data['trajectory1'][1:,:2] - new_data['trajectory1'][:-1,:2], axis=1).sum(), new_data['reward'].sum()]]
                test_ep_reward[idx] += new_data['reward'].sum()
                ttd[idx] += [new_data['ttd1'], new_data['ttd2']]
 
            # collect data
            sendcounts1 = comm.gather( data['trajectory1'].numel(), root=ROOT )
            sendcounts2 = comm.gather( data['trajectory2'].numel(), root=ROOT )
            if rank == ROOT:
                data1 = torch.empty(sum(sendcounts1), dtype=torch.float32).view(-1,3)
                data2 = torch.empty(sum(sendcounts2), dtype=torch.float32).view(-1,3)
            else:
                data1 = data2 = None
            comm.Gatherv(sendbuf=data['trajectory1'], recvbuf=(data1, sendcounts1), root=ROOT)
            comm.Gatherv(sendbuf=data['trajectory2'], recvbuf=(data2, sendcounts2), root=ROOT)

            # Now create heatmap from trajectory information
            if rank == ROOT:
                generate_heatmap(ax[idx], data1, data2, sigma=5.0, label=opponent)
        # Collect information
        all_dist_reward = (np.zeros((size, len(dist_reward), 2), dtype=np.float32) if rank == ROOT else None)
        comm.Gather(np.array(dist_reward, dtype=np.float32), all_dist_reward, ROOT)

        avg_test_ep_reward = (torch.zeros(4, dtype=torch.float32) if rank==ROOT else None)
        comm.Reduce([test_ep_reward, MPI.FLOAT], [avg_test_ep_reward, MPI.FLOAT], MPI.SUM, ROOT)

        avg_ttd = (np.zeros((4,2), dtype=np.float32) if rank==ROOT else None)
        comm.Reduce([ttd, MPI.FLOAT], [avg_ttd, MPI.FLOAT], MPI.SUM, ROOT)
        if rank == ROOT:
            avg_ttd /= size*LOCAL_TEST_EPISODES
            logger.add_scalar('TTD advantage/vs Vanilla', avg_ttd[0,0] - avg_ttd[0,1], global_step=epoch+1)
            logger.add_scalar('TTD advantage/vs Baseline', avg_ttd[1,0] - avg_ttd[1,1], global_step=epoch+1)
            logger.add_scalar('TTD advantage/vs PHHP', avg_ttd[2,0] - avg_ttd[2,1], global_step=epoch+1)
            logger.add_scalar('TTD advantage/Dynamic', (avg_ttd[3,0]+avg_ttd[3,1])/2., global_step=epoch+1)

            avg_test_ep_reward /= size*LOCAL_TEST_EPISODES
            logger.add_scalar('Test Episode Reward/vs Vanilla', avg_test_ep_reward[0], global_step=epoch+1)
            logger.add_scalar('Test Episode Reward/vs Baseline', avg_test_ep_reward[1], global_step=epoch+1)
            logger.add_scalar('Test Episode Reward/vs PHHP', avg_test_ep_reward[2], global_step=epoch+1)
            logger.add_scalar('Test Episode Reward/vs Dynamic', avg_test_ep_reward[3], global_step=epoch+1)

            logger.add_figure('Trajectory of trained policy', fig, global_step=epoch+1)

            all_dist_reward = all_dist_reward.reshape(-1,2).T
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
            ax.scatter(all_dist_reward[0], all_dist_reward[1], s=3, c='k')
            ax.set_xlim(0.0, 11.0)
            ax.set_ylim(-60.0, 10.0)
            logger.add_figure('Episode reward vs traveled distance', fig, global_step=epoch+1)
            del all_dist_reward
            print(f"Episode {epoch+1} took")
            print(f"\tepisodes: {t_ep:.2f} seconds")
            print(f"\ttrain: {t_train:.2f} seconds")
            print(f"\t\tnetwork: {t_network:.2f} seconds")
            print(f"\t\tbackprop: {t_train-t_network:.2f} seconds")
            print(f"\ttest: {t_test:.2f} seconds", flush=True)

        # # Generate checkpoint
        if epoch % args.save_freq == 0 and rank == ROOT:
            sac.checkpoint(epoch+1, checkpoint_dir=f"{os.getenv('WORK')}/checkpoints/{jobID}/{epoch+1}.pt")
    if rank == ROOT:
        logger.close()
    os.system(f"singularity instance stop {ID}")