import os
import gc
import time
import uuid
import torch
import random
import mpi4py
mpi4py.rc.recv_mprobe = False
import argparse
import subprocess
from math import pi
from mpi4py import MPI
from episode.policy.policy import ActorCritic
from rl.sac import ReplayBuffer, SAC

if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--steps_per_epoch", type=int, default=10_000, required=True)
    parser.add_argument("--replay_size", type=int, default=int(1e7))
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
    parser.add_argument("--opponents", type=str, choices=["vanilla", "baseline", "custom", "phhp"], action='append')
    args = parser.parse_args()

    # Initialize MPI
    ID    = uuid.uuid4()
    jobID = os.getenv("SLURM_JOBID")

    comm       = MPI.COMM_WORLD
    size       = comm.Get_size()
    rank       = comm.Get_rank()
    local_rank = int(os.getenv('MPI_LOCALRANKID') or os.getenv('MV2_COMM_WORLD_LOCAL_RANK'))
    LOCAL_BATCH_SIZE = (args.batch_size // size + 1)
    LOCAL_TEST_EPISODES = (args.num_test_episodes // size + 1)

    # Share policy
    replay = ReplayBuffer(obs_dim=640*2+3, act_dim=args.act_dim, size=int(args.replay_size/size+1))
    if rank == 0:
        sac = SAC(actor_critic=ActorCritic( n_scan=2, n_act=args.act_dim ), 
                gamma=args.gamma, polyak=args.polyak, lr=args.lr, alpha=args.alpha)
    else:
        sac = None
    sac = comm.bcast(sac, root=0)


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
    max_restart = comm.allreduce(n_restart, op=MPI.MAX)
    assert max_restart < 10, f"Some simulation failed to launched after 10 trials. Stop training process."

    # Main loop
    total_steps = 0
    for epoch in range(args.epochs):
        t = 0
        while t < args.steps_per_epoch:
            # Save current policy in **local** /tmp directory
            if local_rank == 0:
                torch.save( sac.ac.pi.state_dict(), "/tmp/model.pt" )
            comm.Barrier()
            # Run episode to collect data
            mode = ("explore" if total_steps < args.explore_steps else "exploit")
            opponent = (random.choice(args.opponents) if rank == 0 else None)
            opponent = comm.bcast(opponent, root=0)

            x1, y1, yaw1 = -12.0, random.uniform(-0.5, 0.5), random.uniform(-pi/4., pi/4.)
            gx1, gy1, gyaw1 = -2.0, 0.0, 0.0

            x2, y2, yaw2 = -2.0, random.uniform(-0.5, 0.5), random.uniform(-pi/4., pi/4.) + pi
            gx2, gy2, gyaw2 = -12.0, 0.0, pi

            time.sleep(local_rank/24.0)
            ep_proc = subprocess.Popen([
                "singularity", "run", f"instance://{ID}", "python3", "episode/train_episode.py",
                "--storage", f"/tmp/{ID}.pt", "--network", "/tmp/model.pt", "--mode", mode, "--opponent", opponent,
                "--init_poses", f"{x1}", f"{y1}", f"{yaw1}", "--goal_poses", f"{gx1}", f"{gy1}", f"{gyaw1}",
                "--init_poses", f"{x2}", f"{y2}", f"{yaw1}", "--goal_poses", f"{gx2}", f"{gy2}", f"{gyaw2}",
                "--timeout", "60.0", "--hz", f"{args.policy_hz}"], stderr=subprocess.DEVNULL)
            ep_proc.wait()
            del ep_proc
            gc.collect()

            # Read episode result from file
            data = torch.load(f"/tmp/{ID}.pt")
            s, a, ns, r, d = list(map(data.get, ["state", "action", "next_state", "reward", "done"]))
            ep_len = replay.store(state=s, action=a, next_state=ns, reward=r, done=d)

            ep_steps = comm.allreduce( ep_len, op=MPI.SUM )
            if total_steps > args.update_after:
                for step in range(ep_steps):
                    print(replay.sample_batch(LOCAL_BATCH_SIZE))
                    loss_q, loss_pi = sac.update_mpi(batch=replay.sample_batch(LOCAL_BATCH_SIZE), comm=comm)
                    ####################################################################
                    X = (torch.rand(1, 640*2+3) if rank == 0 else None)
                    X = comm.bcast(X, root=0)
                    Y = sac.ac.act(X, deterministic=True).squeeze()
                    recv_buf = (torch.empty((size, *Y.shape)) if rank == 0 else None)
                    comm.Gather(Y, recv_buf, root=0)
                    if rank==0: 
                        y_min = recv_buf.min(axis=0)
                        y_max = recv_buf.max(axis=0)
                        if torch.count_nonzero(y_max.values - y_min.values) != 0:
                            print(f"!!!Policy sync failed!!!\n\t{y_min.values} ~ {y_min.values}", flush=True)
                    ####################################################################
            t+= ep_steps
            total_steps += ep_steps
        # Perform test episode
        # for _ in range(LOCAL_TEST_EPISODES):
        #     ep_proc = subprocess.Popen([
        #         "singularity", "run", f"instance://{ID}", "python3", "episode/train_episode.py",
        #         "--storage", f"/tmp/{ID}.pt", "--network", "/tmp/model.pt", "--mode", "eval", "--opponent", opponent,
        #         "--init_poses", f"{x1}", f"{y1}", f"{yaw1}", "--goal_poses", f"{gx1}", f"{gy1}", f"{gyaw1}",
        #         "--init_poses", f"{x2}", f"{y2}", f"{yaw1}", "--goal_poses", f"{gx2}", f"{gy2}", f"{gyaw2}",
        #         "--timeout", "60.0", "--hz", args.policy_hz], stderr=subprocess.DEVNULL)
        #     ep_proc.wait()

        # # Generate checkpoint
        # if epoch % args.save_freq == 0:
        #     pass    # save current network