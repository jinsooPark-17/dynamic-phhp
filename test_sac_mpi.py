import os
import time
import uuid
import yaml
import torch
import argparse
import subprocess
from mpi4py import MPI
from script.network import MLPActorCritic
from rl.sac import ReplayBuffer, SAC
from torch.utils.tensorboard import SummaryWriter

def run_episode(output_path, model_path, explore=None, test=False):
    commands = ["python3", "script/gym_episode.py", output_path, "--network_dir", model_path]
    if not explore is None:
        commands += ["--explore", explore]
    elif test:
        commands += ["--test"]
    return subprocess.Popen(commands)

def load_data(sample_path):
    data = torch.load(sample_path)
    os.remove(sample_path)
    return data['state'], data['action'], data['next_state'], data['reward'], data['done']

if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="configuration file *.yml", required=True)
    args = parser.parse_args()

    # Load configuration from yaml file
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Define data storage
    TASK_UUID     = str(uuid.uuid4())
    SLURM_JOBID   = os.getenv("SLURM_JOBID", default='test')
    MODEL_STORAGE = os.path.join("results", SLURM_JOBID, "network")
    MODEL         = os.path.join("results", SLURM_JOBID, "network", "pi.pt")
    EPISODE_DATA  = os.path.join(os.sep, "tmp", f"{TASK_UUID}.pt")
    LOG_DIR       = os.path.join("results", SLURM_JOBID, "tensorboard")
    EXPLORE_CMD   = os.path.join("results", SLURM_JOBID, "explore.command")
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    ROOT = 0

    LOCAL_BATCH_SIZE  = int(config["train"]["batch_size"] / size) + 1
    LOCAL_REPLAY_SIZE = int(config["train"]["replay_size"] / size) + 1

    tensorboard = (SummaryWriter(LOG_DIR) if rank==ROOT else None)

    # Prepare training
    replay = ReplayBuffer(
        obs_dim=config["policy"]["obs_dim"], 
        act_dim=config["policy"]["act_dim"], 
        size=LOCAL_REPLAY_SIZE
    )
    # Share policy
    if rank == ROOT:
        sac = SAC(actor_critic=MLPActorCritic(n_obs=config["policy"]["obs_dim"], n_act=config["policy"]["act_dim"], hidden=(256,256,256,)),
                gamma=config["SAC"]["gamma"], polyak=config["SAC"]["polyak"], lr=config["SAC"]["lr"], alpha=config["SAC"]["alpha"])
        sac.save(MODEL_STORAGE)

        # All episode starts with pure exploration episodes
        with open(EXPLORE_CMD, 'w') as f:
            pass
    else:
        sac = None
    sac = comm.bcast(sac, root=0)

    # Main loop
    ## Run infinite looping episode
    if rank == ROOT:
        ep_proc  = run_episode(output_path=EPISODE_DATA,  model_path=MODEL, explore=EXPLORE_CMD, test=True)
    else:
        ep_proc = run_episode(output_path=EPISODE_DATA, model_path=MODEL, explore=EXPLORE_CMD)

    while not os.path.exists(EPISODE_DATA):
        time.sleep(0.01)

    command = (True if rank==ROOT else False)
    total_steps = 0
    for epoch in range(config["epoch"]):
        start_time = time.time()
        while total_steps // config["steps_per_epoch"] == epoch:
            # Begin episode
            if command is True and rank == ROOT and total_steps >= config["train"]["explore_steps"]:
                command = False
                os.remove(EXPLORE_CMD)

            # Wait for the new episode
            while not os.path.exists(EPISODE_DATA):
                time.sleep(0.01)

            # Use previous episode to train policy
            s, a, ns, r, d = load_data(EPISODE_DATA)
            ep_len = replay.store(s, a, ns, r, d)
            ep_steps = comm.allreduce(ep_len, op=MPI.SUM)
            total_steps += ep_steps

            # Log episode reward to tensorboard
            ep_reward_buf = (torch.empty(size) if rank==ROOT else None)
            comm.Gather(r.sum(), ep_reward_buf, root=ROOT)
            if rank == ROOT:
                tensorboard.add_scalars("reward", {"train": ep_reward_buf[1:].mean(), "test": ep_reward_buf[0]}, total_steps)

            if total_steps > config["train"]["update_after"]:
                loss = torch.zeros((ep_steps, 2), dtype=torch.float64)
                # Train ep_steps step
                for k in range(total_steps-ep_steps, total_steps):
                    batch = replay.sample_batch(LOCAL_BATCH_SIZE)
                    q_info, pi_info, _ = sac.update_mpi(batch=batch, comm=comm)
                    loss[k,0] = pi_info.detach().item()
                    loss[k,1] = q_info.detach().item()

                # Get average loss
                total_loss = (torch.zeros_like(loss) if rank==ROOT else None)
                comm.Reduce([loss, MPI.DOUBLE], [total_loss, MPI.DOUBLE], op=MPI.SUM, root=ROOT)

                # Log new model and loss
                if rank == ROOT:
                    sac.save(MODEL_STORAGE)
                    total_loss = total_loss / size
                    for k in range(total_steps-ep_steps, total_steps):
                        tensorboard.add_scalar("Loss/Pi", total_loss[k,0], k)
                        tensorboard.add_scalar("Loss/Q",  total_loss[k,1], k)
                comm.Barrier()

        # 
        """ No test recording with (Loop-episode + No MPI)
        # For each Epoch, run test episode to show training progress
        test_episode = run_episode(output_path=TEST_EPISODE, model_path=MODEL, test=True)
        test_episode.wait()
        _, _, _, r, _ = load_data(TEST_EPISODE)
        tensorboard.add_scalars("reward", {"test": r.sum()}, total_steps)
        """
        # Save policy for every $SAVE_FREQUENCY epochs
        if rank == ROOT:
            print(f"Epoch {epoch+1} took {time.time() - start_time:.2f} seconds.", flush=True)
            if (epoch+1) % config["save_freqency"] == 0:
                sac.save(os.path.join(MODEL_STORAGE, str(epoch+1)))
        comm.Barrier()

    if rank == ROOT:
        tensorboard.close()
    ep_proc.kill()
    