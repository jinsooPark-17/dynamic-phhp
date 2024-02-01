import os
import time
import uuid
import yaml
import torch
import argparse
import subprocess
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
    TEST_EPISODE  = os.path.join(os.sep, "tmp", f"{TASK_UUID}.pt")
    TRAIN_EPISODE = os.path.join(os.sep, "tmp", f"{TASK_UUID}.pt")
    LOG_DIR       = os.path.join("results", SLURM_JOBID, "tensorboard")
    EXPLORE_CMD   = os.path.join("results", SLURM_JOBID, "explore.command")
    
    # Prepare training
    tensorboard = SummaryWriter(LOG_DIR)
    sac = SAC(actor_critic=MLPActorCritic(n_obs=config["policy"]["obs_dim"], n_act=config["policy"]["act_dim"], hidden=(256,256,256,)),
              gamma=config["SAC"]["gamma"], polyak=config["SAC"]["polyak"], lr=config["SAC"]["lr"], alpha=config["SAC"]["alpha"])
    replay = ReplayBuffer(
        obs_dim=config["policy"]["obs_dim"], 
        act_dim=config["policy"]["act_dim"], 
        size=config["train"]["replay_size"]
    )
    sac.save(MODEL_STORAGE)

    # Main loop
    ## Run infinite looping episode
    with open(EXPLORE_CMD, 'w') as f:
        pass
    train_ep_proc = run_episode(output_path=TRAIN_EPISODE, model_path=MODEL, explore=EXPLORE_CMD)
    test_ep_proc  = run_episode(output_path=TEST_EPISODE,  model_path=MODEL, explore=EXPLORE_CMD)
    while not os.path.exists(TRAIN_EPISODE):
        time.sleep(0.01)

    explore = True
    total_steps = 0
    for epoch in range(config["epoch"]):
        start_time = time.time()
        while total_steps // config["steps_per_epoch"] == epoch:
            # Begin episode
            if explore is True and total_steps >= config["train"]["explore_steps"]:
                explore = False
                os.remove(EXPLORE_CMD)

            # Wait for the new episode
            while not os.path.exists(TRAIN_EPISODE):
                time.sleep(0.01)

            # Use previous episode to train policy
            s, a, ns, r, d = load_data(TRAIN_EPISODE)
            ep_len = replay.store(s, a, ns, r, d)
            total_steps += ep_len
            tensorboard.add_scalars("reward", {"train": r.sum()}, total_steps)

            if total_steps > config["train"]["update_after"]:
                for k in range(total_steps-ep_len, total_steps):
                    batch = replay.sample_batch(config["train"]["batch_size"])
                    q_info, pi_info = sac.update(batch=batch)
                    tensorboard.add_scalar("Loss/Pi", pi_info.detach().item(), k)
                    tensorboard.add_scalar("Loss/Q",  q_info.detach().item(), k)
                sac.save(MODEL_STORAGE)

        """ No test recording with (Loop-episode + No MPI)
        # For each Epoch, run test episode to show training progress
        test_episode = run_episode(output_path=TEST_EPISODE, model_path=MODEL, test=True)
        test_episode.wait()
        _, _, _, r, _ = load_data(TEST_EPISODE)
        tensorboard.add_scalars("reward", {"test": r.sum()}, total_steps)
        """
        # Save policy for every $SAVE_FREQUENCY epochs
        if (epoch+1) % config["save_freqency"] == 0:
            sac.save(os.path.join(MODEL_STORAGE, str(epoch+1)))

        print(f"Epoch {epoch+1} took {time.time() - start_time:.2f} seconds.", flush=True)

    tensorboard.close()
    train_ep_proc.kill()
    # test_episode.kill()