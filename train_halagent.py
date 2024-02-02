import os
import time
import uuid
import yaml
import torch
import argparse
import subprocess
from script.network import ActorCritic
from rl.sac import ReplayBuffer, SAC
from torch.utils.tensorboard import SummaryWriter

def launch_simulation(uuid, args=''):
    # Initiate ROS-Gazebo simulation
    for n_restart in range(10): # Max restart 10 times
        os.system(f"singularity instance start --net --network=none {os.getenv('CONTAINER')} {uuid} {args}")
        time.sleep(5.0)
        test_proc = subprocess.Popen(["singularity", "run", f"instance://{uuid}", "/wait_until_stable"])
        try:
            test_proc.wait( timeout=60.0 )
        except subprocess.TimeoutExpired as e:
            print(f"Restarting {uuid}...", flush=True)
            os.system(f"singularity instance stop {uuid} > /dev/null 2>&1")
        else:
            return
    raise RuntimeError("Simulation failed to launched after 10 trials. Stop training process.")

def run_episode(uuid, output_path, model_path, config_path, command, test=False):
    commands = ["singularity", "run", f"instance://{uuid}", "python3.7", "script/halagent_episode.py", output_path, "--network_dir", model_path, "--config", config_path, "--command", command]
    if test is True:
        commands += ["--test"]
    return subprocess.Popen(commands)

def load_data(sample_path):
    data = torch.load(sample_path)
    os.remove(sample_path)
    return data['state'], data['action'], data['next_state'], data['reward'], data['done']

def random_opponent(opponent_list):
    opponent_list = opponent_list
    def gen_command():
        idx = torch.randint(0, len(opponent_list), (1,)).item()
        with open("opponent.command", 'w') as f:
            f.write(opponent_list[idx])
    return gen_command

if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="configuration file *.yml", required=True)
    args = parser.parse_args()

    # Load configuration from yaml file
    with open(args.config) as f:
        config = yaml.safe_load(f)
    N_PLAN = int(config["policy"]["sensor_horizon"]/config["policy"]["plan_interval"]*2.)
    config["policy"]["n_plan"] = N_PLAN
    config["policy"]["obs_dim"] = 2*config["policy"]["n_scan"]*640 + N_PLAN + 2
    choose_opponent = random_opponent(config["episode"]["opponents"])

    # Define data storage
    TASK_UUID     = str(uuid.uuid4())
    SLURM_JOBID   = os.getenv("SLURM_JOBID", default='test')
    DATA_STORAGE  = os.path.join("results", SLURM_JOBID)
    MODEL_STORAGE = os.path.join(DATA_STORAGE, "network")
    MODEL         = os.path.join(DATA_STORAGE, "network", "pi.pt")
    TEST_EPISODE  = os.path.join(os.sep, "tmp", f"{TASK_UUID}.pt")
    TRAIN_EPISODE = os.path.join(os.sep, "tmp", f"{TASK_UUID}.pt")
    LOG_DIR       = os.path.join(DATA_STORAGE, "tensorboard")
    EXPLORE_CMD   = os.path.join(DATA_STORAGE, "explore.command")

    # Prepare training
    tensorboard = SummaryWriter(LOG_DIR)
    actor_critic = ActorCritic(n_scan=config["policy"]["n_scan"],
                               n_plan=N_PLAN,
                               action_dim=config["policy"]["act_dim"],
                               combine=config["policy"]["combine_scans"])

    sac = SAC(actor_critic=actor_critic,
              gamma=config["SAC"]["gamma"], 
              polyak=config["SAC"]["polyak"], 
              lr=config["SAC"]["lr"], 
              alpha=config["SAC"]["alpha"])

    replay = ReplayBuffer(
        obs_dim=config["policy"]["obs_dim"], 
        act_dim=config["policy"]["act_dim"], 
        size=config["train"]["replay_size"]
    )
    sac.save(MODEL_STORAGE)

    # Launch simulation
    time.sleep( torch.rand(1).item()*10 )
    launch_simulation(uuid=TASK_UUID, args='gui:=true')

    # Main loop
    ## Generate commands
    choose_opponent()
    with open(EXPLORE_CMD, 'w') as f:
        pass
    ## Run infinite looping episode
    train_ep_proc = run_episode(uuid=TASK_UUID,
                                output_path=TRAIN_EPISODE, 
                                model_path=MODEL,
                                config_path=args.config,
                                command=DATA_STORAGE)
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
            time.sleep(0.1)

            # Use previous episode to train policy
            choose_opponent()   # choose opponent robot behavior for the next episode
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