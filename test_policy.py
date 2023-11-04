import os
import math
import time
import argparse
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot1', type=str, choices=['eband', 'dwa', 'teb'], default='eband')
    parser.add_argument('--robot2', type=str, choices=['eband', 'dwa', 'teb'], default='eband')
    args = parser.parse_args()

    try:
        os.system(f'singularity instance start ~/Desktop/bwi-melodic.simg test gui:=true rviz:=true local_planner1:={args.robot1} local_planner2:={args.robot2} > /dev/null 2>&1')
        time.sleep(60.0)

        for ID, opponent in enumerate(['vanilla', 'baseline', 'phhp', 'dynamic']):
            x1,  y1,  yaw1  = -12.0, 0.0, 0.0
            gx1, gy1, gyaw1 =  -2.0, 0.0, 0.0
            x2,  y2,  yaw2  =  -2.0, 0.0, math.pi
            gx2, gy2, gyaw2 = -12.0, 0.0, math.pi

            ep_proc = subprocess.Popen([
            "singularity", "run", f"instance://test", "python3", "episode/train_episode.py",
            "--storage", f"~/{ID}.pt", "--network", "/tmp/model.pt", "--mode", "evaluate", "--opponent", opponent,
            "--init_poses", f"{x1}", f"{y1}", f"{yaw1}", "--goal_poses", f"{gx1}", f"{gy1}", f"{gyaw1}",
            "--init_poses", f"{x2}", f"{y2}", f"{yaw2}", "--goal_poses", f"{gx2}", f"{gy2}", f"{gyaw2}",
            "--timeout", "60.0", "--hz", f"{args.policy_hz}"], stderr=subprocess.DEVNULL)

    finally:
        os.system(f'singularity instance stop test')
