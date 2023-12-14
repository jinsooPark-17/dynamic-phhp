import os
import torch
import numpy as np
import argparse
import gpytorch
from copy import deepcopy
from math import pi
from itertools import combinations
from collections import namedtuple
from envs.robots import AllinOne, quaternion_to_yaw
from envs.simulation import Gazebo

import rospy
from nav_msgs.msg import OccupancyGrid
Pose = namedtuple( "Pose", "x y yaw" )
rng = np.random.default_rng()

def reset_episode( robots, reset_poses, init_poses ):
    # Reset episode
    ## Immediately stop any previous motion
    for r in robots:
        r.stop( )
    rospy.sleep( 1.0 )

    ## Teleport robots to initial pose
    for r, p in zip( robots, reset_poses ):
        env.teleport( r.id, p.x, p.y, p.yaw )
    for r, p in zip( robots, reset_poses ):
        # Run second round to avoid overlap
        env.teleport( r.id, p.x, p.y, p.yaw )
    rospy.sleep( 1.0 )

    ## Localize robots to new position
    for _ in range(15):
        for r, p in zip( robots, reset_poses ):
            r.localize( p.x, p.y, p.yaw )
        rospy.sleep( 0.1 )
    rospy.sleep( 2.0 )

    ## Clear costmap of each robots
    for r in robots:
        r.clear_costmap( )

    ## Move to episode init_pose
    for r, (x, y, yaw) in zip( robots, init_poses ):
        r.move(x, y, yaw)

    while not rospy.is_shutdown():
        for r in robots:
            if r.is_running() is True:
                break
        else:
            break

def begin_episode( robots, reset_poses, init_poses, goal_poses, max_retry=10, timeout=60. ):
    for _ in range(max_retry):
        # Retry episode if ANY robots failed within 0.5 seconds
        reset_episode(robots=robots, reset_poses=reset_poses, init_poses=init_poses)

        for r, g in zip(robots, goal_poses):
            r.move( g.x, g.y, g.yaw, mode='vanilla', timeout=timeout)
        rospy.sleep( 0.5 )

        if all( [r.is_running() for r in robots] ):
            break
    else:
        raise RuntimeError( f"!!!Episode failed after {max_retry} attempts!!!" )

def path_distance(r0, r1, g0, g1):
    plan0 = r0.make_plan( r0.pose, g0 )
    plan1 = r1.make_plan( r1.pose, g1 )

    if plan0.size == 0 or plan1.size == 0:
        return -1.0, None, None
    dist0 = np.linalg.norm( plan0 - plan1[0], axis=1 )
    dist1 = np.linalg.norm( plan1 - plan0[0], axis=1 )
    idx0 = dist0.argmin()
    idx1 = dist1.argmin()

    d0 = (np.inf if dist0[idx0]>0.8 else np.cumsum(np.linalg.norm( plan0[1:] - plan0[:-1] , axis=1 ))[idx0-1])
    d1 = (np.inf if dist1[idx1]>0.8 else np.cumsum(np.linalg.norm( plan1[1:] - plan1[:-1] , axis=1 ))[idx1-1])
    return min(d0, d1), plan0[0], plan1[0]

def sample_waypoint_features(name, p0, p1, threshold=0.65, n_sample=800):
    costmap_msg = rospy.wait_for_message( os.path.join(name, 'move_base', 'local_costmap', 'costmap'), OccupancyGrid )
    h = costmap_msg.info.height
    w = costmap_msg.info.width
    resolution = costmap_msg.info.resolution
    origin = np.array([costmap_msg.info.origin.position.x, costmap_msg.info.origin.position.y])
    costmap = np.array( costmap_msg.data ).reshape(h,w)
    grid_world = ( costmap < 50 ).astype(np.float64)
    wall_point = np.argwhere( (costmap >98) == True )

    def visibility_from_corner(grid):
        w, h = grid.shape
        for x in range(w):
            for y in range( int(x==0), h):
                grid[x,y] *= (x*grid[x-1, y] + y*grid[x, y-1]) / (x + y)
    visibility_from_corner( grid_world[h//2::1,  w//2::1 ] )
    visibility_from_corner( grid_world[h//2::-1, w//2::1 ] )
    visibility_from_corner( grid_world[h//2::-1, w//2::-1] )
    visibility_from_corner( grid_world[h//2::1,  w//2::-1] )

    # Sample available waypoints
    waypoint_point = np.argwhere( (grid_world > threshold) == True )
    sample_idx = rng.choice( waypoint_point.shape[0], n_sample, replace=False)
    samples = waypoint_point[sample_idx]    # y,x format

    # Calculate d3, d4
    disp_to_wall = samples[:, np.newaxis, :] - wall_point[np.newaxis, :, :]
    dist_to_wall = np.linalg.norm( disp_to_wall, axis=2 )
    w1 = disp_to_wall[np.arange(n_sample), np.argmin(dist_to_wall, axis=1), :]
    d3 = dist_to_wall.min(axis=1) * resolution
    for i in range( n_sample ):
        idx = np.argwhere( disp_to_wall[i].dot( w1[i] ) >= 0. )
        dist_to_wall[i, idx] = np.inf
    d4 = dist_to_wall.min(axis=1) * resolution

    samples = samples[:,::-1] * resolution + origin
    d1 = np.linalg.norm( samples - p0, axis=1 )
    d2 = np.linalg.norm( samples - p1, axis=1 )

    return samples, np.vstack((d1, d2, d3, d4)).T

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5)) # 0.5 / 1.5 / 2.5 (default)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def load_data(storage):
    if not os.path.isfile( storage ):
        return torch.zeros((0,4)).float(), torch.zeros(0).float()
    train_data = torch.load(storage)
    return train_data['features'].float(), train_data['observations'].float()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("storage", type=str, help="Name of train data (.pt file)")
    parser.add_argument("train_iter", type=int, default=50, help="Gaussian Process Regression training iteration")
    parser.add_argument("detection_range", type=float, default=8.0, help="detection range of robots")
    parser.add_argument("n_sample", type=int, default=800, help="Number of samples from visible area")
    parser.add_argument("epsilon", type=float, default=0.05, help="Epsilon value of epsilon greedy algorithm")
    parser.add_argument("timeout", type=float, default=60.0, help="Timeout of episode")
    parser.add_argument("reward_constant", type=float, default=50.0, help="Constant to make successful episodic reward positive value")
    args = parser.parse_args()

    robot_ids = np.array([ 'marvin', 'rob' ], dtype=str)

    env = Gazebo( )
    robot_idx = np.arange( robot_ids.shape[0] )
    robots = np.array( [AllinOne(id) for id in robot_ids] )
    available = np.full_like( robot_ids, fill_value=True, dtype=np.bool_ )
    info = np.zeros((robot_ids.shape[0], 2), dtype=object)    # goal, TTD

    # Load Gaussian Process Regression model
    train_x, train_y = load_data( args.storage )
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gpr = ExactGPModel(train_x, train_y, likelihood)

    likelihood.train()
    gpr.train()
    optimizer = torch.optim.Adam( gpr.parameters(), lr=0.1 )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gpr)
    for _ in range(args.train_iter):
        optimizer.zero_grad()
        output = gpr(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
    likelihood.eval()
    gpr.eval()

    # Define episode configuration
    r0, p0, g0 = Pose(-20.0, 0.0, 0.0), Pose( -16.0, 0.0, 0.0 ), Pose(   0.0, 0.0, 0.0 )
    r1, p1, g1 = Pose( -2.0, 0.0,  pi), Pose(  -6.0, 0.0,  pi ), Pose( -22.0, 0.0,  pi )

    # Halting episode
    begin_episode(robots=robots, reset_poses=[r0,r1], init_poses=[p0,p1], goal_poses=[g0,g1], timeout=args.timeout)
    while not rospy.is_shutdown():
        # Termination condition: when all robots stopped
        if any([r.is_running() for r in robots]) is False:
            break

        for r0, r1 in combinations( robots[available], 2 ):
            # if d_r0r1 < Detection_range
            d, p0, p1 = path_distance( r0, r1, r0.goal, r1.goal )
            if d < args.detection_range:
                # sample waypoints from r0, r1 and measure d1, d2, d3, d4
                samples  = np.empty( shape=(args.n_sample*2, 2), dtype=np.float64 )
                features = np.empty( shape=(args.n_sample*2, 4), dtype=np.float64 )
                samples[:args.n_sample], features[:args.n_sample] = sample_waypoint_features( r0.id, p0, p1, threshold=0.6, n_sample=args.n_sample )
                samples[args.n_sample:], features[args.n_sample:] = sample_waypoint_features( r1.id, p1, p0, threshold=0.6, n_sample=args.n_sample )

                mode = rng.choice(['explore', 'exploit'], p=[args.epsilon, 1-args.epsilon])
                if mode == 'explore':
                    waypoint_idx = rng.choice( np.arange(args.n_sample*2) )
                else:
                    test_x = torch.from_numpy( features ).float()
                    reward_pred = likelihood( gpr( test_x ) ).sample()
                    waypoint_idx = reward_pred.argmax().item()
                polite   = (r0 if waypoint_idx < args.n_sample else r1)

                # Store information
                polite_idx = np.where( robot_ids == polite.id )
                available[polite_idx] = False
                info[polite_idx, 0] = deepcopy(polite.goal)
                info[polite_idx, 1] = deepcopy(polite.ttd)     # rospy.Time

                # Move the polite robot to designated waypoint
                wx, wy = samples[waypoint_idx]
                yaw = polite.trajectory[polite.traj_idx-1][2]   # x, y, yaw
                polite.move( wx, wy, yaw, timeout=args.timeout )

        for idx, polite, (polite_goal, ttd) in zip( robot_idx[~available], robots[~available], info[~available] ):
            if polite.is_running() is True:
                continue
            
            d_nearest = np.inf
            for bold in robots[available]:
                d, _, _ = path_distance(polite, bold, polite_goal, bold.goal)
                d_nearest = min(d_nearest, d)

            if d_nearest > args.detection_range:
                x = polite_goal.target_pose.pose.position.x
                y = polite_goal.target_pose.pose.position.y
                yaw = quaternion_to_yaw( polite_goal.target_pose.pose.orientation )
                polite.move( x, y, yaw, timeout=args.timeout )
                polite.ttd = ttd
                available[ idx ] = True
        rospy.sleep(0.05)

    # End of episode
    print("End of episode")
    reward = 0.0
    success = True
    for r in robots:
        reward -= r.ttd
        success *= r.is_arrived()
        print(f"\t{r.id}: {r.ttd:.2f} seconds")
    reward += args.reward_constant - args.timeout*(not success)
    print(f"Episode reward: {reward}")
    print(f"Timeout: {not success}")