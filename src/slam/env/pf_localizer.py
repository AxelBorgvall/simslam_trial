from pathlib import Path
from .gridmap import GridMap
from .robot import *
from numba import njit, prange
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


class PFLocalizer:
    def __init__(self, env: Environment, n_part: int, ang_noise=0.20, vel_noise=0.20,resampling_temperature=8.0):
        self.env = env
        # map of distance to closest wall (used for computing particle likelyhood)
        self.distmap = distance_transform_edt(~env.map.grid) * self.env.map.dx

        self.resampling_temp = resampling_temperature
        self.particles = np.random.rand(n_part, 3)
        self.particles[:, :2] *= self.env.l
        self.particles[:, 2] = self.particles[:, 2] * 2 * np.pi - np.pi
        self.n_part = n_part

        self.ang_noise = ang_noise
        self.vel_noise = vel_noise
        self.weights = np.ones(n_part) / n_part

    def step_env(self, dt: float):
        action, scan = self.env.step(dt)
        return action, scan

    def update_particles(
        self, action: tuple[float, float], scan: np.ndarray, dt: float
    ):
        """
        action: (v, omega) tuple
        scan: ndarray
        """
        # Params for the computed likelyhood, lidar sigma and a minimum logprob to prevent single rays from nuking a particlr
        sigsqr = 0.20
        min_logprob = -5.0

        # Noise scales with speed/angvel
        std_v = self.vel_noise * abs(action[0]) + 0.05
        std_w = self.ang_noise * abs(action[1]) + 0.05

        n_part = self.n_part
        vel = action[0] + np.random.randn(n_part) * std_v
        omega = action[1] + np.random.randn(n_part) * std_w

        # Update particle poses
        old_theta = self.particles[:, 2].copy()
        self.particles[:, :2] += (
            np.column_stack((np.cos(old_theta), np.sin(old_theta))) 
            * (vel * dt)[:, None]
        )
        self.particles[:, 2] += omega * dt
        self.particles[:, 2] = (self.particles[:, 2] + np.pi) % (2 * np.pi) - np.pi

        # What rays actually hit something? Ignore all that didng
        lidar_mask = scan < (self.env.max_lidar_range - 0.01)
        lidar_angles = self.particles[:, 2, None] + self.env.robot.angles[None, :]
        lidar_points = self.particles[:, None, :2] + scan[None, :, None] * np.stack(
            (np.cos(lidar_angles), np.sin(lidar_angles)), axis=-1
        )
        indices = (lidar_points / self.env.map.dx).astype(np.int64)
        xs = indices[:, :, 0]
        ys = indices[:, :, 1]
        max_y, max_x = self.distmap.shape
        valid_mask = (xs >= 0) & (xs < max_x) & (ys >= 0) & (ys < max_y)
        xs_safe = np.clip(xs, 0, max_x - 1)
        ys_safe = np.clip(ys, 0, max_y - 1)
        dists = self.distmap[ys_safe, xs_safe]

        prob_hit = np.exp(-(dists**2) / (2 * sigsqr))
        z_hit = 0.95
        z_rand = 0.05
        p_total = z_hit * prob_hit + z_rand * (1.0 / self.env.max_lidar_range)
        ray_logprobs = np.log(p_total)

        ray_logprobs[~valid_mask] = np.log(z_rand * (1.0 / self.env.max_lidar_range))
        ray_logprobs[:, ~lidar_mask] = 0.0

        # Sum and update weights
        logprob = np.sum(ray_logprobs, axis=1)
        logprob = logprob / self.resampling_temp
        
        logprob = np.exp(logprob - np.max(logprob))
        self.weights = logprob / np.sum(logprob)

        n_eff = 1.0 / (np.sum(self.weights**2) + 1e-10)
        if n_eff < n_part / 2.0:
            self.resample()

    def resample(self):
        cdf = np.cumsum(self.weights)
        
        # n_keep = int(self.n_part * 0.95)
        n_keep = int(self.n_part * 1.0)
        n_random = self.n_part - n_keep
        
        r = np.random.uniform(0, 1 / n_keep)
        pointers = r + np.arange(n_keep) / n_keep
        indices = np.searchsorted(cdf, pointers)
        kept_particles = self.particles[indices].copy()
        
        # Inject 5% random testicles
        random_particles = np.random.rand(n_random, 3)
        random_particles[:, :2] *= self.env.l
        random_particles[:, 2] = random_particles[:, 2] * 2 * np.pi - np.pi
        
        self.particles = np.vstack((kept_particles, random_particles))
        self.weights = np.ones(self.n_part) / self.n_part

    def render(self, lidar_scan=None):
        h, w = self.env.map.grid.shape
        img_true = np.ones((h, w, 3), dtype=np.uint8) * 255
        img_true[self.env.map.grid] = np.zeros(3, dtype=np.uint8)
        img_guess = img_true.copy()
        self.env.render_robot_to_img(img_true, lidar_scan)

        centroid=(np.sum(self.weights[:,None]*self.particles[:,:2],axis=0)/np.sum(self.weights)/self.env.map.dx).astype(np.int64)
        cv2.circle(img_true, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)


        # render particles to second image
        xp = self.particles[:, 0]
        yp = self.particles[:, 1]
        for x, y in zip(xp, yp):
            xi = int(x / self.env.map.dx)
            yi = int(y / self.env.map.dx)
            cv2.circle(img_guess, (xi, yi), 2, (255, 0, 0), -1)
        return img_true, img_guess
