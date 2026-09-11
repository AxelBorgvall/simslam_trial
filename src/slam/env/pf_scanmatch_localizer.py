import numpy as np
import cv2
from numba import njit, prange
from scipy.ndimage import distance_transform_edt
from .gridmap import GridMap
from .robot import *

@njit(parallel=True)
def correlative_scan_match(
    priors, scan, angles, distmap, dx, max_range, sigsqr,
    n_xy_steps, xy_step, n_th_steps, th_step, z_hit, z_rand
):
    """
    Docstring for correlative_scan_match
    
    :param priors: particle poses [Npart,3]
    :param scan: lidar data [Nray]
    :param angles: lidar angles relative to robot pose [Nray]
    :param distmap: distance map [Nsqr,Nsqr]
    :param dx: (float) index->spatial
    :param max_range: (float) max lidar range
    :param sigsqr: (float) noise model param
    :param n_xy_steps: (int) number of xy steps out from center in search grid
    :param xy_step: (flaot) search grid stepsize
    :param n_th_steps: (int) num theta steps
    :param th_step: (float) theta stepsize in search grid 
    :param z_hit: multiplier for logprob.
    :param z_rand: additive modifier for logprob to prevent a single bad ray from killing a promising particle. 
    """
    # N=num particles
    # M=num lidar rays
    N = priors.shape[0]
    M = scan.shape[0]
    max_y, max_x = distmap.shape
    
    means = np.zeros((N, 3), dtype=np.float64)
    covs = np.zeros((N, 3, 3), dtype=np.float64)
    weights = np.zeros(N, dtype=np.float64)
    
    num_test_poses = (2 * n_xy_steps + 1)**2 * (2 * n_th_steps + 1)
    
    # Parallelized loop over particles since they can be evaluated independently
    for i in prange(N):
        px_prior = priors[i, 0]
        py_prior = priors[i, 1]
        pth_prior = priors[i, 2]
        
        test_poses = np.zeros((num_test_poses, 3), dtype=np.float64)
        log_probs = np.zeros(num_test_poses, dtype=np.float64)
        
        idx = 0
        max_log_prob = -1e9 # Low initial value
        
        for idx_x in range(-n_xy_steps, n_xy_steps + 1):
            for idx_y in range(-n_xy_steps, n_xy_steps + 1):
                for idx_th in range(-n_th_steps, n_th_steps + 1):
                    
                    px = px_prior + idx_x * xy_step
                    py = py_prior + idx_y * xy_step
                    pth = pth_prior + idx_th * th_step
                    
                    test_poses[idx, 0] = px
                    test_poses[idx, 1] = py
                    test_poses[idx, 2] = pth
                    
                    total_lp = 0.0
                    
                    for j in range(M):
                        r = scan[j]
                        if r < max_range - 0.01:
                            hx = px + r * np.cos(pth + angles[j])
                            hy = py + r * np.sin(pth + angles[j])
                            
                            ix = int(hx / dx)
                            iy = int(hy / dx)
                            
                            if 0 <= ix < max_x and 0 <= iy < max_y:
                                dist = distmap[iy, ix]
                            else:
                                # Penalty for OOB rays
                                dist = max_range  
                                
                            prob_hit = np.exp(-(dist * dist) / (2.0 * sigsqr))
                            p_total = z_hit * prob_hit + z_rand * (1.0 / max_range)
                            total_lp += np.log(p_total)
                            
                    log_probs[idx] = total_lp
                    if total_lp > max_log_prob:
                        max_log_prob = total_lp
                    
                    idx += 1
                    
        # Normalize probabilities 
        sum_p = 0.0
        for k in range(num_test_poses):
            p = np.exp(log_probs[k] - max_log_prob)
            sum_p += p
            # We are now using log prob to store regular noramlized probs
            log_probs[k] = p

        # Compute weighted mean pose
        for k in range(num_test_poses):
            norm_p = log_probs[k] / sum_p # log_probs no longer log
            means[i, 0] += norm_p * test_poses[k, 0]
            means[i, 1] += norm_p * test_poses[k, 1]
            
            # For angles, take the mean of the deviation to safely handle pi/-pi wrap
            d_th = test_poses[k, 2] - pth_prior
            d_th = (d_th + np.pi) % (2*np.pi) - np.pi
            means[i, 2] += norm_p * d_th
            
        means[i, 2] = (pth_prior + means[i, 2] + np.pi) % (2*np.pi) - np.pi
            
        # Compute covariance matrix
        for k in range(num_test_poses):
            norm_p = log_probs[k] / sum_p
            dx_cov = test_poses[k, 0] - means[i, 0]
            dy_cov = test_poses[k, 1] - means[i, 1]
            
            dth_cov = test_poses[k, 2] - means[i, 2]
            dth_cov = (dth_cov + np.pi) % (2*np.pi) - np.pi
            
            covs[i, 0, 0] += norm_p * dx_cov * dx_cov
            covs[i, 0, 1] += norm_p * dx_cov * dy_cov
            covs[i, 0, 2] += norm_p * dx_cov * dth_cov
            
            covs[i, 1, 0] += norm_p * dy_cov * dx_cov
            covs[i, 1, 1] += norm_p * dy_cov * dy_cov
            covs[i, 1, 2] += norm_p * dy_cov * dth_cov
            
            covs[i, 2, 0] += norm_p * dth_cov * dx_cov
            covs[i, 2, 1] += norm_p * dth_cov * dy_cov
            covs[i, 2, 2] += norm_p * dth_cov * dth_cov
            
        # Global weight of this particle
        weights[i] = np.log(sum_p) + max_log_prob

    return means, covs, weights


class PFScanMatcher:
    def __init__(self, env:Environment, n_part: int, ang_noise=0.20, vel_noise=0.20, resampling_temperature=8.0):
        self.env = env
        self.distmap = distance_transform_edt(~env.map.grid) * self.env.map.dx
        self.resampling_temp = resampling_temperature
        
        self.particles = np.random.rand(n_part, 3)
        self.particles[:, :2] *= self.env.l
        self.particles[:, 2] = self.particles[:, 2] * 2 * np.pi - np.pi
        
        self.n_part = n_part
        self.ang_noise = ang_noise
        self.vel_noise = vel_noise
        self.weights = np.ones(n_part) / n_part
        
        # Scan Matcher Parameters
        self.xy_step = self.env.map.dx      # Resolution of spatial search
        self.n_xy_steps = 2                 # +/- 2 steps -> 5x5 grid
        self.th_step = 0.05                 # ~2.8 degrees resolution
        self.n_th_steps = 2                 # +/- 2 steps -> 5 angles
        # Total local search space: 5 * 5 * 5 = 125 poses per particle

    def step_env(self, dt: float):
        action, scan = self.env.step(dt)
        return action, scan

    def update_particles(self, action: tuple[float, float], scan: np.ndarray, dt: float):
        # Odometry model to generate priors before scanmatching
        std_v = self.vel_noise * abs(action[0]) + 0.05
        std_w = self.ang_noise * abs(action[1]) + 0.05

        vel = action[0] + np.random.randn(self.n_part) * std_v
        omega = action[1] + np.random.randn(self.n_part) * std_w

        priors = self.particles.copy()
        priors[:, 0] += vel * np.cos(priors[:, 2]) * dt
        priors[:, 1] += vel * np.sin(priors[:, 2]) * dt
        priors[:, 2] += omega * dt
        priors[:, 2] = (priors[:, 2] + np.pi) % (2 * np.pi) - np.pi

        # Means are the mean position of the particle after the scanmatch
        # covs are the convariances
        # log_weights are the weight of each particle
        sigsqr = 0.20 ** 2
        means, covs, log_weights = correlative_scan_match(
            priors, scan, self.env.robot.angles, self.distmap, self.env.map.dx,
            self.env.max_lidar_range, sigsqr,
            self.n_xy_steps, self.xy_step, self.n_th_steps, self.th_step,
            z_hit=0.95, z_rand=0.05
        )

        # Draw particle poses from computed distribution
        for i in range(self.n_part):
            # Add small noise to diagonal to make sure matrix is invertible
            safe_cov = covs[i] + np.eye(3) * 1e-6
            try:
                new_pose = np.random.multivariate_normal(means[i], safe_cov)
            except np.linalg.LinAlgError:
                new_pose = means[i] # Fallback if math fails
            
            self.particles[i] = new_pose
            
        self.particles[:, 2] = (self.particles[:, 2] + np.pi) % (2 * np.pi) - np.pi

        # Update particle weights
        log_weights = log_weights / self.resampling_temp
        log_weights = log_weights - np.max(log_weights)
        weights = np.exp(log_weights)
        self.weights = weights / np.sum(weights)

        # Resample
        n_eff = 1.0 / (np.sum(self.weights**2) + 1e-10)
        if n_eff < self.n_part / 2.0:
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

        centroid = (np.sum(self.weights[:, None] * self.particles[:, :2], axis=0) / np.sum(self.weights) / self.env.map.dx).astype(np.int64)
        cv2.circle(img_true, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)

        xp = self.particles[:, 0]
        yp = self.particles[:, 1]
        for x, y in zip(xp, yp):
            xi = int(x / self.env.map.dx)
            yi = int(y / self.env.map.dx)
            cv2.circle(img_guess, (xi, yi), 2, (255, 0, 0), -1)
            
        return img_true, img_guess