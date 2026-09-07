import json
import numpy as np
from pathlib import Path
from .gridmap import GridMap
from numba import njit, prange
import cv2


@njit
def bresenham_ray(grid, x0, y0, x1, y1, max_range, dx, rx, ry):
    rows, cols = grid.shape

    dx_idx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy_idx = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx_idx + dy_idx

    while True:
        if x0 < 0 or x0 >= cols or y0 < 0 or y0 >= rows:
            return max_range

        if grid[y0, x0]:
            hit_x_m = (x0 * dx) + (dx / 2.0)
            hit_y_m = (y0 * dx) + (dx / 2.0)
            dist = np.sqrt((hit_x_m - rx) ** 2 + (hit_y_m - ry) ** 2)
            return min(dist, max_range)

        if x0 == x1 and y0 == y1:
            return max_range

        e2 = 2 * error
        if e2 >= dy_idx:
            if x0 == x1:
                break
            error += dy_idx
            x0 += sx
        if e2 <= dx_idx:
            if y0 == y1:
                break
            error += dx_idx
            y0 += sy

    return max_range


@njit(parallel=True)
def simulate_lidar_kernel(grid, rx, ry, rtheta, angles, max_range, dx):
    n_rays = len(angles)
    distances = np.empty(n_rays, dtype=np.float64)

    start_x = int(rx / dx)
    start_y = int(ry / dx)

    for i in prange(n_rays):
        ray_angle = rtheta + angles[i]

        end_x_m = rx + max_range * np.cos(ray_angle)
        end_y_m = ry + max_range * np.sin(ray_angle)

        end_x = int(end_x_m / dx)
        end_y = int(end_y_m / dx)

        distances[i] = bresenham_ray(
            grid, start_x, start_y, end_x, end_y, max_range, dx, rx, ry
        )
    return distances


def dist(r: tuple, r0: tuple):
    x, y = r
    x0, y0 = r0
    return np.sqrt((x - x0) ** 2 + (y - y0) ** 2)


class Robot:
    def __init__(self, n_rays, spread, speed, angvel, reactrange=1.3,k=0.2):
        self.n_rays = n_rays
        self.spread = spread
        self.speed = speed
        self.angvel = angvel
        self.reactrange = reactrange
        self.angles = self.angles = np.linspace(
            -self.spread / 2, self.spread / 2, self.n_rays, dtype=np.float64
        )
        self.k=k

    def move(self, lidardata: np.ndarray):
        close_mask=lidardata<self.reactrange
        dists=lidardata[close_mask]
        angles=self.angles[close_mask]
        magnitudes = -self.k / (dists**2 + 1e-6)
        fx=np.sum(magnitudes*np.cos(angles))
        fy=np.sum(magnitudes*np.sin(angles))

        vector_x=self.speed+fx
        vector_y=fy
        tgt_heading=np.arctan2(vector_y,vector_x)
        omega=np.clip(tgt_heading*2.0,-self.angvel,self.angvel)
        v = np.clip(vector_x, -self.speed, self.speed)
        return v,omega


class Environment:
    def __init__(self, robot: Robot, map: GridMap):
        self.robot = robot
        self.map = map
        self.l = map.grid.shape[0] * map.dx
        self.n_side = map.grid.shape[0]
        self.max_lidar_range = 8.0
        self.lidarscan=None

        self.X, self.Y = np.meshgrid(
            np.linspace(0, self.l, map.grid.shape[0]),
            np.linspace(0, self.l, map.grid.shape[0]),
        )

        # PLace the robot
        self.robopos = None
        for x in np.arange(1,self.l-1,0.5):
            for y in np.arange(1, self.l-1, 0.5):
                close2robo = dist((x, y), (self.X, self.Y)) < 0.25
                if np.all(~map.grid[close2robo]):
                    self.robopos = (x, y, 0.0)
                    break
            else:
                continue
            break
        if self.robopos is None:
            raise ValueError("We couldnt find anywhere to put your robo. Sorry")
        self.angles = np.linspace(
            -robot.spread / 2, robot.spread / 2, robot.n_rays, dtype=np.float64
        )
        print(self.robopos)

    def lidardata(self):
        rx, ry, rtheta = self.robopos

        # Superfast binary call
        distances = simulate_lidar_kernel(
            self.map.grid,
            rx,
            ry,
            rtheta,
            self.angles,
            self.max_lidar_range,
            self.map.dx,
        )

        noise = np.random.normal(0, 0.03, size=distances.shape)
        noisy_distances = distances + noise

        # Clip to [0,maxrange]
        return np.clip(noisy_distances, 0, self.max_lidar_range)

    def step(self, dt: float):
        if self.lidarscan is None:
            self.lidarscan=self.lidardata()
        v, omega = self.robot.move(self.lidarscan)
        
        x, y, theta = self.robopos
        x_new = x + (v * np.cos(theta) * dt)
        y_new = y + (v * np.sin(theta) * dt)
        theta_new = theta + (omega * dt)
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi
        self.robopos = (x_new, y_new, theta_new)
        
        lidar_scan_slam = self.lidardata()
        self.lidarscan=lidar_scan_slam.copy()
        
        return (v, omega), lidar_scan_slam

    def render(self, lidar_scan=None):
        h, w = self.map.grid.shape
        img = np.ones((h, w, 3), dtype=np.uint8) * 255 
        
        img[self.map.grid]=np.zeros(3,dtype=np.uint8)


        rx, ry, rtheta = self.robopos
        px = int(rx / self.map.dx)
        py = int(ry / self.map.dx)
        if lidar_scan is not None:
            for angle, dist in zip(self.angles, lidar_scan):
                ray_theta = rtheta + angle
                ex = int((rx + dist * np.cos(ray_theta)) / self.map.dx)
                ey = int((ry + dist * np.sin(ray_theta)) / self.map.dx)
                cv2.line(img, (px, py), (ex, ey), (200, 255, 200), 1)

        radius_px = max(2, int(0.2 / self.map.dx)) 
        cv2.circle(img, (px, py), radius_px, (255, 0, 0), -1)

        hx = int(px + radius_px * 2 * np.cos(rtheta))
        hy = int(py + radius_px * 2 * np.sin(rtheta))
        cv2.line(img, (px, py), (hx, hy), (0, 0, 255), 2)

        return img

    def render_robot_to_img(self,img:np.ndarray,lidar_scan=None):
        h, w = self.map.grid.shape

        rx, ry, rtheta = self.robopos
        px = int(rx / self.map.dx)
        py = int(ry / self.map.dx)
        if lidar_scan is not None:
            for angle, dist in zip(self.angles, lidar_scan):
                ray_theta = rtheta + angle
                ex = int((rx + dist * np.cos(ray_theta)) / self.map.dx)
                ey = int((ry + dist * np.sin(ray_theta)) / self.map.dx)
                cv2.line(img, (px, py), (ex, ey), (200, 255, 200), 1)

        radius_px = max(2, int(0.2 / self.map.dx)) 
        cv2.circle(img, (px, py), radius_px, (255, 0, 0), -1)

        hx = int(px + radius_px * 2 * np.cos(rtheta))
        hy = int(py + radius_px * 2 * np.sin(rtheta))
        cv2.line(img, (px, py), (hx, hy), (0, 0, 255), 2)

        return
