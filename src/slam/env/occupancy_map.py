import json
import numpy as np
from pathlib import Path
from .gridmap import GridMap
from numba import njit, prange
from .robot import *
import cv2


@njit
def update_segment_bresenham(
    grid, cx, cy, chunk_size, gx0, gy0, gx1, gy1, l_free, l_occ
):
    # gx0, gy0, gx1, gy1 are gloabal cell indices
    dx_idx = abs(gx1 - gx0)
    sx = 1 if gx0 < gx1 else -1
    dy_idx = -abs(gy1 - gy0)
    sy = 1 if gy0 < gy1 else -1
    error = dx_idx + dy_idx

    x = gx0
    y = gy0

    chunk_min_x = cx * chunk_size
    chunk_max_x = chunk_min_x + chunk_size
    chunk_min_y = cy * chunk_size
    chunk_max_y = chunk_min_y + chunk_size

    while True:
        # Check if the global ray is currently passing through THIS chunk
        if (
            x >= chunk_min_x
            and x < chunk_max_x
            and y >= chunk_min_y
            and y < chunk_max_y
        ):
            # Convert global index to local 64x64 chunk index
            lx = x - chunk_min_x
            ly = y - chunk_min_y

            # If it's the end of the ray, it's an obstacle hit. Otherwise, it's free space.
            if x == gx1 and y == gy1:
                grid[ly, lx] += l_occ
            else:
                grid[ly, lx] += l_free

            # Clamp log-odds so the map doesn't get infinitely stuck on ghosts
            if grid[ly, lx] > 10.0:
                grid[ly, lx] = 10.0
            elif grid[ly, lx] < -10.0:
                grid[ly, lx] = -10.0

        if x == gx1 and y == gy1:
            break

        e2 = 2 * error
        if e2 >= dy_idx:
            if x == gx1:
                break
            error += dy_idx
            x += sx
        if e2 <= dx_idx:
            if y == gy1:
                break
            error += dx_idx
            y += sy


def bresenham_chunks(cx0, cy0, cx1, cy1):
    """Standard Bresenham to find which chunks a line traverses."""
    chunks = []
    dx = abs(cx1 - cx0)
    sx = 1 if cx0 < cx1 else -1
    dy = -abs(cy1 - cy0)
    sy = 1 if cy0 < cy1 else -1
    error = dx + dy

    while True:
        chunks.append((cx0, cy0))
        if cx0 == cx1 and cy0 == cy1:
            break
        e2 = 2 * error
        if e2 >= dy:
            if cx0 == cx1:
                break
            error += dy
            cx0 += sx
        if e2 <= dx:
            if cy0 == cy1:
                break
            error += dx
            cy0 += sy
    return chunks


class OccupancyMapper:
    def __init__(self, env: Environment, cellsize: tuple = (64, 64)):
        self.env = env
        self.outer_map: dict[tuple, np.ndarray] = {}  # Log odds of obstacle present
        self.robot_pose = (0.0, 0.0, 0.0)
        self.cellsize = np.array(cellsize)

        self.window_offset = (0, 0)

    def new_map_segment(self):
        return np.zeros(self.cellsize, dtype=np.float32)

    def step_env(self, dt: float):
        # Noisy scan, clean action
        action, scan = self.env.step(dt)
        return action, scan


    def update_map(self, action: tuple[float, float], scan: np.ndarray, dt: float):
        # Update robot position
        v, omega = action
        x, y, theta = self.robot_pose
        x_new = x + (v * np.cos(theta) * dt)
        y_new = y + (v * np.sin(theta) * dt)
        theta_new = theta + (omega * dt)
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi
        self.robot_pose = (x_new, y_new, theta_new)

        lidar_angles = self.env.robot.angles + self.robot_pose[2]
        lidar_points = np.array([self.robot_pose[0], self.robot_pose[1]]) + scan[
            :, np.newaxis
        ] * np.stack((np.cos(lidar_angles), np.sin(lidar_angles)), axis=-1)

        chunk_size = self.cellsize[0]  # 64
        dx=self.env.map.dx

        # Robot's starting global cell
        gx0 = int(self.robot_pose[0] / dx)
        gy0 = int(self.robot_pose[1] / dx)
        cx0 = gx0 // chunk_size
        cy0 = gy0 // chunk_size

        for i in range(len(scan)):
            gx1 = int(lidar_points[i, 0] / dx)
            gy1 = int(lidar_points[i, 1] / dx)

            cx1 = gx1 // chunk_size
            cy1 = gy1 // chunk_size

            base_chunks = bresenham_chunks(cx0, cy0, cx1, cy1)

            touched_chunks = set()
            for cx, cy in base_chunks:
                touched_chunks.update(
                    [
                        (cx, cy),
                        (cx + 1, cy),
                        (cx, cy + 1),
                        (cx + 1, cy + 1),
                        (cx - 1, cy),
                        (cx, cy - 1),
                        (cx - 1, cy - 1),
                    ]
                )

            for cx, cy in touched_chunks:
                if (cx, cy) not in self.outer_map:
                    self.outer_map[(cx, cy)] = self.new_map_segment()

                update_segment_bresenham(
                    self.outer_map[(cx, cy)],
                    cx,
                    cy,
                    chunk_size,
                    gx0,
                    gy0,
                    gx1,
                    gy1,
                    l_free=-0.4,
                    l_occ=0.85,
                )

    def render(self, lidar_scan=None):
        h, w = self.env.map.grid.shape
        img_true = np.ones((h, w, 3), dtype=np.uint8) * 255

        if not self.outer_map:
            return img_true, np.zeros((self.cellsize[1], self.cellsize[0]), dtype=np.uint8)

        # Render guessed map
        keys = list(self.outer_map.keys())
        minx = min(k[0] for k in keys)
        maxx = max(k[0] for k in keys)
        miny = min(k[1] for k in keys)
        maxy = max(k[1] for k in keys)

        pixel_width = (maxx - minx + 1) * self.cellsize[0]
        pixel_height = (maxy - miny + 1) * self.cellsize[1]
        img_guess_log_odds = np.zeros((pixel_height, pixel_width), dtype=np.float32)
        for (cx, cy), cell_data in self.outer_map.items():
            x0 = (cx - minx) * self.cellsize[0]
            y0 = (cy - miny) * self.cellsize[1]
            img_guess_log_odds[y0:y0+self.cellsize[1], x0:x0+self.cellsize[0]] = cell_data

        prob = 1.0 - (1.0 / (1.0 + np.exp(img_guess_log_odds)))
        img_guess = (255 - (prob * 255)).astype(np.uint8)

        img_true[self.env.map.grid] = np.zeros(3, dtype=np.uint8)
        self.env.render_robot_to_img(img_true, lidar_scan)

        return img_true, img_guess




