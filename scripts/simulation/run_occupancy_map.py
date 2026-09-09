import cv2
import numpy as np
from pathlib import Path
from slam import *

def fit_to_canvas(img, max_w, max_h, bg_color=(127, 127, 127)):
    """Scales an image to fit inside max_w x max_h while keeping aspect ratio."""
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    pad_top = (max_h - new_h) // 2
    pad_bottom = max_h - new_h - pad_top
    pad_left = (max_w - new_w) // 2
    pad_right = max_w - new_w - pad_left
    
    return cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right, 
        cv2.BORDER_CONSTANT, value=bg_color
    )

def main():
    my_map = GridMap.from_file(DATA_DIR / Path("map1.npz"))
    robot = Robot(n_rays=20, spread=np.pi, speed=0.5, angvel=1.0, reactrange=3, k=0.05)
    env = Environment(robot=robot, map=my_map)
    
    mapper = OccupancyMapper(env, cellsize=(64, 64))
    dt = 0.20
    
    PANEL_SIZE = 600 
    
    while True:
        action, scan = mapper.step_env(dt)
        mapper.update_map(action, scan, dt)
        
        img_true, img_pred = mapper.render(lidar_scan=scan)

        if len(img_pred.shape) == 2:
            img_pred = cv2.cvtColor(img_pred, cv2.COLOR_GRAY2BGR)

        display_frame1 = fit_to_canvas(img_true, PANEL_SIZE, PANEL_SIZE, bg_color=(50, 50, 50))
        display_frame2 = fit_to_canvas(img_pred, PANEL_SIZE, PANEL_SIZE, bg_color=(127, 127, 127))

        combined_frame = cv2.hconcat([display_frame1, display_frame2])
        cv2.imshow("robotest", combined_frame)
        
        key = cv2.waitKey(int(dt * 100)) & 0xFF
        if key == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()