
import numpy as np
from slam import * 



def main():
    my_map = GridMap.from_file(DATA_DIR/Path("map1.npz"))
    robot = Robot(n_rays=20, spread=np.pi, speed=0.5, angvel=1.0,reactrange=3,k=0.05)
    env = Environment(robot=robot, map=my_map)

    dt = 0.20

    

    print("Starting simulation... Press 'q' to quit.")
    
    while True:
        action, scan = env.step(dt)
        frame = env.render(lidar_scan=scan)
        
        # Scale up the image so it's easier to see on modern monitors
        # INTER_NEAREST prevents blurry edges when scaling pixel art/grids
        scale = 2 
        h, w = frame.shape[:2]
        display_frame = cv2.resize(frame, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        
        cv2.imshow("robotest", display_frame)
        
        # Wait for dt milliseconds. If 'q' is pressed, exit.
        key = cv2.waitKey(int(dt * 20)) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()