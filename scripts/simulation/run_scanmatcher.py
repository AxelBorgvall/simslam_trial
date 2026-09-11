import numpy as np
from slam import * 



def main():
    my_map = GridMap.from_file(DATA_DIR/Path("map1.npz"))
    robot = Robot(n_rays=20, spread=np.pi, speed=0.5, angvel=1.0,reactrange=3,k=0.05)
    env = Environment(robot=robot, map=my_map)
    localizer=PFScanMatcher(env,200,resampling_temperature=8.0)
    dt = 0.20
    scale=2
    while True:
        action,scan=localizer.step_env(dt)
        localizer.update_particles(action,scan,dt)
        img_true,img_pred=localizer.render(lidar_scan=scan)

        # Scale each image up 
        h1, w1 = img_true.shape[:2]
        display_frame1 = cv2.resize(img_true, (w1 * scale, h1 * scale), interpolation=cv2.INTER_NEAREST)
        h2, w2 = img_pred.shape[:2]
        display_frame2 = cv2.resize(img_pred, (w2 * scale, h2 * scale), interpolation=cv2.INTER_NEAREST)
        
        # Concatenate images
        combined_frame = cv2.hconcat([display_frame1, display_frame2])
        cv2.imshow("robotest", combined_frame)
        
        # Wait for dt milliseconds. If 'q' is pressed, exit.
        key = cv2.waitKey(int(dt * 20)) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()