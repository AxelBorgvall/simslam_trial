import numpy as np
import cv2
from slam import * 

def main():
    my_map = GridMap.from_file(DATA_DIR/Path("map1.npz"))
    robot = Robot(n_rays=20, spread=np.pi, speed=0.5, angvel=1.0, reactrange=3, k=0.05)
    env = Environment(robot=robot, map=my_map)
    localizer = PFLocalizer(env, 1000, resampling_temperature=15.0)
    
    dt = 0.20
    scale = 2
    
    # --- Video Recording Setup ---
    duration_seconds = 16
    fps = 30
    total_frames = duration_seconds * fps
    
    video_writer = None
    output_filename = "simulation_video.mp4"
    
    print(f"Recording {duration_seconds}s video at {fps} FPS ({total_frames} frames)...")
    
    for frame_idx in range(total_frames):
        action, scan = localizer.step_env(dt)
        localizer.update_particles(action, scan, dt)
        img_true, img_pred = localizer.render(lidar_scan=scan)

        # Scale each image up 
        h1, w1 = img_true.shape[:2]
        display_frame1 = cv2.resize(img_true, (w1 * scale, h1 * scale), interpolation=cv2.INTER_NEAREST)
        
        h2, w2 = img_pred.shape[:2]
        display_frame2 = cv2.resize(img_pred, (w2 * scale, h2 * scale), interpolation=cv2.INTER_NEAREST)
        
        # Concatenate images
        combined_frame = cv2.hconcat([display_frame1, display_frame2])
        
        # Initialize VideoWriter on the first frame once dimensions are known
        if video_writer is None:
            height, width = combined_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 format
            video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
            
        # Write the frame to the file
        video_writer.write(combined_frame)
        
        # Print progress every second
        if (frame_idx + 1) % fps == 0:
            print(f"Recorded {(frame_idx + 1) // fps} / {duration_seconds} seconds")

    # Clean up and save
    if video_writer is not None:
        video_writer.release()
    print(f"Video saved to {output_filename}")

if __name__ == "__main__":
    main()