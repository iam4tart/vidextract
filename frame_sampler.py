import cv2
import os

def sample_frames(video_path: str, output_dir: str = "/tmp/vidextract_frames", fps: int = 1) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    video_fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_every = max(1, int(video_fps/fps))
    
    print(f"Video: {video_fps:.1f}fps, {total_frames} frames — sampling every {sample_every} frames")
    
    saved = []
    frame_idx = 0
    
    while True:
        success, frame = capture.read()
        if not success:
            break
        if frame_idx % sample_every == 0:
            path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(path, frame)
            saved.append(path)
        frame_idx += 1
        
    capture.release()
    print(f"Saved {len(saved)} frames to {output_dir}")
    return saved