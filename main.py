import argparse
import json
import os
from frame_sampler import sample_frames
from detector import detect_all
from dedup import deduplicate
from tech_filter import score_objects
 
def main():
    parser = argparse.ArgumentParser(description="vidextract — video to re-engineerable objects JSON")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", default="objects.json", help="Output JSON path")
    parser.add_argument("--threshold", type=float, default=0.4, help="Minimum eng_score to include")
    parser.add_argument("--fps", type=int, default=1, help="Frames to sample per second")
    args = parser.parse_args()
 
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
 
    print(f"\n=== vidextract ===")
    print(f"Video    : {args.video}")
    print(f"Output   : {args.output}")
    print(f"Threshold: {args.threshold}")
    print(f"Sample   : {args.fps} fps\n")
    
    frames = sample_frames(args.video, fps=args.fps)
    detections = detect_all(frames)
    unique_objects = deduplicate(detections)
    tech_objects = score_objects(unique_objects, threshold=args.threshold)
    
    output = {
        "video": args.video,
        "total_frames_sampled": len(frames),
        "total_unique_detected": len(unique_objects),
        "total_tech_objects": len(tech_objects),
        "objects": tech_objects
    }
    
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    
 
    print(f"\n=== done ===")
    print(f"Found {len(tech_objects)} re-engineerable objects")
    print(f"Saved to {args.output}\n")

    print(f"{'OBJECT':<30} {'CATEGORY':<15} {'SCORE':<10}")
    print("-" * 60)

    for obj in tech_objects:
        print(
            f"{obj['label']:<30} "
            f"{obj['category']:<15} "
            f"{obj['eng_score']:.2f}"
        )
 
if __name__ == "__main__":
    main()