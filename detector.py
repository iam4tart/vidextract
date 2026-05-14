import os
import gc
import torch
from PIL import Image
from ultralytics import YOLOWorld
 
def __clear_vram():
    gc.collect()
    torch.cuda.empty_cache()
    
def __used_vram():
    print(torch.cuda.memory_allocated() / 1e9, "GB")
    
def load_classes(path: str ="prompts.txt") -> list[str]:
    if not os.path.exists(path):
        return [
            "object", "tool", "device", "machine", "component",
            "equipment", "structure", "vehicle", "motor",
            "wheel", "blade", "pipe", "sensor", "gear"
        ]
    with open(path) as f:
        classes = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(classes)} classes from {path}")
    return classes

def run_yolo_world(frame_paths: list[str]) -> list[dict]:
    print("Loading YOLO-World...")
    model = YOLOWorld("yolov8l-worldv2.pt")
    model.set_classes(load_classes())
    
    results = []
    for path in frame_paths:
        detections = model.predict(path, conf=0.15, verbose=False)[0]
        for box, cls, conf in zip(
            detections.boxes.xyxy.cpu().tolist(),
            detections.boxes.cls.cpu().tolist(),
            detections.boxes.conf.cpu().tolist()
        ):
            results.append({
                "label": model.names[int(cls)],
                "confidence": round(float(conf), 3),
                "bbox": [round(v,1) for v in box],
                "frame_path": path,
                "source": "yolo"
            })
    
    del model
    __clear_vram()
    print(f"YOLO-World: {len(results)} detections")
    return results

# dense captioning task for this VLM
def run_florence2(frame_paths: list[str]) -> list[dict]:
    from transformers import AutoProcessor,AutoModelForCausalLM
    
    print("Loading Florence-2...")
    # suitable precision to run on my local machine
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base",
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).cuda()
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-base",
        trust_remote_code=True
    )
    model.eval()
    
    results = []
    for path in frame_paths:
        image = Image.open(path).convert("RGB")
        inputs = processor(
            text="<DENSE_REGION_CAPTION>",
            images=image,
            return_tensors="pt"
        )
        inputs = {k: v.cuda() if hasattr(v, "cuda") else v for k,v in inputs.items()}
        # input image values from float32 to float16
        inputs['pixel_values'] = inputs['pixel_values'].half()
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids = inputs["input_ids"],
                pixel_values = inputs["pixel_values"],
                max_new_tokens=128
            )
            
        output = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            output,
            task="<DENSE_REGION_CAPTION>",
            image_size=image.size
        )
        
        captions = parsed.get("<DENSE_REGION_CAPTION>", {})
        labels = captions.get("labels", [])
        bboxes = captions.get("bboxes", [])
        
        for label, bbox in zip(labels, bboxes):
            label = label.strip()
            if label:
                results.append({
                 "label": label,
                    "confidence": None, # florence doesnt give confidence scores
                    "bbox": [round(v, 1) for v in bbox],
                    "frame_path": path,
                    "source": "florence2"
                })

    del model, processor
    __clear_vram()
    print(f"Florence-2: {len(results)} detections")
    return results

# currently deduplication is NOT handled
def detect_all(frame_paths: list[str]) -> list[dict]:
    yolo_results = run_yolo_world(frame_paths)
    florence_results = run_florence2(frame_paths)
    return yolo_results + florence_results