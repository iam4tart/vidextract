from collections import defaultdict

# deduplication is based on object labels only, not considering spatial bboxes

def deduplicate(detections: list[dict]) -> list[dict]:
    # i need all the object labels at the very least
    # yet currently we are biased towards florence because
    # it increases frames_seen but do not affect confidence
    
    buckets = defaultdict(lambda: {
        "confidence_sum": 0.0,
        "confidence_count": 0,
        "frames_seen": set(),
        "sources": set(),
        "bboxes": []
    })
    
    for d in detections:
        key = d["label"].lower().strip()
        
        conf = d["confidence"]
        
        # florence doesnt contribute to confidence
        if conf is not None:
            buckets[key]["confidence_sum"] += conf
            buckets[key]["confidence_count"] += 1
            
        buckets[key]["frames_seen"].add(d["frame_path"])
        buckets[key]["sources"].add(d["source"])
        buckets[key]["bboxes"].append(d["bbox"])

    merged = []
    
    for label, data in buckets.items():
        count = len(data["frames_seen"])
        
        avg_conf = (round(data["confidence_sum"]/data["confidence_count"], 3) if data["confidence_count"] > 0 else None)
        
        merged.append({
            "label": label,
            "avg_confidence": avg_conf,
            "frames_seen": count,
            "sources": list(data["sources"]),
            "sample_bbox": data["bboxes"][0]
        })
        
    # sort by frames_seen descending
    # so objects appearing in most frames comes first
    merged.sort(key=lambda x: -x["frames_seen"])
        
    print(f"Deduplicated to {len(merged)} unique objects")
    return merged