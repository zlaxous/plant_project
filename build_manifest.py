"""Build manifest using cmd subprocess for ()-compatible path handling."""
import json, random, subprocess
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = PROJECT_ROOT / "plant_disease_detector"

CANONICAL = [
    "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew","Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight","Corn_(maize)___healthy",
    "Grape___Black_rot","Grape___Esca_(Black_Measles)","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)","Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot","Peach___healthy",
    "Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy",
    "Potato___Early_blight","Potato___Late_blight","Potato___healthy",
    "Raspberry___healthy","Soybean___healthy","Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch","Strawberry___healthy",
    "Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight",
    "Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus","Tomato___healthy",
]

ALIASES = {
    "Pepper__bell___Bacterial_spot":"Pepper,_bell___Bacterial_spot",
    "Pepper__bell___healthy":"Pepper,_bell___healthy",
    "Tomato_Bacterial_spot":"Tomato___Bacterial_spot",
    "Tomato_Early_blight":"Tomato___Early_blight",
    "Tomato_Late_blight":"Tomato___Late_blight",
    "Tomato_Leaf_Mold":"Tomato___Leaf_Mold",
    "Tomato_Septoria_leaf_spot":"Tomato___Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite":"Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato__Target_Spot":"Tomato___Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus":"Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato__Tomato_mosaic_virus":"Tomato___Tomato_mosaic_virus",
    "Tomato_healthy":"Tomato___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot":"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust":"Corn_(maize)___Common_rust_",
    "Corn___Northern_Leaf_Blight":"Corn_(maize)___Northern_Leaf_Blight",
    "Corn___healthy":"Corn_(maize)___healthy",
    "Background_without_leaves":"Tomato___healthy",
}

def normalize(name):
    if name in CANONICAL: return name
    if name in ALIASES: return ALIASES[name]
    n = name.strip().replace("__","___")
    if n in CANONICAL: return n
    return None

def main():
    r = subprocess.run(["cmd","/c","type","_all_images.txt"], capture_output=True, text=True, timeout=60)
    lines = [l.strip() for l in r.stdout.splitlines() if l.strip()]
    print(f"Total files: {len(lines)}")
    
    samples = []
    seen = set()
    for fp in lines:
        if fp in seen: continue
        seen.add(fp)
        parts = fp.replace("/","\\").split("\\")
        cn = None
        for p in parts:
            n = normalize(p)
            if n: cn = n; break
        if not cn: continue
        src = "vipoooool/new-plant-diseases-dataset" if "new_plant" in fp.lower() else "abdallahalidev/plantvillage-dataset"
        samples.append({"image_path": fp, "class_name": cn, "class_id": CANONICAL.index(cn), "source": src})
    
    print(f"Samples: {len(samples)}")
    if not samples: return
    
    rng = random.Random(42)
    by_class = defaultdict(list)
    for s in samples: by_class[s["class_id"]].append(s)
    for cs in by_class.values():
        rng.shuffle(cs); n = len(cs); tc = int(n*0.8); vc = tc+int(n*0.1)
        for i,s in enumerate(cs):
            s["split"] = "train" if i < tc else "val" if i < vc else "test"
    
    sc = Counter(s["split"] for s in samples)
    cc = defaultdict(int)
    for s in samples: cc[s["class_name"]] += 1
    
    manifest = {"sources":["abdallahalidev/plantvillage-dataset","vipoooool/new-plant-diseases-dataset"],"num_classes":len(CANONICAL),"seed":42,"samples":samples}
    health = {"total_samples":len(samples),"split_counts":dict(sc),"class_counts":dict(sorted(cc.items())),"low_count_classes":sorted([n for n,c in cc.items() if c<50]),"low_count_threshold":50}
    lm = {str(i):c for i,c in enumerate(CANONICAL)}
    
    (PACKAGE_ROOT / "label_map.json").write_text(json.dumps(lm, indent=2))
    (PACKAGE_ROOT / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (PACKAGE_ROOT / "data" / "dataset_health.json").write_text(json.dumps(health, indent=2))
    print("\n=== Dataset Health ===")
    print(json.dumps(health, indent=2))

if __name__ == "__main__": main()