import torch
import os
import sys

model_path = "/home/robot/WCL/GraspCoT/checkpoints/llava-graspcot-lora/checkpoint-2"
bin_file = os.path.join(model_path, "mm_projector.bin")



if not os.path.exists(bin_file):
    print(f"File not found: {bin_file}")
    sys.exit(1)

print(f"Loading {bin_file}...")
state_dict = torch.load(bin_file, map_location="cpu")

print(f"Total keys: {len(state_dict)}")
print("Sample keys:")
for k in list(state_dict.keys())[:10]:
    print(k)

print("\nChecking for GraspNet keys (grasp_tower)...")
grasp_keys = [k for k in state_dict.keys() if "grasp_tower" in k]
print(f"Found {len(grasp_keys)} grasp_tower keys.")
if grasp_keys:
    print("Sample grasp_tower keys:")
    for k in grasp_keys[:5]:
        print(k)
        
    # Check bias values
    bias_keys = [k for k in grasp_keys if "bias" in k]
    if bias_keys:
        print(f"\nChecking bias values for {bias_keys[0]}:")
        print(state_dict[bias_keys[0]])
        if torch.all(state_dict[bias_keys[0]] == 0):
            print("WARNING: Bias is all zeros!")
        else:
            print("Bias is NOT all zeros.")

print("\nChecking for det_head keys...")
det_keys = [k for k in state_dict.keys() if "det_head" in k]
print(f"Found {len(det_keys)} det_head keys.")
if det_keys:
    print("Sample det_head keys:")
    for k in det_keys[:5]:
        print(k)

print("\nChecking for mm_projector keys...")
proj_keys = [k for k in state_dict.keys() if "mm_projector" in k]
print(f"Found {len(proj_keys)} mm_projector keys.")
if proj_keys:
    print("Sample mm_projector keys:")
    for k in proj_keys[:5]:
        print(k)
