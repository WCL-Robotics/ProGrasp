import argparse
import torch
import numpy as np
from llava.model.builder import load_pretrained_grasp_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    get_model_name_from_path,
    tokenizer_image_token,
    tokenizer_special_token_v2
)
from llava.constants import IMAGE_TOKEN_INDEX

# from llava.train.GraspcotDataset import GraspcotDataset_Test
# from llava.train.Graspcot_Task_Dataset import GraspcotDataset_Test
from llava.train.Graspcot_Grasp_Dataset import GraspcotDataset_Test
from llava.train.train import DataCollatorForSupervisedDataset

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import transformers
from typing import Dict

import os
import pickle
from tqdm import tqdm

import random
from llava import conversation as conversation_lib
import time
import gc
from collections import defaultdict
from sklearn.metrics import average_precision_score



# def precess_data(tokenizer, instance):

#     # input_ids = tuple(instance[key] for key in ("input_ids"))
#     # input_ids = input_ids.unsqueeze(0).cuda()
#     # labels = labels.unsqueeze(0).cuda() 
#     # input_ids = input_ids[:, :tokenizer.model_max_length]
#     # labels = labels[:, :tokenizer.model_max_length]
#     batch = dict()
    
#     # ==========Too many videos or images may lead to OOM, so we encode them one by one======================
#     if 'image' in instance:
#         images = instance["image"].unsqueeze(0).to(torch.bfloat16).cuda()
#         batch['images'] = images  # (B, V, H, W)
#     if 'grasps' in instance:
#         grasps = instance["grasps"].unsqueeze(0).to(torch.bfloat16).cuda()
#         batch['gs'] = grasps  # (B, N, 7)

#     if 'depth' in instance:
#         depths = instance["depth"].unsqueeze(0).to(torch.bfloat16).cuda()
#         poses = instance["pose"].unsqueeze(0).to(torch.bfloat16).cuda()
#         correct_answers = instance["correct_answer"]

#         intrinsics = instance["intrinsic"].unsqueeze(0).to(torch.bfloat16).cuda()
        
#         batch['depths'] = depths  # (B, V, H, W)
#         batch['poses'] = poses  # (B, V, 4, 4)
#         batch['intrinsics'] = intrinsics  # (B, 4, 4)
#         batch['correct_answers'] = correct_answers

#         # batch['clicks'] = clicks  # (num_clicks, 3)
#     if 'pure_img' in instance:
#         pure_imgs = instance["pure_img"].unsqueeze(0).to(torch.bfloat16).cuda()
#         batch['pure_imgs'] = pure_imgs # (B, 3, 168, 168)
#     return batch

def precess_data(tokenizer, instance):

    input_ids, labels = tuple(instance[key] for key in ("input_ids", "labels"))
    input_ids = input_ids.unsqueeze(0).cuda()
    labels = labels.unsqueeze(0).cuda() 
    input_ids = input_ids[:, :tokenizer.model_max_length]
    labels = labels[:, :tokenizer.model_max_length]
    batch = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )
    batch["input_token_len"] = input_ids.shape[1]
    # ==========Too many videos or images may lead to OOM, so we encode them one by one======================
    if 'image' in instance:
        images = instance["image"].unsqueeze(0).to(torch.bfloat16).cuda()
        batch['images'] = images  # (B, V, H, W)


    if 'grasps' in instance:
        grasps = instance["grasps"].unsqueeze(0).to(torch.float32).cuda()
        batch['gs'] = grasps  # (B, N, 7)
        
    if 'pc' in instance:
        pcs = instance["pc"].unsqueeze(0).to(torch.float32).cuda()
        batch['pcs'] = pcs  # (B, N, 7)

    if 'depth' in instance:
        depths = instance["depth"].unsqueeze(0).to(torch.bfloat16).cuda()
        poses = instance["pose"].unsqueeze(0).to(torch.bfloat16).cuda()

        intrinsics = instance["intrinsic"].unsqueeze(0).to(torch.bfloat16).cuda()
        
        batch['depths'] = depths  # (B, V, H, W)
        batch['poses'] = poses  # (B, V, 4, 4)
        batch['intrinsics'] = intrinsics  # (B, 4, 4)

        # batch['clicks'] = clicks  # (num_clicks, 3)
    if 'pure_img' in instance:
        pure_imgs = instance["pure_img"].unsqueeze(0).to(torch.bfloat16).cuda()
        batch['pure_imgs'] = pure_imgs # (B, 3, 168, 168)
    return batch


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_path) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    eval_dataset = GraspcotDataset_Test(data_path, tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=None,
                eval_dataset=eval_dataset,
                data_collator=data_collator)

def check_grasp(gt, pred):
    fail = []
    for i in range(gt.shape[0]):
        if abs(gt[i] - pred[i]) >= 0.5:
            fail.append(i)
    
    print("fail grasps", len(fail))
    print("fail indices:", fail)
    return len(fail)

def safe_ap(y_true, y_score):
    """AP 需要同时有正/负样本，否则该 instance 的 AP 没意义，返回 None。"""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.size == 0:
        return None
    if (y_true == 1).sum() == 0 or (y_true == 0).sum() == 0:
        return None
    return float(average_precision_score(y_true, y_score))

def get_instance_key(sample: dict):
    """
    从 dataset sample 里提取 instance 标识。
    你可以把最匹配你数据集的 key 放到最前面。
    """
    candidate_keys = [
        "pc_path"
    ]
    for k in candidate_keys:
        if k in sample:
            v = sample[k]
            if torch.is_tensor(v):
                if v.numel() == 1:
                    v = v.item()
                else:
                    v = v.detach().cpu().numpy().tolist()
            return str(v)

    # 兜底：如果 sample 里有路径字段，就取 basename
    path_keys = ["image_path", "rgb_path", "depth_path", "path", "file_path", "data_path"]
    for k in path_keys:
        if k in sample:
            return os.path.basename(str(sample[k]))

    # 最后兜底：用 index（如果你传进来）
    if "index" in sample:
        return f"sample_{sample['index']}"
    return "unknown_instance"

def compute_instance_mAP(inst2labels, inst2scores, verbose=False):
    aps = []
    for inst in inst2scores.keys():
        ap = safe_ap(inst2labels[inst], inst2scores[inst])
        if ap is None:
            continue
        aps.append(ap)
        if verbose:
            print(f"[AP] {inst}: {ap:.4f}  (N={len(inst2labels[inst])})")
    mAP = float(np.mean(aps)) if len(aps) > 0 else float("nan")
    return mAP, len(aps)

def compute_top1_acc(inst2labels, inst2scores):
    accs = []
    for inst in inst2scores.keys():
        scores = np.array(inst2scores[inst])
        labels = np.array(inst2labels[inst])
        
        if len(scores) == 0:
            continue
            
        # 找到预测概率最大的索引
        max_idx = np.argmax(scores)
        
        # 检查对应标签是否为1
        is_correct = 1 if labels[max_idx] == 1 else 0
        accs.append(is_correct)
        
    top1_acc = float(np.mean(accs)) if len(accs) > 0 else 0.0
    return top1_acc, len(accs)

def update_global_buffers(global_scores, global_labels, scores, labels):
    """把一个 batch/一个 sample 的 scores/labels 加到全局列表里。"""
    global_scores.extend(np.asarray(scores, dtype=float).reshape(-1).tolist())
    global_labels.extend(np.asarray(labels, dtype=int).reshape(-1).tolist())

def compute_overall_ap(global_scores, global_labels):
    """把所有 instance 混在一起，算一次 AP（micro / overall AP）。"""
    y_score = np.asarray(global_scores, dtype=float)
    y_true  = np.asarray(global_labels, dtype=int)

    # AP 至少需要同时存在正/负样本
    if (y_true == 1).sum() == 0 or (y_true == 0).sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))

def eval_model(args):
    # Model
    disable_torch_init()

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    model_name = get_model_name_from_path(args.model_path)
    print("********************* load model Start *********************")
    tokenizer, model, processor, context_len = load_pretrained_grasp_model(
        args.model_path, args.model_base, model_name, use_flash_attn=True, torch_dtype=torch_dtype
    )
    print("********************* load model End *********************")

    model.eval()


    # if hasattr(model, 'model') and hasattr(model.model, 'grasp_tower'):
    #     print("Forcing grasp_tower to TRAIN mode due to missing BatchNorm stats...")
    #     model.model.grasp_tower.train()
    # elif hasattr(model, 'grasp_tower'):
    #     print("Forcing grasp_tower to TRAIN mode due to missing BatchNorm stats...")
    #     model.grasp_tower.train()

    # DEBUG: Check if BN stats are loaded correctly
    # print("DEBUG: Inspecting Grasp Tower BN Stats...")
    # bn_found = False
    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
    #         print(f"Layer: {name}")
    #         print(f"  running_mean[:5]: {module.running_mean[:5].cpu().tolist()}")
    #         print(f"  running_var[:5]:  {module.running_var[:5].cpu().tolist()}")
    #         print(f"  num_batches_tracked: {module.num_batches_tracked}")
    #         bn_found = True
            
    # if not bn_found:
    #     print("DEBUG: No BatchNorm1d layer found in grasp_tower.")

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "3d" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        # conv_mode = "llava_v0"
        conv_mode = args.conv_mode

    tokenizer.pad_token = tokenizer.unk_token
    if conv_mode in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[conv_mode]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    test_dataset = GraspcotDataset_Test(args.data_path, tokenizer=tokenizer)
    data_lens = len(test_dataset)

    model.config.use_dialogue = True


    global_scores = []
    global_labels = []


    len_data = len(test_dataset)
    num_val = len(test_dataset)
    data_list = list(range(len_data))
    selected_data = random.sample(data_list, num_val)

    grasp_dict = {}
    accuracy_cnt = 0
    avg_time = 0.0
    num_fail = 0
    preds_list = []

    inst2scores = defaultdict(list)  # instance -> [pred_score...]
    inst2labels = defaultdict(list)  # instance -> [0/1...]

    for i in tqdm(selected_data):
        instance = test_dataset.__getitem__(i)
        instance_key = get_instance_key(instance)
        gs_labels = instance["gs_labels"]
        questions = instance.pop("questions")
        inputs = precess_data(tokenizer, instance)
        inputs['gs'] = inputs['gs'].reshape(25, 6, 3)
        inputs['pcs'] = inputs['pcs'].repeat(inputs['gs'].shape[0], 1, 1)
        inputs['input_ids'] = inputs['input_ids'].repeat(inputs['gs'].shape[0], 1)
        inputs['labels'] = inputs['labels'].repeat(inputs['gs'].shape[0], 1)
        inputs['attention_mask'] = inputs['attention_mask'].repeat(inputs['gs'].shape[0], 1)
        inputs['images'] = inputs['images'].repeat(inputs['gs'].shape[0], 1, 1, 1, 1)
        inputs['depths'] = inputs['depths'].repeat(inputs['gs'].shape[0], 1, 1, 1)
        inputs['poses'] = inputs['poses'].repeat(inputs['gs'].shape[0], 1, 1, 1)
        inputs['intrinsics'] = inputs['intrinsics'].repeat(inputs['gs'].shape[0], 1, 1, 1)

        grasp_out = False
        start_time = time.time()
        with torch.inference_mode():
            grasp_outs, outputs = model(**inputs)
            preds = grasp_outs['all_cls_scores'][0]
            preds = preds.squeeze(0).squeeze(-1)
            gs_labels = gs_labels.to(preds.device).float()
            y_score = preds.detach().float().view(-1).cpu().numpy()
            y_true  = gs_labels.detach().float().view(-1).cpu().numpy()

            update_global_buffers(global_scores, global_labels, y_score, y_true)

            loss_fct = torch.nn.BCELoss()
            loss = loss_fct(preds, gs_labels.unsqueeze(1))
            # loss = loss_fct(preds, gs_labels)
            print("Grasp Prediction Loss:", loss.item())
            num_fail += check_grasp(gs_labels, preds)
            y_score = preds.detach().float().view(-1).cpu().numpy().tolist()
            y_true  = gs_labels.detach().float().view(-1).cpu().numpy().tolist()
            inst2scores[instance_key].extend(y_score)
            inst2labels[instance_key].extend(y_true)


        avg_time += time.time() - start_time

        del inputs, grasp_outs, outputs, preds, gs_labels, loss
        torch.cuda.empty_cache()
        gc.collect()

    # np.save("preds_list.npy", preds_list)

    print(f"Average Inference Time per Sample: {avg_time / num_val:.2f} seconds")
    print("PA:",1-num_fail/(25*num_val))
    instance_mAP, n_valid = compute_instance_mAP(inst2labels, inst2scores, verbose=False)
    print(f"Instance mAP (valid instances={n_valid}): {instance_mAP:.4f}")
    
    top1_acc, n_acc = compute_top1_acc(inst2labels, inst2scores)
    print(f"Top-1 Accuracy (valid instances={n_acc}): {top1_acc:.4f}")
    # overall_ap = compute_overall_ap(global_scores, global_labels)
    # print(f"Overall AP (micro, all instances merged): {overall_ap:.4f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/robot/WCL/GraspCoT/checkpoints/llava-graspcot-lora/")
    parser.add_argument("--model-base", type=str, default="/home/robot/WCL/GraspCoT/pretrained_llms/llava-3d-7b/")
    parser.add_argument("--output_dir", type=str, default="/home/robot/WCL/GraspCoT/checkpoints/llava-graspcot-lora/")
    parser.add_argument("--data-path", type=str, default="/media/robot/data/WCL/taskgrasp/taskgrasp_image")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    args = parser.parse_args()

    eval_model(args)
