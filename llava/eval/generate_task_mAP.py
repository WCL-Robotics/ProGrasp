import argparse
import torch

from llava.model.builder import load_pretrained_grasp_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    get_model_name_from_path,
    tokenizer_image_token,
    tokenizer_special_token_v2
)
from llava.constants import IMAGE_TOKEN_INDEX

# from llava.train.GraspcotDataset import GraspcotDataset_Test
from llava.train.Graspcot_Task_Dataset import GraspcotDataset_Test

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


def precess_data(tokenizer, instance):

    # input_ids = tuple(instance[key] for key in ("input_ids"))
    # input_ids = input_ids.unsqueeze(0).cuda()
    # labels = labels.unsqueeze(0).cuda() 
    # input_ids = input_ids[:, :tokenizer.model_max_length]
    # labels = labels[:, :tokenizer.model_max_length]
    batch = dict()
    
    # ==========Too many videos or images may lead to OOM, so we encode them one by one======================
    if 'image' in instance:
        images = instance["image"].unsqueeze(0).to(torch.bfloat16).cuda()
        batch['images'] = images  # (B, V, H, W)

    if 'grasps' in instance:
        grasps = instance["grasps"].unsqueeze(0).to(torch.float32).cuda()
        batch['gs'] = grasps  # (B, N, 7)
    
    if 'input_ids' in instance:
        input_ids = instance["input_ids"].unsqueeze(0).cuda()
        batch['input_ids'] = input_ids  # (B, L)
    
    if 'labels' in instance:
        labels = instance["labels"].unsqueeze(0).cuda()
        batch['labels'] = labels  # (B, L)
        
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

from sklearn.metrics import average_precision_score
import numpy as np

def calculate_map(y_true, y_pred, y_score, classes):
    aps = []
    
    # Filter classes that exist in ground truth
    valid_classes = [c for c in classes if c in y_true]
    
    for cls in valid_classes:
        # Ground truth binary labels for current class
        y_true_cls = [1 if gt == cls else 0 for gt in y_true]
        
        # Retrieval score: If prediction matches class, use confidence score. Else 0.
        y_score_cls = [score if pred == cls else 0.0 for pred, score in zip(y_pred, y_score)]
        
        # Check if there are any positive samples (already filtered by valid_classes)
        if sum(y_true_cls) > 0:
            ap = average_precision_score(y_true_cls, y_score_cls)
            aps.append(ap)
            
    return np.mean(aps) if aps else 0.0

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

    num_val = 42
    len_data = len(test_dataset)
    data_list = list(range(len_data))
    selected_data = random.sample(data_list, num_val)

    grasp_dict = {}
    accuracy_cnt = 0
    avg_time = 0.0

    # Storage for mAP calculation
    y_true_list = []
    y_pred_list = []
    ypred_score_list = []
    class_set = set() # All unique classes

    for i in tqdm(selected_data):

        instance = test_dataset.__getitem__(i)
        gt_grasps = instance["grasps"]
        pc_path = instance["pc_path"]
        scene = instance["scene"]
        gs_labels = instance["gs_labels"]
        correct_answers = instance.pop("correct_answer")
        inputs = precess_data(tokenizer, instance)
        if "labels" in inputs:
             inputs.pop("labels", None)
        inputs["attention_mask"] = inputs["input_ids"].ne(tokenizer.pad_token_id)

        # outputs = model(**inputs)


        # generation 不需要 labels，去掉避免落入训练分支
        grasp_out = False

        # Iterative Generation Logic
        if 'questions' in instance:
            questions = instance['questions']
            
            # Initialize conversation
            conv = conversation_lib.default_conversation.copy()
            conv.messages = []
            
            print(f"\nProcessing scene {scene} with {len(questions)} turns (Iterative).")
            start_time = time.time()
            for step_idx, question in enumerate(questions):
                # 1. Append User Question
                conv.append_message(conv.roles[0], question)
                # 2. Append Assistant Placeholder
                conv.append_message(conv.roles[1], None)
                
                # 3. Get Prompt
                prompt = conv.get_prompt()
                
                # 4. Tokenize
                # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                input_ids = tokenizer_special_token_v2(prompt, tokenizer, return_tensors='pt').unsqueeze(0).cuda()
                # 5. Prepare Batch
                curr_batch = dict(
                    input_ids=input_ids,
                    attention_mask=input_ids.ne(tokenizer.pad_token_id),
                    grasp_out=grasp_out
                )
                # Copy multimodal inputs
                for k in ['images', 'depths', 'poses', 'intrinsics', 'pure_imgs', 'gs', 'pcs']:
                    if k in inputs:
                        curr_batch[k] = inputs[k]
                
                # 6. Generate
                with torch.inference_mode():
                    generated = model.generate(
                        **curr_batch,
                        max_new_tokens=args.max_new_tokens,
                        # temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                
                # 7. Decode Answer & Confidence
                outputs = generated.sequences
                scores = generated.scores
                
                # Calculate confidence score
                try:
                    # Depending on transformers version
                    transition_scores = model.compute_transition_scores(
                        outputs, scores, normalize_logits=True
                    )
                    # Use geometric mean of probabilities (exp of mean log prob)
                    # We compute only for the FIRST answer in batch (since batch=1 usually)
                    log_probs = transition_scores[0]
                    confidence = np.exp(torch.mean(log_probs).cpu().numpy())
                except:
                    # Fallback if compute_transition_scores fails or model doesn't support
                    confidence = 0.0
                
                new_tokens = outputs[:, input_ids.shape[1]:]
                text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
                gt_text = correct_answers[step_idx+1]

                if step_idx != 1:
                    if gt_text == text:
                        accuracy_cnt += 1
                    
                    # Accumulate for mAP
                    y_pred_list.append(text)
                    y_true_list.append(gt_text)
                    ypred_score_list.append(confidence)
                    class_set.add(gt_text)
                    class_set.add(text)

                print(f"Generated answer {step_idx+1}:", text, f"(Conf: {confidence:.4f})")
                print(f"Correct answer {step_idx+1}:", correct_answers[step_idx+1])
                conv.messages[-1][-1] = text


            avg_time += time.time() - start_time
            print("Time taken for scene:", time.time() - start_time)
        else:
            # Fallback for datasets without 'questions' list
            print(f"\nProcessing scene {scene} (Single Turn / Legacy).")
            # ... existing logic ...
            valid_input_ids = inputs['input_ids'][0].clone()
            valid_input_ids = valid_input_ids[valid_input_ids >= 0]
            
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    # temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                )
                new_tokens = generated[:, inputs["input_ids"].shape[1]:]
                text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
                print("Generated answer:", text)
    avg_time /= num_val
    print(f"\nAverage time per scene: {avg_time:.2f} seconds")
    print(f"\nOverall accuracy: {accuracy_cnt}/{len(y_true_list)} = {accuracy_cnt/len(y_true_list):.4f}")
    
    if len(y_true_list) > 0:
        map_score = calculate_map(y_true_list, y_pred_list, ypred_score_list, list(class_set))
        print(f"Overall mAP: {map_score:.4f}")
    else:
        print("No valid samples for mAP calculation.")
    


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
