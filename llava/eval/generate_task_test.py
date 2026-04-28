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

    obj = '229_screwdriver'
    obj_index = test_dataset.get_obj_task(obj)
    task = 'Grasp the screwdriver to hammer the nail into the wood.'
    instance = test_dataset.get_data_info(index=obj_index, task=task)
    selected_data = [instance]

    # num_val = 42
    # len_data = len(test_dataset)
    # data_list = list(range(len_data))
    # selected_data = random.sample(data_list, num_val)

    grasp_dict = {}
    accuracy_cnt = 0
    avg_time = 0.0
    for i in tqdm(selected_data):
        gt_grasps = instance["grasps"]
        pc_path = instance["pc_path"]
        scene = instance["scene"]
        gs_labels = instance["gs_labels"]
        correct_answers = instance.pop("correct_answer")
        inputs = precess_data(tokenizer, instance)
        inputs["attention_mask"] = inputs["input_ids"].ne(tokenizer.pad_token_id)

        # outputs = model(**inputs)


        # generation 不需要 labels，去掉避免落入训练分支
        inputs.pop("labels", None)
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
                    )
                
                # 7. Decode Answer
                new_tokens = generated[:, input_ids.shape[1]:]
                text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
                if step_idx != 1 and correct_answers[step_idx+1] == text:
                    accuracy_cnt += 1
                print(f"Generated answer {step_idx+1}:", text)
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

    print(f"\nTime per scene: {avg_time:.2f} seconds")
        # with torch.inference_mode():
        #     generated = model.generate(
        #         **inputs,
        #         max_new_tokens=args.max_new_tokens,
        #         temperature=args.temperature,
        #         top_p=args.top_p,
        #         num_beams=args.num_beams,
        #     )
        #     new_tokens = generated[:, inputs["input_ids"].shape[1]:]
        #     text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        #     print("Generated answer:", text)

            
            
            # generation_inputs = {
            #     "input_ids": inputs["input_ids"],
            #     "images": inputs.get("images"),
            #     "depths": inputs.get("depths"),
            #     "poses": inputs.get("poses"),
            #     "intrinsics": inputs.get("intrinsics"),
            # }
            # # 去掉 labels/attention_mask，不需要
            # generation_inputs = {k: v for k, v in generation_inputs.items() if v is not None}

            # model.config.pad_token_id = tokenizer.pad_token_id
            # model.config.eos_token_id = tokenizer.eos_token_id

            # with torch.inference_mode():
            #     generated = model.generate(
            #         **inputs,
            #         max_new_tokens=args.max_new_tokens,
            #         temperature=args.temperature,
            #         top_p=args.top_p,
            #         num_beams=args.num_beams,
            #     )

            # new_tokens = generated[:, inputs["input_ids"].shape[1]:]
            # text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
            # print("Generated answer:", text)

            # grasp_outs, outputs = model(**inputs)

            # grasps = model.predict_grasps(grasp_outs)
            # pred_grasps, scores, labels = grasps[0]

        # grasp_dict[scene]=dict(gen_grasps=pred_grasps, gt_grasps=gt_grasps, scores=scores, pc_path=pc_path, scene=scene)

    # with open(os.path.join(args.output_dir, "scenegrasp_gen_data.pkl"), 'wb') as f:
    #     pickle.dump(grasp_dict, f)
    
    


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
