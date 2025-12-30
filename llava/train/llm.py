import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
import imageio
from skimage.transform import rotate, resize
import open3d as o3d
import trimesh.transformations as tra
from sklearn.decomposition import PCA
import os
from pathlib import Path
from openai import OpenAI
import base64
import os
import time
import re
import pickle

# set your DASHSCOPE_API_KEY here
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

# move_attributes = {
#     "Surface Slide": {"description": "", "examples": []},
#     "Press and Impact": {"description": "", "examples": []},
#     "Transport": {"description": "", "examples": []},
#     "Hang": {"description": "", "examples": []},
#     "Twist": {"description": "", "examples": []},
#     "Pour": {"description": "", "examples": []},
#     "Scoop": {"description": "", "examples": []},
#     "Cut": {"description": "", "examples": []},
#     "Stir": {"description": "", "examples": []},
#     "Insert": {"description": "", "examples": []},
#     "Shake": {"description": "", "examples": []},
#     "Squeeze": {"description": "", "examples": []},
#     "Flip": {"description": "", "examples": []}
# }

DASHSCOPE_API_KEY = "sk-f921c5fc67e549768f697c60410d11fc"
PREPOSITION_VECTORS = [
    ("front",      np.array([-1, 0,  0])),
    ("front-left",    np.array([-1, 1,  0])),
    ("front-right",   np.array([-1, -1,  0])),

    ("front-above",    np.array([-1, 0,  1])),
    ("front-above-left",  np.array([-1, 1,  1])),
    ("front-above-right",  np.array([-1, -1,  1])),

    ("front-below",    np.array([-1, 0, -1])),
    ("front-below-left",  np.array([-1, 1, -1])),
    ("front-below-right",  np.array([-1, -1, -1])),

    ("left",      np.array([0,  1,  0])),
    ("right",      np.array([0, -1,  0])),

    ("above",      np.array([0,  0,  1])),
    ("above-left",    np.array([0, 1,  1])),
    ("above-right",    np.array([0, -1,  1])),

    ("below",      np.array([0,  0, -1])),
    ("below-left",    np.array([0, 1, -1])),
    ("below-right",    np.array([0, -1, -1])),

    ("back",      np.array([1, 0,  0])),
    ("back-left",    np.array([1, 1,  0])),
    ("back-right",    np.array([1, -1,  0])),

    ("back-above",    np.array([1,  0,  1])),
    ("back-above-left",  np.array([1, 1,  1])),
    ("back-above-right",  np.array([1, -1,  1])),

    ("back-below",    np.array([1, 0, -1])),
    ("back-below-left",  np.array([1, 1, -1])),
    ("back-below-right",  np.array([1, -1, -1])),
]

# client = OpenAI(
#     api_key=DASHSCOPE_API_KEY,
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )

def assign_prepositions(direction_vector: np.ndarray):
    """
    direction_vector: shape = (N, 3)，每一行是一个方向向量。
    返回：
        best_preps:  长度 N 的列表/数组，对应每个向量的空间方位（字符串）
        best_sims:   长度 N 的数组，对应最大余弦相似度
        best_indices:长度 N 的数组，对应匹配到的方位索引（0~25）
    """
    assert direction_vector.ndim == 2 and direction_vector.shape[1] == 3

    # 1) 归一化输入向量
    vecs = direction_vector.astype(float)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    zero_mask = norms.squeeze() == 0
    norms[norms == 0] = 1.0  # 避免除 0
    vecs_norm = vecs / norms

    # 2) 构造 26 个单位方向向量
    canon_vecs = np.stack([v for _, v in PREPOSITION_VECTORS], axis=0).astype(float)
    canon_norms = np.linalg.norm(canon_vecs, axis=1, keepdims=True)
    canon_vecs_norm = canon_vecs / canon_norms  # shape = (26, 3)

    # 3) 计算余弦相似度：N×26
    cos_sim = vecs_norm @ canon_vecs_norm.T  # (N, 26)

    # 4) 取每行最大值
    best_indices = np.argmax(cos_sim, axis=1)
    best_sims = cos_sim[np.arange(vecs.shape[0]), best_indices]
    best_preps = [PREPOSITION_VECTORS[i][0] for i in best_indices]

    # 对于零向量（长度为 0 的），没有方向：设为 None / NaN
    for i, is_zero in enumerate(zero_mask):
        if is_zero:
            best_preps[i] = None
            best_sims[i] = np.nan

    return np.array(best_preps, dtype=object), best_sims, best_indices

# 读取本地图片并转换为base64
# def encode_image_to_base64(image_path):
#     with open(image_path, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
#         return f"data:image/jpeg;base64,{encoded_string}"

def encode_image_to_base64(image_path):
    suffix = image_path.suffix.lower()
    if suffix == ".png":
        mime = "image/png"
    elif suffix in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    else:
        raise ValueError(f"Unsupported image type: {suffix}")

    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def get_object_part(pth, object, file):
    # 确保 pth 是 Path 对象
    pth = Path(pth)
    image_path_main = encode_image_to_base64(pth / "0_color.png")
    image_path_left = encode_image_to_base64(pth / "1_color.png")
    # image_path_right = encode_image_to_base64(pth / "2_color.png")

    # prompt_text = f"这是同一个物体三个视角的图像，图像中的物体是一个{object}。请直接输出该物体由哪几个部件组成,输出格式用逗号分隔的部件名称列表，例如“部件1, 部件2, 部件3”。"
    prompt_text = f"These are two images of the same object from different viewpoints. The object in the images is a {object}. Please directly output which parts the object consists of, using a comma-separated list of part names, e.g., 'Part 1,Part 2,Part 3'."
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_path_main}},
                {"type": "image_url", "image_url": {"url": image_path_left}},
                # {"type": "image_url", "image_url": {"url": image_path_right}},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    completion = client.chat.completions.create(
        model="qwen3-vl-235b-a22b-instruct",
        messages=messages,
        extra_body={"enable_thinking": True},
    )
    response = completion.choices[0].message.content
    print(response)
    # out_path = f"/home/robot/WCL/GraspCoT/task_data/{file}/object_part.txt"
    out_path = f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{file}/object_part.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(response)

def read_object_part(pth):
    obj_part = []
    with open(pth, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                obj_part = line
    # print("物体部件列表:", obj_part)
    return obj_part

def read_part_attributes(pth):
    part_attributes = []
    with open(pth, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                part_attributes.append(line)
    # print("物体部件属性列表:", part_attributes)
    return part_attributes

def read_move_attributes(pth):
    attributes = pth + "/Movement_Attributes.txt"
    move_attributes = []
    with open(attributes, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()  # 去掉每行前后的空白字符
            if line:  # 跳过空行
                move_attributes.append(line)
    
    # examples = pth + "/Movement_Examples.txt"
    # with open(examples, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         line = line.strip()  # 去掉每行前后的空白字符
    #         if line:  # 跳过空行
    #             key = line.split(':', 1)[0].strip()
    #             move_attributes[key]["examples"].append(line.split(':', 1)[1].strip())

    return move_attributes

def read_task(pth):
    with open(pth, 'r', encoding='utf-8') as f:
        task = f.read().strip()
    return task         

def read_grasp_part(pth):
    grasp_part = []
    with open(pth, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                grasp_part.append(line)
    return grasp_part

def get_grasp_prompt(root, subdirs_sorted, grasp_labels, obj):
    results = []
    for d in subdirs_sorted:
        grasp_idx = int(d.name)
        grasp_parts = grasp_labels[grasp_idx]['part_label']
        grasp_preps = grasp_labels[grasp_idx]['prep_label']

        image_path_main = encode_image_to_base64(d / "grasp_main.png")
        image_path_left = encode_image_to_base64(d / "grasp_left.png")
        image_path_right = encode_image_to_base64(d / "grasp_right.png")
        # prompt_text = f"图中的物体是一个马克杯，四根绿色圆柱体的代表是机器人平行二指夹爪，夹抓在{grasp_parts}的{grasp_preps}位置，请你用抓取部位+接近方向+姿态来描述这个抓取。例如：将夹爪从上方接近，夹持锤头中部，夹爪的张开方向与锤柄轴线垂直。"
        # prompt_text = "The object in the images is a mug, and the four green cylinders represent the grasp of a robot’s parallel two-finger gripper. Please output which part of the mug the gripper is grasping, for example, the mug body or the handle."
        # prompt_text = f"图像中的物体是一个马克杯，四个绿色的圆柱体代表机器人平行的两指夹爪的抓取。夹爪在{grasp_parts}的{grasp_preps}的方向，请你描述这个抓取。举例：从右上方接近抓取手柄，夹爪方向垂直于手柄的环状结构，使钳口能够包裹住手柄顶部的弯曲处。"
        prompt_text = f"The object in the images is a {obj}, and the four green cylinders represent the fingers of a parallel-jaw robotic gripper.The gripper is grasping the {grasp_parts} from the {grasp_preps} direction.Please describe the grasp using the format “Grasp the [Part] from the [Direction], with the gripper [Relative Orientation], [Geometric Effect]. For example, secure the hammer by its head, gripping near the center of the head from above, with the gripper jaws oriented perpendicular to the handle axis."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_path_main}},
                    # {"type": "image_url", "image_url": {"url": image_path_left}},
                    # {"type": "image_url", "image_url": {"url": image_path_right}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        completion1 = client.chat.completions.create(
            model="qwen3-vl-235b-a22b-instruct",
            messages=messages,
            extra_body={"enable_thinking": True},
        )
        response = completion1.choices[0].message.content
        print(grasp_idx)
        print(response)
        line = f"{grasp_idx};{response}"
        results.append(line)

    # 写入到 txt 文件
    out_path = root / "temp.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for line in results:
            f.write(line + "\n")

    print(f"saved as: {out_path}")

def get_object_property(pth, obj, file, part):
    pth = Path(pth)
    img = encode_image_to_base64(pth / "0_color.png")
    prompt_text = f"图中的物体是{obj},它由{part}组成。物体的属性描述分为硬度：坚硬、柔软等，边缘形状：锋利、光滑等，几何结构：圆柱形、弧形等，请你按这个格式描述图中各个部件的属性，硬度：【】，边缘形状【】，几何结构：【】。"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img}},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    completion = client.chat.completions.create(
        model="qwen3-vl-235b-a22b-instruct",
        messages=messages,
        extra_body={"enable_thinking": True},
    )
    response = completion.choices[0].message.content
    print(response)

def get_object_property_openai(pth, obj, part):
    openai_client = OpenAI(
        base_url="https://api.zhizengzeng.com/v1",
        api_key="sk-zk2f7b0762a500c2b32bd4c24657b238d29693cf94767c23"
    )
    pth = Path(pth)
    img = encode_image_to_base64(pth / "0_color.png")
    # prompt_text = f"图中的物体是{obj},它由{part}组成。物体的属性描述分为硬度：坚硬、柔软等，边缘形状：锋利、光滑等，几何结构：圆柱形、弧形等，请你按这个格式描述图中各个部件的属性，硬度：【】，边缘形状【】，几何结构：【】。"
    # prompt_text = f"图中的物体是{obj},它由{part}组成。物体的属性描述描述由硬度，边缘形状和几何结构组成，请你按这个格式描述图中各个部件的属性，硬度：【】，边缘形状【】，几何结构：【】。"
    # prompt_text = f"""
    # The object in the image is a {obj}, which consists of the following parts: {part}.
    # The attributes of the object are described from three aspects:
    # 1. Hardness (e.g., rigid, soft, brittle, elastic).
    # 2. Edge Shape (e.g., smooth, sharp, rough, serrated).
    # 3. Geometric Structure (e.g., short cylindrical, elongated and curved, hemispherical, conical).

    # Please strictly follow the format below when producing the output:
    # Part:[], Hardness:[], Edge Shape:[], Geometric Structure:[].
    # """
    prompt_text = f"""
    The object in the image is a {obj}, which consists of the following parts: {part}.
    Please analyze the visual features strictly based on the provided image.
    The attributes of the object are described from three aspects:
    1. Hardness (e.g., rigid, soft, brittle, elastic).
    2. Edge Shape (e.g., smooth, sharp, rough, serrated).
    3. Geometric Topology Structure (e.g., short cylindrical, elongated and straight, curved tubular, hemispherical, perforated).
    *Note:The examples below are ONLY style references, NOT an exhaustive list. You MUST freely create new phrases if needed.*

    Please strictly follow the format below when producing the output:
    Part:[], Hardness:[], Edge Shape:[], Geometric Structure:[].
    """
    results = openai_client.chat.completions.create(
        model="gpt-5",
        messages=[
            {
                "role": "system",
                "content": "You are a precise computer vision assistant specializing in physical property extraction from images. You provide detailed, accurate descriptions based solely on the visual input."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": img}
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            }
        ],
    )
    result = results.choices[0].message.content
    print(result)
    # 写入到 txt 文件
    out_path = pth / "object_property.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result + "\n")

    print(f"saved as: {out_path}")
    return result

def get_grasp_motion_openai(pth, obj, part, part_attr, description, grasp_part):
    openai_client = OpenAI(
        base_url="https://api.zhizengzeng.com/v1",
        api_key="sk-zk2f7b0762a500c2b32bd4c24657b238d29693cf94767c23"
    )
    pth = Path(pth)
    parts_list = [p.strip() for p in part.split(',')]
    parts_list = [p for p in parts_list if 'Pin' not in p]
    part = ', '.join(parts_list)

    # Filter part_attr to remove elements where Part contains 'Pin'
    filtered_part_attr = []
    for attr in part_attr:
        match = re.search(r'Part:\[(.*?)\]', attr)
        if match:
            if 'Pin' not in match.group(1):
                filtered_part_attr.append(attr)
        else:
            filtered_part_attr.append(attr)
    part_attr = filtered_part_attr
    out = []
    for index in range(9, 10):
        grasp_part_name = grasp_part[index]
        # img_main = encode_image_to_base64(pth / f"visual_grasps/{index}/grasp_main.png")
        img_main = encode_image_to_base64(pth / f"0_color.png")
        system_prompt = """
        You are a precise computer vision assistant. Output appropriate motion primitives based on visual input and component attributes.
        """
        prompt_text = f"""
        There are 13 categories of motion primitives, defined as: {description}
        The object is {obj}, which consists of the following parts: {part}. The attributes of each part are: {part_attr}.
        Based on the image, please analyze which motion primitives are suitable for each part from the following perspectives.
        Output the feasible motion primitives selected from the provided definitions and their corresponding components. 
        Output Format: Comma-separated strings (e.g., Transport-Handle, Surface Slide-Neck).
        """

        # prompt_text = f"""
        # There are 13 categories of motion primitives, defined as: {description}
        # The object is {obj}, which consists of the following parts: {part}. The attributes of each part are: {part_attr}.
        # The robot is grasping the {grasp_part_name} part. The red arrow in the image shows the grasping posture, and the tip of the arrow represents the grasping position.
        # Please consider from the following perspectives what motion primitives are suitable for such grasping.
        # Perspective: Grasp the {grasp_part_name} part and use the attributes of other components to perform operations on motion primitives. 
        # Note: Analyze if the grasp point in the diagram is at the critical point of the two components. If so, assess whether the robot's physical volume causes collisions that prevent using the adjacent component's motion primitives.
        
        # Output the feasible motion primitives selected from the provided definitions and their corresponding components. Format: Comma-separated strings (e.g., Transport-Handle, Surface Slide-Neck).
        # """

        results = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": img_main}
                        },
                        {
                            "type": "text",
                            "text": prompt_text
                        }
                    ]
                }
            ],
        )

        result = results.choices[0].message.content
        out.append(result)
    print(out)
    # 写入到 txt 文件
    out_path = pth / "grasps_motion.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")

    print(f"saved as: {out_path}")
    return out

def get_grasp_part_openai(pth, obj, part):
    openai_client = OpenAI(
        base_url="https://api.zhizengzeng.com/v1",
        api_key="sk-zk2f7b0762a500c2b32bd4c24657b238d29693cf94767c23"
    )
    pth = Path(pth)
    out = []
    # Filter out parts containing 'Pin'
    parts_list = [p.strip() for p in part.split(',')]
    parts_list = [p for p in parts_list if 'Pin' not in p]
    part = ', '.join(parts_list)

    for index in range(25):
        img_main = encode_image_to_base64(pth / f"{index}/grasp_main.png")
        # img_right = encode_image_to_base64(pth / f"{index}/grasp_right.png")

        system_prompt = """
        You are a robotic vision assistant specializing in semantic part identification.
        Your specific task is to identify which part of the object is marked based on the visual markers.
        CRITICAL OUTPUT RULES:
        1.  Output ONLY the part name.
        2.  Do NOT output sentences, punctuation, or explanations.
        """

        prompt_text = f"""
        The object in the image is a {obj}, which consists of the following parts: {part}.
        Please identify which part in {part} the red arrow is pointing to in the image. If it is not in the list, please output None.
        Please analyze the visual features strictly based on the provided image.
        Please directly output the part name, for example: Powl or None.
        """
        
        results = openai_client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": img_main}
                        },
                        # {
                        #     "type": "image_url",
                        #     "image_url": {"url": img_right}
                        # },
                        {
                            "type": "text",
                            "text": prompt_text
                        }
                    ]
                }
            ],
        )
        result = results.choices[0].message.content
        out.append(result)
    print(out)
    # 写入到 txt 文件
    out_path = pth / "grasps_part.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")

    print(f"saved as: {out_path}")
    return out

def get_obj_task(pth, obj, parts, part_attr, description, task):
    pth = Path(pth)
    part = [item.strip() for item in parts.split(',')]
    # prompt_text = f"""
    # 任务由动词和名词组成,
    # 物体是{obj},它由{parts}组成,各个部件属性如下:{part_attr},当部件{part[1]}作为主要交互部件时,请你生成十个具有创新性的任务,即任务与物体类别无关，但与部件的属性相关,
    # 任务里不能出现部件名,但任务却需要该部件,输出的格式应该为:动词+名词，例如:清扫桌面上的碎屑、悬挂马克杯等
    # """
    prompt_text = f"""
    任务为: {task},运动属性有13种类别,定义为:{description}.
    物体是{obj},它由{parts}组成,各个部件属性如下:{part_attr}.
    请你为每个任务输出以下内容
    1、任务对应的交互属性是[] 
    2、根据任务语义和交互属性[]推理出部件需要具备什么属性
    3、该物体的部件中,哪个部件与之匹配,也可以输出None,代表没有合适的部件    
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    completion = client.chat.completions.create(
        model="qwen3-vl-235b-a22b-instruct",
        messages=messages,
        extra_body={"enable_thinking": True},
    )
    response = completion.choices[0].message.content
    print(response)

def get_obj_task_openai(pth, obj, parts, part_attr):
    openai_client = OpenAI(
        base_url="https://api.zhizengzeng.com/v1",
        api_key="sk-zk2f7b0762a500c2b32bd4c24657b238d29693cf94767c23"
    )
    pth = Path(pth)
    part = [item.strip() for item in parts.split(',')]
    # prompt_text = f"""
    # 任务为: {task},运动属性有13种类别,定义为:{description}.
    # 物体是{obj},它由{parts}组成,各个部件属性如下:{part_attr}.
    # 请你为每个任务输出以下内容
    # 1、任务对应的交互属性是[] 
    # 2、根据任务语义和交互属性[]推理出部件需要具备什么属性
    # 3、该物体的部件中,哪个部件与之匹配,也可以输出None,代表没有合适的部件
    # """
    prompt_text = f"""
    物体是{obj},它由{parts}组成,各个部件属性如下:{part_attr}.
    根据该物体的属性,请你生成4个该类物体相关或与该物体的部件属性相关的任务,输出的格式应该为:grasp the {obj} to + 动词 + 名词或grasp the {obj} to + 动词, 例如:grasp the {obj} to sweep the debris on the table、grasp the {obj} to hang.
    """
    results = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            }
        ],
    )
    result = results.choices[0].message.content
    print(result)
    out_path = pth / "task_common.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result + "\n")

    print(f"saved as: {out_path}")
    return result

def get_task_infer_openai(pth, obj, parts, part_attr, description, task):
    openai_client = OpenAI(
    base_url="https://api.zhizengzeng.com/v1",
    api_key="sk-zk2f7b0762a500c2b32bd4c24657b238d29693cf94767c23"
    )
    pth = Path(pth)
    part = [item.strip() for item in parts.split(',')]
    # prompt_text = f"""
    # 任务为: {task},运动属性有13种类别,定义为:{description}.
    # 物体是{obj},它由{parts}组成,各个部件属性如下:{part_attr}.
    # 请你为每个任务输出以下内容
    # 1、任务对应的交互属性是[] 
    # 2、根据任务语义和交互属性[]推理出部件需要具备什么属性
    # 3、该物体的部件中,哪个部件与之匹配,也可以输出None,代表没有合适的部件
    # """
    prompt_text = f"""
    The task is: {task}. There are 13 categories of motion primitives, defined as: {description}.
    The object is {obj}, which consists of the following parts: {parts}. The attributes of each part are: {part_attr}.

    For this task, answer three questions.
    Question 1: Which type of motion primitive does the task belong to?
    Question 2: Combining the task semantics and the motion primitives it belongs to, infer what attributes the component needs.
    Question 3: Does the object have any components that meet the task requirements? If not, answer "None"
    please output the results strictly in the following format.

    Example Format (with matching part):
    Task 1: Grasp the squeezer to squeeze the juice from fruits
    1.Squeeze.
    2.The part should provide opposing forces and should be rigid to deform the fruit, allowing liquid to be extracted. The part should have a shape suitable for holding or accommodating the fruit and creating pressure.
    3.Bowl.

    Example Format (no suitable part):
    Task 2: Grasp the squeezer to hammer a nail into wood
    1.Press and Impact.
    2.The part should be heavy and flat on one end to deliver strong normal force upon contact. The geometry should allow a focused impact onto the nail head.
    3.None.

    Now apply this format to the current input:
    Task: {task}
    (Continue with steps 1-3 as shown above)
    """
    results = openai_client.chat.completions.create(
        model="gpt-5",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            }
        ],
    )
    result = results.choices[0].message.content
    print(result)
    out_path = pth / "task_infer.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result + "\n")

    print(f"saved as: {out_path}")
    return result

def parse_txt_to_records(txt_path: str):
    text = Path(txt_path).read_text(encoding="utf-8").strip()

    # 按 "Task N:" 分段（保留分隔符）
    blocks = re.split(r'(?m)^\s*(Task\s+\d+\s*:\s*)', text)

    records = []

    for i in range(1, len(blocks), 2):
        header = blocks[i].strip()       # e.g., "Task 1:"
        content = blocks[i + 1].strip()
        block = (header + " " + content).strip()

        # 1) Task sentence: Task 1: <...>
        m_task = re.search(r'(?is)Task\s+\d+\s*:\s*(.+?)\s*\n', block)
        if not m_task:
            continue
        task = m_task.group(1).strip()

        # 2) Primitive: 把末尾句号也捕获进来
        m_prim = re.search(r'(?im)^\s*1\s*[.)]\s*(.+?)([.。])?\s*$', block)
        if not m_prim:
            continue
        primitive = (m_prim.group(1) + (m_prim.group(2) or "")).strip()

        # 3) Requirement: 从 "2." 这一段开始直到遇到 "3."
        m_req = re.search(r'(?is)\n\s*2\s*[.)]\s*(.+?)\n\s*3\s*[.)]\s*', block)
        if not m_req:
            continue
        requirement = re.sub(r'\s+', ' ', m_req.group(1)).strip()
        # 只去掉末尾空白，不删句号
        requirement = re.sub(r'\s+$', '', requirement)

        # 4) Part: 同样把末尾句号捕获进来
        m_part = re.search(r'(?im)^\s*3\s*[.)]\s*(.+?)([.。])?\s*$', block)
        if not m_part:
            continue
        part = (m_part.group(1) + (m_part.group(2) or "")).strip()

        records.append([task, primitive, requirement, part])

    return records

def txt_to_pkl(txt_path: str, pkl_path: str):
    records = parse_txt_to_records(txt_path)
    with open(pkl_path, "wb") as f:
        pickle.dump(records, f)
    return records

if __name__ == "__main__":
    task_obj = '/home/robot/WCL/GraspGPT_public/data/taskgrasp/scans'
    folders = [f for f in os.listdir(task_obj) if os.path.isdir(os.path.join(task_obj, f))]
    folders_sorted = sorted(folders, key=lambda s: int(s[:3]))
    folders_sorted_selected = folders_sorted[6:7]  # 仅处理第一个文件夹作为示例

    movements = read_move_attributes("/media/robot/data/WCL/taskgrasp/taskgrasp_image/task")
    # task = read_task(f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{folders_sorted_selected[0]}/task_part.txt")
    
    
    for file in folders_sorted_selected:
        obj = file.split('_', 1)[1]
        # data = txt_to_pkl(f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{file}/task_infer.txt", f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{file}/prompt.pkl")
        # print(file, len(data)==4)

        obj_part = read_object_part(f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{file}/object_part.txt")
        part_attributes = read_part_attributes(f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{file}/object_property.txt")
        grasp_part = read_grasp_part(f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{file}/visual_grasps/grasps_part.txt")

        # get_obj_task_openai(pth=f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{file}", obj=obj, parts=obj_part, part_attr=part_attributes)
        # task = read_task(f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{file}/task_common.txt")
        # get_task_infer_openai(pth=f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{file}", obj=obj, parts=obj_part, part_attr=part_attributes, description=movements, task=task)
        # get_grasp_part_openai(pth=f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{file}/visual_grasps", obj=obj, part=obj_part)
        get_grasp_motion_openai(pth=f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{file}", obj=obj, part=obj_part, part_attr=part_attributes, description=movements, grasp_part=grasp_part)
        
    



    # obj_part = read_object_part(f"/home/robot/WCL/GraspCoT/task_data/{file}/object_part.txt")

    # for i, (p, s) in enumerate(zip(preps, sims)):
    #     print(f"向量 {i}: 空间方位 = {p}, 余弦相似度 = {s:.4f}")

    # root = Path(f"/home/robot/WCL/GraspCoT/task_data/{file}")
    # subdirs = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    # subdirs_sorted = sorted(subdirs, key=lambda p: int(p.name))
    # # get_dir_objpart(root, subdirs_sorted, obj, obj_part)
    # grasp_labels = read_object_dir(f"/home/robot/WCL/GraspCoT/task_data/{file}/grasp_part_spatial.txt")
    
    # get_grasp_prompt(root, subdirs_sorted, grasp_labels, obj)