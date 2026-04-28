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
    # return "\n".join(part_attributes)
    return part_attributes

def read_part(pth):
    obj_part = []
    with open(pth, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                parts = [p.strip() for p in line.split(',') if p.strip()]
                obj_part.extend(parts)
    # print("物体部件列表:", obj_part)
    return obj_part

def read_move_attributes(pth):
    attributes = pth + "/Movement_Property.txt"
    move_attributes = []
    with open(attributes, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()  # 去掉每行前后的空白字符
            if line:  # 跳过空行
                move_attributes.append(line)

    # return "\n".join(move_attributes)
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
    # return "\n".join(grasp_part)
    return grasp_part



def get_object_property_openai(pth, obj, part):
    openai_client = OpenAI(
        base_url="https://api.zhizengzeng.com/v1",
        api_key="sk-zk2f7b0762a500c2b32bd4c24657b238d29693cf94767c23"
    )
    pth = Path(pth)
    img = encode_image_to_base64(pth / "0_color.png")
    prompt_text = f"""
    The object in the image is a {obj}, which consists of the following parts: {part}.
    Please analyze the visual features strictly based on the provided image.
    The attributes of the object are described from three aspects:
    1. Material (e.g., rigid, soft, brittle, elastic, stiff, compliant).
    2. Surface (e.g., smooth, sharp, rough, rounded-edged, matte).
    3. Geometric Structure (e.g., rectangular prism-like, elongated and straight, short cylindrical, concave, flat, short cylindrical, elongated and curved, hook-shaped, conical).
    4. Topological Structure (e.g., solid, hollow, perforated, through-hole, cavity, slotted, grooved, looped, hinged, jaw-like).
    
    *Note:The examples below are ONLY style references, NOT an exhaustive list. You MUST freely create new phrases if needed.*

    Please strictly follow the format below when producing the output:
    Part:[], Material:[], Surface:[], Geometric Structure:[], Topological Structure:[].
    """
    results = openai_client.chat.completions.create(
        model="gpt-5.4",
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

    # print(f"saved as: {out_path}")
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


def get_obj_irrelevant_task_infer(pth, obj, part_attr, ins):
    openai_client = OpenAI(
        base_url="https://api.zhizengzeng.com/v1",
        api_key="sk-zk2f7b0762a500c2b32bd4c24657b238d29693cf94767c23"
    )
    pth = Path(pth)
    prompt_text = f"""
    The object is a {obj}.
    The parts of this object and their attributes are:
    {part_attr}.

    The following interaction properties are available:
    {ins}

    Generate 5 plausible part-irrelevant tasks for this object.

    Definition:
    A part-irrelevant task is a task that cannot be accomplished using this object, because none of its parts have the physical attributes required for the task.

    For each task, answer the following three questions:
    Question 1: Which type of Interaction Property does the task belong to?
    Question 2: Based on the task semantics and the Interaction Property, infer the necessary physical properties that the required component must have. Only include the aspects that are essential for the task. The physical properties may be considered from the following four aspects:
    1. Material (e.g., rigid, soft, brittle, elastic, stiff, compliant).
    2. Surface (e.g., smooth, sharp, rough, rounded-edged, matte).
    3. Geometric Structure (e.g., rectangular prism-like, elongated and straight, short cylindrical, concave, flat, short cylindrical, elongated and curved, hook-shaped, conical).
    4. Topological Structure (e.g., solid, hollow, perforated, through-hole, cavity, slotted, grooved, looped, hinged, jaw-like).
    Question 3: Does the object have any component that meets the task requirements?
    Since all generated tasks must be part-irrelevant, the answer to Question 3 must always be: None

    Requirements:
    1. Output exactly 5 tasks.
    2. Each task should include both the action and the object being operated on.
    3. Do NOT mention the tool itself, the object category, or any part of the tool in the task description.
    4. Do NOT mention how the task is performed, what part is used, or any functional explanation in the task description.
    5. Each task should be clearly incompatible with the physical attributes of all parts of this object.
    6. Each task must correspond to one Interaction Property listed above.
    7. For Question 2, only describe the physical property aspects that are necessary for accomplishing the task. Do not force all four aspects to be included.
    8. For Question 3, always output exactly: None
    9. Keep each task concise, natural, and similar in style to everyday task instructions.
    10. Prefer tasks that are ruled out by the specific missing or mismatched attributes of this object's parts, rather than generic tasks that many unrelated objects also cannot perform.

    Please output the results strictly in the following format:

    Task 1: [task]
    1. [Interaction Property]
    2. Material: rigid, hard; Geometric Structure: flat
    3. None

    Task 2: ...
    1. ...
    2. ...
    3. None

    Now generate the results.
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
    out_path = pth / "task_part_irrelevant_infer.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result + "\n")

    print(f"saved as: {out_path}")
    return result


def get_obj_part_task_openai(pth, obj, parts, part_attr, category_tasks):
    openai_client = OpenAI(
        base_url="https://api.zhizengzeng.com/v1",
        api_key="sk-zk2f7b0762a500c2b32bd4c24657b238d29693cf94767c23"
    )
    pth = Path(pth)
    img = encode_image_to_base64(pth / "0_color.png")
    prompt_text = f"""
    The object in the image is a {obj}. It consists of the following parts: {parts}. 
    The attributes of each part are: 
    {part_attr}.
    The following tasks are category-related tasks for this object:{category_tasks[0]}

    Based on the properties of different parts of this object, generate 5 plausible part-related tasks that this object could be used for.
    Also consider a hanging-related task only if the part attributes clearly indicate a structure that supports suspension, such as a through-hole, loop, open handle, ring, or hookable opening; otherwise, do not include such a task.
    
    Requirements:
    1. Output exactly 5 tasks as a numbered list from 1 to 5.
    2. The task should include both the action and the object being operated on.
    3. Do NOT mention the tool itself, the object category, or any part of the tool.
    4. Do NOT mention how the task is performed, what part is used, or any functional explanation.
    5. If the task does not involve directly acting on another object, it is allowed to mention the relevant target noun involved in the task.
    6. Keep each task concise, natural, and similar in style to everyday task instructions.
    7. The generated tasks should be as different as possible from the category-related tasks, and should reflect different relevant parts when possible.
    8. The tasks should vary in action type when possible, not just in the operated object.

    Examples of valid style:
    1. Spread butter onto bread
    2. Cut vegetables into slices
    3. Hammer a nail into the wall
    4. Scoop soup into a bowl
    5. Tighten a loose screw

    Now generate 5 tasks.
    """
    # prompt_text = f"""
    # The object in the image is a {obj}. It consists of the following parts: {parts}.

    # The following tasks have already been identified as Category-related tasks for this object category:
    # {category_tasks}

    # Generate 5 task descriptions that belong to the category of Part-related tasks only.

    # Definition:
    # - Part-related task: a task that is not included in the Category-related tasks of this object category, but may still be accomplished because one or more parts of the object have suitable physical characteristics, shapes, or structural affordances.
    # - These tasks should be non-canonical for the object category, and should rely on the possible use of a specific part rather than the object's typical category-level function.

    # Additional constraints:
    # - A Part-related task should exploit a specific part in a secondary or non-primary way, rather than simply reusing the object's main functional structure for a different content, material, or context.
    # - Do not generate tasks that merely treat the object as a generic container, holder, storage item, support, or weight, unless the task clearly depends on a distinctive part-specific affordance.
    # - Favor tasks that involve a different interaction mode from the object's canonical use, not just a different target object or a different material being handled.

    # Requirements:
    # 1. Each task must describe the intended action or goal, not the grasping action.
    # 2. If the task acts on an external target, do NOT explicitly mention the object category name, the tool name, or a direct noun phrase referring to the object itself. If the task acts directly on the object itself, explicitly mentioning the object category name is allowed when needed for a natural and clear task description.
    # 3. Each task must contain a verb. If the task acts on an external target, include the manipulated object, substance, or recipient when appropriate. If the task acts on the object itself, explicitly naming the object category is allowed and often preferred for clarity.
    # 4. The tasks should be realistic, concise, and diverse in wording.
    # 5. Do not generate tasks that are semantically equivalent to the listed Category-related tasks.
    # 6. Output exactly 5 tasks as a numbered list from 1 to 5.
    # 7. Do NOT explicitly mention the names of the object's parts in the task sentence.

    # Illustrative examples of Part-related tasks:
    # - For an object whose canonical use is hammering, a Part-related task may involve using a non-primary elongated part to push or sweep small items aside.
    # - For an object whose canonical use is drinking, a Part-related task may involve suspending the object from a support by relying on a side feature.
    # - For an object whose canonical use is scooping, a Part-related task may involve using a broad curved surface to spread a soft substance across another surface.

    # Important:
    # - These examples are only for explaining the concept of Part-related tasks.

    # A good Part-related task should reflect a non-canonical interaction mode enabled by a specific part, rather than simply using the object as a generic container or describing how a person holds or carries it.

    # Output format examples only:
    # 1. Verb + object
    # 2. Verb + object + preposition + target
    # 3. Verb + object + into + resulting form
    # 4. Verb + substance + from + container
    # 5. Verb + the object category name + preposition + target when the task acts on the object itself
    # """
    results = openai_client.chat.completions.create(
        model="gpt-5",
        messages=[
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
    out_path = pth / "task_part_related_new.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result + "\n")

    print(f"saved as: {out_path}")
    return result


def get_obj_category_task_openai(pth, obj):
    openai_client = OpenAI(
        base_url="https://api.zhizengzeng.com/v1",
        api_key="sk-zk2f7b0762a500c2b32bd4c24657b238d29693cf94767c23"
    )
    pth = Path(pth)
    # part = [item.strip() for item in parts.split(',')]
    # prompt_text = f"""
    # 物体是{obj},它由{parts}组成,各个部件属性如下:{part_attr}.
    # 根据该物体的属性,请你生成4个该类物体相关或与该物体的部件属性相关的任务,输出的格式应该为:grasp the {obj} to + 动词 + 名词或grasp the {obj} to + 动词, 例如:grasp the {obj} to sweep the debris on the table、grasp the {obj} to hang.
    # """
    img = encode_image_to_base64(pth / "0_color.png")
    prompt_text = f"""
    The object in the image is a {obj}.
    Generate 5 task descriptions that belong to the category of Category-related tasks only.
    Definition:
    - Category-related task: a task that reflects the core, canonical, and typical function of this object category.
    Task validity rules:
    - A valid Category-related task must be a task that this object directly performs during normal use.
    - The task must describe the object's primary function, not a downstream use of the result.
    - Do not generate tasks that are only enabled by a specific part attribute or special part structure.
    - Do not generate post-processing, transfer, serving, filtering, or follow-up tasks after the main function is completed.

    Requirements:
    1. Each task must describe the intended action or goal, not the grasping action.
    2. Do NOT mention the tool or object itself in the task sentence.
    3. Each task must contain a verb and a manipulated target object, substance, or recipient when appropriate.
    4. All 5 tasks must stay within the same canonical function of this object category.
    5. If the object category has a narrow canonical use, vary the target object or wording, but do not expand to secondary functions.
    6. Output exactly 5 tasks as a numbered list from 1 to 5.

    Examples of valid style:
    1. Spread butter onto bread
    2. Cut vegetables into slices
    3. Hammer a nail into the wall
    4. Scoop soup into a bowl
    5. Tighten a loose screw
    """
    results = openai_client.chat.completions.create(
        model="gpt-5.4",
        messages=[
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
    out_path = pth / "task_category_related.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result + "\n")

    print(f"saved as: {out_path}")
    return result

def get_task_infer_openai(pth, obj, part_attr, description, task):
    openai_client = OpenAI(
    base_url="https://api.zhizengzeng.com/v1",
    api_key="sk-zk2f7b0762a500c2b32bd4c24657b238d29693cf94767c23"
    )
    pth = Path(pth)
    # part = [item.strip() for item in parts.split(',')]
    # prompt_text = f"""
    # 任务为: {task},运动属性有13种类别,定义为:{description}.
    # 物体是{obj},它由{parts}组成,各个部件属性如下:{part_attr}.
    # 请你为每个任务输出以下内容
    # 1、任务对应的交互属性是[] 
    # 2、根据任务语义和交互属性[]推理出部件需要具备什么属性
    # 3、该物体的部件中,哪个部件与之匹配,也可以输出None,代表没有合适的部件
    # """
    prompt_text = f"""
        The tasks are: 
        {task}. 
        There are 15 categories of Interaction Property, defined as: 
        {description}.

        The tool object is {obj}. The parts of the tool object and their attributes are:
        {part_attr}.

        For each task, answer three questions separately.
        Important rule for Question 1:
        Do not infer hidden downstream goals, unstated results, or extra objects that are not explicitly mentioned in the task.
        If a task may imply multiple possible effects, choose the one that most directly applies to the explicitly mentioned manipulated object in the task.
        
        Question 1: Which type of Interaction Property does the task belong to?
        Question 2: Based on the task semantics and the Interaction Property, infer the necessary physical properties. The physical properties may be considered from the following four aspects:
        1. Material (e.g., rigid, soft, brittle, elastic, stiff, compliant).
        2. Surface (e.g., smooth, sharp, rough, rounded-edged, matte).
        3. Geometric Structure (e.g., rectangular prism-like, elongated and straight, short cylindrical, concave, flat, short cylindrical, elongated and curved, hook-shaped, conical).
        4. Topological Structure (e.g., solid, hollow, perforated, through-hole, cavity, slotted, grooved, looped, hinged, jaw-like).
        For Question 2, output only the necessary aspects, which may be one, several, or all four.
        Question 3: Does the tool object have any components that satisfy the physical property requirements in Question 2? If not, answer "None". If yes, output the functional part and the grasp part. Functional part: the part that interacts with the environment. Grasp part: the part to be grasped. The functional part and the grasp part must be different parts. Only when the tool object has only one part may the functional part and the grasp part be the same.
        please output the results strictly in the following format.

        Example Format (with matching part):
        Task 1: scoop soup into a bowl
        1.Scoop.
        2.Material: rigid; Geometric Structure: concave; Topological Structure: cavity, non-perforated, continuous.
        3.Functional part: Bowl; Grasp part: Handle.

        Example Format (no suitable part):
        Task 2: hammer a nail into wood
        1.Impact.
        2.Material: rigid; Geometric Structure: broad and heavy striking surface.
        3.None.

        Now apply this format to every task above.
        (Continue with steps 1-3 as shown above)
        """
    # prompt_text = f"""
    # The task is: {task}. There are 14 categories of Interaction Property, defined as: 
    # {description}.

    # The object is {obj}. The parts of this object and their attributes are:
    # {part_attr}.

    # For this task, answer three questions.
    # Question 1: Which type of Interaction Property does the task belong to?
    # Question 2: Based on the task semantics and the Interaction Property, infer the necessary physical properties that the required component must have. Only include the aspects that are essential for the task. The physical properties may be considered from the following four aspects:
    # 1. Material (e.g., rigid, soft, brittle, elastic, stiff, compliant).
    # 2. Surface (e.g., smooth, sharp, rough, rounded-edged, matte).
    # 3. Geometric Structure (e.g., rectangular prism-like, elongated and straight, short cylindrical, concave, flat, short cylindrical, elongated and curved, hook-shaped, conical).
    # 4. Topological Structure (e.g., solid, hollow, perforated, through-hole, cavity, slotted, grooved, looped, hinged, jaw-like).
    # Question 3: Does the object have any components that meet the task requirements? If not, answer "None"
    # please output the results strictly in the following format.

    # Example Format (with matching part):
    # Task 1: squeeze the juice from fruits
    # 1.Compression.
    # 2.Material: rigid
    # 3.Bowl.

    # Example Format (no suitable part):
    # Task 2: hammer a nail into wood
    # 1.Impact.
    # 2.Material: rigid
    # 3.None.

    # Now apply this format to the current input:
    # Task: {task}
    # (Continue with steps 1-3 as shown above)
    # """
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
    out_path = pth / "task_part_related_infer.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result + "\n")

    # print(f"saved as: {out_path}")
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
    task_obj = '/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans'
    folders = [f for f in os.listdir(task_obj) if os.path.isdir(os.path.join(task_obj, f))]
    folders_sorted = sorted(folders, key=lambda s: int(s[:3]))
    folders_sorted_selected = folders_sorted[:]

    
    # for file in folders_sorted_selected:
    #     print(f"Processing folder: {file}")
    #     obj = file.split('_', 1)[1]
    #     obj_part = read_object_part(f"{pth}/{file}/object_part.txt")
    #     part_attributes = read_part_attributes(f"{pth}/{file}/object_property.txt")
    #     category_task = read_grasp_part(f"{pth}/{file}/task_part_related_new.txt")
    #     get_task_infer_openai(pth=f"{pth}/{file}", obj=obj, part_attr=part_attributes, description=interaction_property, task=category_task)
    #     # get_obj_irrelevant_task_infer(pth=f"{pth}/{file}", obj=obj, part_attr=part_attributes, ins=interaction_property)
    #     # get_obj_part_task_openai(pth=f"{pth}/{file}", obj=obj, parts=obj_part, part_attr=part_attributes, category_tasks=category_task)
    #     time.sleep(2)

    
    # for file in folders_sorted_selected:
    #     obj = file.split('_', 1)[1]
    #     # data = txt_to_pkl(f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{file}/task_infer.txt", f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{file}/prompt.pkl")
    #     # print(file, len(data)==4)

    #     obj_part = read_object_part(f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{file}/object_part.txt")
    #     part_attributes = read_part_attributes(f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{file}/object_property.txt")
    #     grasp_part = read_grasp_part(f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{file}/visual_grasps/grasps_part.txt")

    #     # get_obj_task_openai(pth=f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{file}", obj=obj, parts=obj_part, part_attr=part_attributes)
    #     # task = read_task(f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{file}/task_common.txt")
    #     # get_task_infer_openai(pth=f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{file}", obj=obj, parts=obj_part, part_attr=part_attributes, description=movements, task=task)
    #     # get_grasp_part_openai(pth=f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{file}/visual_grasps", obj=obj, part=obj_part)
    #     get_grasp_motion_openai(pth=f"/media/robot/data/WCL/taskgrasp/taskgrasp_image/scans/{file}", obj=obj, part=obj_part, part_attr=part_attributes, description=movements, grasp_part=grasp_part)
