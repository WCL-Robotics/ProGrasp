# deepspeed --include localhost:0 \
#     llava/train/train_mem.py \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path ./pretrained_llms/llava-3d-7b \
#     --version v1 \
#     --data_path ./data/grasp_anything \
#     --dialogue True \
#     --cot True \
#     --vision_tower pretrained_llms/clip-vit-large-patch14-336 \
#     --video_tower SpatialAwareModule \
#     --num_sample_tokens 1152 \
#     --mm_projector_type mlp2x_gelu \
#     --tune_mm_mlp_adapter True \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir ./checkpoints/llava-graspcot \
#     --num_train_epochs 4 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 214 \
#     --save_total_limit 4 \
#     --learning_rate 1e-4 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --model_max_length 4096 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to tensorboard \
#     --logging_first_step True 
python -u llava/train/train_mem.py \
    --model_name_or_path ./pretrained_llms/llava-3d-7b \
    --lora_model_path /home/robot/WCL/GraspCoT/checkpoints/llava-graspcot-lora/ \
    --lora_enable False \
    --lora_alpha 16 \
    --lora_r 16 \
    --lora_dropout 0.05 \
    --lora_bias none \
    --version v1 \
    --data_path /media/robot/data/WCL/taskgrasp/taskgrasp_image \
    --dialogue True \
    --cot True \
    --vision_tower pretrained_llms/clip-vit-large-patch14-336 \
    --video_tower SpatialAwareModule \
    --voxel_tower SecondVoxelnet \
    --grasp_tower GraspNet \
    --num_sample_tokens 1152 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-graspcot-lora\
    --num_train_epochs 35 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 300 \
    --save_total_limit 4 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --logging_first_step True