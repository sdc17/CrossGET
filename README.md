<div align="center">
<h1>CrossGET: Cross-Guided Ensemble of Tokens for Accelerating Vision-Language Transformers</h1>
</div>

<p align="center">
    <a href="https://arxiv.org/pdf/2305.17455.pdf">
        <img alt="Paper" src="https://img.shields.io/badge/paper-link-blue?logo=quicklook" />
    </a>
    <a href="https://arxiv.org/abs/2305.17455">
        <img alt="ArXiv" src="https://img.shields.io/badge/arXiv-2301.13741-B31B1B?logo=arxiv" />
    </a>
    <a href="https://github.com/sdc17/CrossGET">
        <img alt="Code" src="https://img.shields.io/badge/code-link-181717?logo=github" />
    </a>
</p>

## On LLaVA-1.5

### 🏃 Installation
The code is tested on `Pytorch==2.1.1`, `cuda==12.1`, and `python==3.10.13`. Please follow [LLaVA-1.5 repo](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#install) to install other dependencies.

### 📑 Evaluation

1. Download [playground/data](https://github.com/haotian-liu/LLaVA/tree/main/playground/data) from [LLaVA-1.5 repo](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#install).
2. Follow instructions [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) for preparing datasets.
3. Download following checkpoints and put them under `LLaVA/checkpoints/`.

    Model | Link | 
    --- | :---: 
    LLaVA-1.5-7B with CrossGET | [Google Drive](https://drive.google.com/drive/folders/1E1Qegfy1yeBUwX6rXW6q6kakb3VPtXxU?usp=sharing)
    LLaVA-1.5-13B with CrossGET | [Google Drive](https://drive.google.com/drive/folders/1EjL-u602_DBLhT9mmfgJj_0uCyQRz8YX?usp=sharing)

4. Use scripts under [LLaVA/scripts/v1_5/eval](https://github.com/sdc17/CrossGET/tree/main/LLaVA/scripts/v1_5/eval) and follow instructions [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) for evaluation. Logs are provided under [LLaVA/log](https://github.com/sdc17/CrossGET/tree/main/LLaVA/log).

    Dataset | VQAv2 | GQA | VisWiz | SQA^I | VQA^T | POPE | MME | MMB | MMB^CN | SEED^I
    --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: 
    LLaVA-1.5-7B | 78.5 | 62.0 | 50.0 | 66.8 | 58.2 | 85.9 | 1510.7 | 64.3 | 58.3 | 66.2
    w/ CrossGET (~1.9x Tput) | 77.3 | 61.4 | 47.7 | 66.7 | 54.9 | 83.9 | 1510.2 | 64.7 | 55.2 | 64.4
    
    Dataset | VQAv2 | GQA | VisWiz | SQA^I | VQA^T | POPE | MME | MMB | MMB^CN | SEED^I
    --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: 
    LLaVA-1.5-13B | 80.0 | 63.3 | 53.6 | 71.6 | 61.3 | 85.9 | 1531.3 | 67.7 | 63.6 | 68.2
    w/ CrossGET (~2.0x Tput) | 78.7 | 62.6 | 51.8 | 71.4 | 58.0 | 84.9 | 1548.8 | 66.3 | 62.0 | 67.5

### 📚 Visual Instruction Tuning

1. Download [playground/data](https://github.com/haotian-liu/LLaVA/tree/main/playground/data) from [LLaVA-1.5 repo](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#install).
2. Follow instructions [here](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#visual-instruction-tuning) for preparing datasets.
3. Run `python LLaVA/scripts/construct_dataset.py` to create 'llava_v1_5_mix67k.json'.
4. Follow instructions [here](https://github.com/haotian-liu/LLaVA/tree/main?tab=readme-ov-file#visual-instruction-tuning) for visual instruction tuning. For example, use [LLaVA/scripts/v1_5/finetune_task.sh](https://github.com/sdc17/CrossGET/blob/main/LLaVA/scripts/v1_5/finetune_task.sh)

    ```bash
    #!/bin/bash
    
    deepspeed llava/train/train_mem.py \
        --deepspeed ./scripts/zero3.json \
        --model_name_or_path liuhaotian/llava-v1.5-7b \
        --version v1 \
        --data_path ./playground/data/llava_v1_5_mix67k.json \
        --image_folder ./playground/data \
        --vision_tower openai/clip-vit-large-patch14-336 \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir ./checkpoints/llava-v1.5-7b-mix67k-ours \
        --num_train_epochs 1 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 50000 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --report_to wandb
    ```

## 💬 Acknowledgments
This code is built upon <a href="https://github.com/haotian-liu/LLaVA">LLaVA</a> and <a href="https://github.com/facebookresearch/ToMe">ToMe</a>. Thanks for these awesome open-source projects!


## ✨ Citation
```bibtex
@article{shi2023crossget,
  title={Crossget: Cross-guided ensemble of tokens for accelerating vision-language transformers},
  author={Shi, Dachuan and Tao, Chaofan and Rao, Anyi and Yang, Zhendong and Yuan, Chun and Wang, Jiaqi},
  journal={arXiv preprint arXiv:2305.17455},
  year={2023}
}
```
