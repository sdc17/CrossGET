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
</p>

## On LLaVA-1.5

### ⚙️ Installation
The code is tested on `Pytorch==2.1.1`, `cuda==12.1`, and `python==3.10.13`. Please follow [LLaVA-1.5](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#install) to install other dependencies.

### 📑 Evaluation

1. Download [playground/data](https://github.com/haotian-liu/LLaVA/tree/main/playground/data) from [LLaVA-1.5](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#install).
2. Follow instructions [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) for preparing datasets.
3. Download following checkpoints and put them under `checkpoints/`.

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

1. Download [playground/data](https://github.com/haotian-liu/LLaVA/tree/main/playground/data) from [LLaVA-1.5](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#install).
2. Follow instructions [here](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#visual-instruction-tuning) for preparing datasets.
3. Run `python LLaVA/scripts/construct_dataset.py` to create `llava_v1_5_mix67k.json`.
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

---

## On CLIP

### ⚙️ Installation
The code is tested on `Pytorch==2.0.0`, `cuda==11.7`, and `python==3.9.16`. The dependencies can be installed by:
```
conda env create -f environment.yml
```

### 📑 Evaluation

1. Download the [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) dataset, unzip it under the `datasets` folder, and accordingly modify the `image_root` in [config](https://github.com/sdc17/CrossGET/blob/main/CLIP/configs/retrieval_flickr_clip.yaml).
2. Download annotations from [Google Drive](https://drive.google.com/uc?export=download&id=19Vk07K3DbQYa68DipJ4dFNcF0_Br7cmD) or [Baidu Drive](https://pan.baidu.com/s/1lJbfCsXqszmaTvxrpt7-xQ?pwd=1c2i), unzip it under the `annotation` folder, and accordingly modify the `annotation` in [config](https://github.com/sdc17/CrossGET/blob/main/CLIP/configs/retrieval_flickr_clip.yaml).
3. Download the following checkpoint and put it under `output/`.

    Model | Link | 
    --- | :---: 
    CLIP with CrossGET | [Google Drive](https://drive.google.com/drive/folders/1bc5ZCjSVRsOiir8wv672JxMHGQg5JwLD?usp=sharing)

4. Use the following script to evaluate. Logs are provided under [CLIP/log](https://github.com/sdc17/CrossGET/tree/main/CLIP/log).

    ```bash
    python -m torch.distributed.run --nproc_per_node=8 train_retrieval_clip.py \
    --config ./configs/retrieval_flickr_clip.yaml --evaluate --reduce ours --rv 16 --rl 0 \
    --pretrained output/train_retrieval_flickr_clip_ours/checkpoint_best.pth \
    --output_dir output/evaluate_retrieval_flickr_clip_ours_test
    ```

    Image to Text (Flickr30k) | Recall@1 | Recall5 | Recall@10 
    --- | :---: | :---: | :---: 
    CLIP | 92.1 | 99.1 | 99.7 
    w/ CrossGET (~58% GFLOPs) | 92.1 | 99.7 | 99.8

    Text to Image (Flickr30k) | Recall@1 | Recall5 | Recall@10 
    --- | :---: | :---: | :---: 
    CLIP | 79.3 | 95.7 | 98.0
    w/ CrossGET (~58% GFLOPs) | 79.6 | 95.7 | 98.0
   

### 📚 Fine-tuning

1. Download the [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) dataset, unzip it under the `datasets` folder, and accordingly modify the `image_root` in [config](https://github.com/sdc17/CrossGET/blob/main/CLIP/configs/retrieval_flickr_clip.yaml).
2. Download annotations from [Google Drive](https://drive.google.com/uc?export=download&id=19Vk07K3DbQYa68DipJ4dFNcF0_Br7cmD) or [Baidu Drive](https://pan.baidu.com/s/1lJbfCsXqszmaTvxrpt7-xQ?pwd=1c2i), unzip it under the `annotation` folder, and accordingly modify the `annotation` in [config](https://github.com/sdc17/CrossGET/blob/main/CLIP/configs/retrieval_flickr_clip.yaml).
3. Download the following checkpoint and put it under `output/`.

    Model | Link | 
    --- | :---: 
    CLIP | [Google Drive](https://drive.google.com/drive/folders/120n9RWZ0Ir8vzHP2Zd7b-6xJ-e5WxmFC?usp=sharing)
   
4. Use the following scripts to fine-tuning. Logs are provided under [CLIP/log](https://github.com/sdc17/CrossGET/tree/main/CLIP/log).

    ```bash
    # Vision-only
    python -W ignore -m torch.distributed.run --nproc_per_node=8 train_retrieval_clip.py \
    --config ./configs/retrieval_flickr_clip.yaml --lr 1e-5 --epoch 12 --reduce ours --rv 16 --rl 0 \
    --pretrained output/finetune_retrieval_flickr_clip/checkpoint_best.pth \
    --output_dir output/train_retrieval_flickr_clip_ours
    ```

    ```bash
    # Language-only
    python -W ignore -m torch.distributed.run --nproc_per_node=8 train_retrieval_clip.py \
    --config ./configs/retrieval_flickr_clip.yaml --lr 1e-5 --epoch 12 --reduce ours --rv 0 --rl 6 \
    --pretrained output/finetune_retrieval_flickr_clip/checkpoint_best.pth \
    --output_dir output/train_retrieval_flickr_clip_ours
    ```

    ```bash
    # Vision and Language
    python -W ignore -m torch.distributed.run --nproc_per_node=8 train_retrieval_clip.py \
    --config ./configs/retrieval_flickr_clip.yaml --lr 1e-5 --epoch 12 --reduce ours --rv 16 --rl 6 \
    --pretrained output/finetune_retrieval_flickr_clip/checkpoint_best.pth \
    --output_dir output/train_retrieval_flickr_clip_ours
    ```

---


## On CoOp Benchmark

### ⚙️ Installation
The code is tested on `Pytorch==2.0.0`, `cuda==11.7`, and `python==3.9.17`. Please follow [CoOp](https://github.com/KaiyangZhou/CoOp?tab=readme-ov-file#how-to-install) to install other dependencies.

### 📑 Evaluation

1. Prepare datasets following instructions [here](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md).
2. Download checkpoints ending **without** `_original` from the following link and put them under `output/`. Logs are also provided in the same link.

    Model | Link | 
    --- | :---: 
    CoOp with CrossGET | [Google Drive](https://drive.google.com/drive/folders/1-j_ardAxXow9wrCbhGon-YNiOM-AOOgL?usp=sharing)

3. Adjust `model-dir` in [CoOp/eval.sh](https://github.com/sdc17/CrossGET/blob/main/CoOp/eval.sh) accordingly.
4. Use the following scripts to evaluate.

    ```bash
    # Second last parameter: 16 for ~58% GFLOPs, 12 for ~69% GFLOPs, and 8 for ~80% GFLOPs
    ./eval.sh stanford_cars vit_b16 ours 16 0 
    ./eval.sh oxford_flowers vit_b16 ours 16 0 
    ./eval.sh food101 vit_b16 ours 16 0 
    ./eval.sh fgvc_aircraft vit_b16 ours 16 0 
    ./eval.sh sun397 vit_b16 ours 16 0 
    ./eval.sh dtd vit_b16 ours 16 0 
    ./eval.sh eurosat vit_b16 ours 16 0 
    ./eval.sh ucf101 vit_b16 ours 16 0 
    ./eval.sh caltech101 vit_b16 ours 16 0 
    ./eval.sh oxford_pets vit_b16 ours 16 0 
    ./eval.sh imagenet vit_b16_ep50 ours 16 0 
    ```

    Dataset | ImageNet | Caltech101 | OxfordPets | StanfordCars | Flowers102 | Food101 | FGVCAircraft | SUN397 | DTD | EuroSAT | UCF101 | Average
    --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: 
    CoOp | 71.1 | 95.4 | 93.3 | 77.5 | 95.6 | 86.5 | 37.3 | 75.1 | 65.8 | 82.6 | 83.7 | 78.5 
    w/ CrossGET (~80% GFLOPs) | 70.8 | 94.6 | 90.8 | 81.9 | 95.8 | 82.0 | 43.7 | 74.1 | 65.7 | 88.4 | 82.2 | 79.1
    w/ CrossGET (~69% GFLOPs) | 70.2 | 94.9 | 90.1 | 81.1 | 95.0 | 81.5 | 43.1 | 73.5 | 65.9 | 86.9 | 81.9 | 78.6
    w/ CrossGET (~58% GFLOPs) | 67.6 | 93.9 | 89.5 | 76.6 | 93.3 | 79.7 | 41.3 | 72.1 | 64.2 | 84.5 | 80.5 | 76.7

### 📚 Prompt Tuning

1. Prepare datasets following instructions [here](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md).
2. Download checkpoints ending **with** `_original` from the following link and put them under `output/`. Logs are also provided in the same link.

    Model | Link | 
    --- | :---: 
    CoOp | [Google Drive](https://drive.google.com/drive/folders/1-j_ardAxXow9wrCbhGon-YNiOM-AOOgL?usp=sharing)

3. Adjust `model-dir` in [CoOp/train.sh](https://github.com/sdc17/CrossGET/blob/main/CoOp/train.sh) accordingly.
4. Use the following scripts to prompt tuning.

    ```bash
    # Second last parameter: 16 for ~58% GFLOPs, 12 for ~69% GFLOPs, and 8 for ~80% GFLOPs
    ./train.sh stanford_cars vit_b16 end 16 16 False ours 16 0
    ./train.sh oxford_flowers vit_b16 end 16 16 False ours 16 0
    ./train.sh food101 vit_b16 end 16 16 False ours 16 0
    ./train.sh fgvc_aircraft vit_b16 end 16 16 False ours 16 0
    ./train.sh sun397 vit_b16 end 16 16 False ours 16 0
    ./train.sh dtd vit_b16 end 16 16 False ours 16 0
    ./train.sh eurosat vit_b16 end 16 16 False ours 16 0
    ./train.sh ucf101 vit_b16 end 16 16 False ours 16 0
    ./train.sh caltech101 vit_b16 end 16 16 False ours 16 0
    ./train.sh oxford_pets vit_b16 end 16 16 False ours 16 0
    ./train.sh imagenet vit_b16_ep50 end 16 16 False ours 16 0
    ```

## 💬 Acknowledgments
This code is built upon <a href="https://github.com/haotian-liu/LLaVA">LLaVA</a>, <a href="https://github.com/facebookresearch/ToMe">ToMe</a>, <a href="https://github.com/sdc17/UPop">UPop</a>, <a href="https://github.com/salesforce/BLIP">BLIP</a>, and <a href="https://github.com/KaiyangZhou/CoOp">CoOp</a>. Thanks for these awesome open-source projects!


## ✨ Citation
```bibtex
@article{shi2023crossget,
  title={Crossget: Cross-guided ensemble of tokens for accelerating vision-language transformers},
  author={Shi, Dachuan and Tao, Chaofan and Rao, Anyi and Yang, Zhendong and Yuan, Chun and Wang, Jiaqi},
  journal={arXiv preprint arXiv:2305.17455},
  year={2023}
}
```
