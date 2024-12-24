'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
# torch.use_deterministic_algorithms(True, warn_only=True)

from clip import clip
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader

import io
from petrel_client.client import Client
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count_table

# from line_profiler import LineProfiler
from torch.cuda.amp import autocast as autocast

def train(model, data_loader, optimizer, epoch, device, config, scaler=None, w=0.5):
    # train
    model.train()  
    reduce = model.module.reduce
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if reduce=='ours':
        metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_kd', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i,(image, caption, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device,non_blocking=True)   
        idx = idx.to(device,non_blocking=True)   
        
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))
        
        with autocast():
            if reduce=='ours':
                loss_ita, loss_kd = model(image, caption, alpha=alpha, idx=idx)   
                loss = (1 - w) * loss_ita + w * loss_kd
            else:
                loss = model(image, caption, alpha=alpha, idx=idx)   
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        metric_logger.update(loss=loss.item())
        if reduce == 'ours':
            metric_logger.update(loss_ita=loss_ita.item() * (1 - w))
            metric_logger.update(loss_kd=loss_kd.item() * w)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


@torch.no_grad()
def evaluation(model, data_loader, device, config):
    # test
    model.eval() 
    print('Computing features for evaluation...')
    texts = data_loader.dataset.text   
    num_text = len(texts)
    text_bs = 256
    text_embeds = []  
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = model.tokenize(text).to(device) 
        text_output = model.encode_text(text_input)
        text_embed = text_output / text_output.norm(dim=1, keepdim=True)
        text_embeds.append(text_embed)   
    text_embeds = torch.cat(text_embeds,dim=0)

    image_embeds = []
    for image, img_id in data_loader: 
        image = image.to(device) 
        image_feat = model.encode_image(image)
        image_embed = image_feat / image_feat.norm(dim=1, keepdim=True)
        image_embeds.append(image_embed)
    image_embeds = torch.cat(image_embeds,dim=0)

    sims_matrix = image_embeds @ text_embeds.t()
    return sims_matrix.cpu().numpy(), sims_matrix.t().cpu().numpy()

            
@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    #Images->Text 
    ranks = np.zeros(scores_i2t.shape[0])
    for index,score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    #Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r_mean': tr_mean,
                    'img_r1': ir1,
                    'img_r5': ir5,
                    'img_r10': ir10,
                    'img_r_mean': ir_mean,
                    'r_mean': r_mean}
    return eval_result


def main(args, config, client=None):
    utils.init_distributed_mode(args)    
    
    config['max_epoch'] = args.epoch
    config['init_lr'] = args.lr
    config['pretrained'] = args.pretrained

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    # seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.enabled=False
    cudnn.deterministic = True

    #### Dataset #### 
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('retrieval_%s'%config['dataset'], config, client)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],
                                                          is_trains=[True, False, False], 
                                                          collate_fns=[None,None,None])   
    
    #### Model #### 
    print("Creating model")
    model, preprocess = clip.load(name=config['pretrained'], device=device, client=client, 
                                  evaluate=args.evaluate, reduce=args.reduce, r={'rv': args.rv, 'rl': args.rl})
    model.tokenize = clip.tokenize
    # model.init_cross() ###
    #### Count parameters and FLOPs ####
    model.eval()
    class Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, inputs):
            image, text, alpha, idx = inputs
            return self.model(image, text, alpha, idx)
            # return self.model.inference(image, text, alpha, idx)
    with torch.no_grad():
        wrapper_model = Wrapper(model); 
        inputs = [torch.randn(1, 3, config['image_size'], config['image_size']).to(device) , 
                ["car driving down a road behind a lot of sheep"], 
                0.0,
                torch.full((1, ),-100).to(device) 
                ]
        flop = FlopCountAnalysis(wrapper_model, inputs)
        print(flop_count_table(flop, max_depth=7, show_param_shapes=True))
        print("Total", flop.total() / 1e9)
    # model.reset_queue()
    model.train()
    #####################################
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module   

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay']) 

    best = 0
    best_epoch = 0
    scaler = torch.cuda.amp.GradScaler()

    print("Start training")
    start_time = time.time()    
    for epoch in range(0, config['max_epoch']):    
        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            
            train_stats = train(model, train_loader, optimizer, epoch, device, config, scaler=scaler)  

        score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, device, config)
        score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, device, config)
    
        if utils.is_main_process():  
      
            val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)  
            print(val_result)
                                
            if val_result['r_mean']>best:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                if client is not None:
                    with io.BytesIO() as f:
                        torch.save(save_obj, f)
                        f.seek(0)
                        client.put(os.path.join('s3://sdcBucket/TokenCompression', args.output_dir, 'checkpoint_best.pth'), f)
                else:
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))  
                best = val_result['r_mean']        
                best_epoch = epoch  
                
                test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt) 
                print(test_result)
            
            if args.evaluate:                
                log_stats = {**{f'val_{k}': v for k, v in val_result.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},                  
                            }
                # with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                #     f.write(json.dumps(log_stats) + "\n")     
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_result.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},  
                             'epoch': epoch,
                             'best_epoch': best_epoch,
                            }
                # with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                #     f.write(json.dumps(log_stats) + "\n")   
            print("LOG: ", log_stats)

        if args.evaluate: 
            break

        dist.barrier()     
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/retrieval_flickr.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_flickr')        
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--lr', default=1e-6, type=float, help='learning rate')
    parser.add_argument('--epoch', default=12, type=int, help='number of epoches')
    parser.add_argument('--pretrained', default='output/finetune_retrieval_flickr_clip/checkpoint_best.pth', type=str)
    parser.add_argument('--reduce', default='ours', type=str, choices=['none', 'ours'])
    parser.add_argument('--rv', default=16, type=int)
    parser.add_argument('--rl', default=0, type=int)
    
    
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    # client = Client('~/petreloss.conf', enable_mc=True)
    # client.put(os.path.join('s3://sdcBucket/TokenCompression', args.output_dir, 'config.yaml'), yaml.dump(config))
    
    # main(args, config, client)
    main(args, config)