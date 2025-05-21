'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import argparse
import copy
import json
import os
from torch.cuda.amp import GradScaler, autocast
import networkx as nx
import numpy as np
import random
import time
import datetime
from pathlib import Path
import dill as pickle
import torch
from ruamel.yaml import YAML
import torch.backends.cudnn as cudnn
from torch import nn
from tqdm import tqdm

from pj.batch import multidigraph_to_geometric
from pj.dataset.graphtext_dataset import GraphTextDataset, load_pairs
from pj.eval import load_test_dataset, test, validate
from pj.models.model_pretrain import CGLP
from models.tokenization_bert import BertTokenizer
import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

def process_text_to_inputs(text, tokenizer, max_length=512, device='cuda'):
    """
    Split the text into sentences and convert them into model input format.

    Args:
        text (str): Input text
        tokenizer: Tokenizer
        max_length (int): Maximum sequence length
        device (str): Device to use ('cuda' or 'cpu')

    Returns:
        list: Model inputs for each sentence
    """
    # Store processed inputs
    processed_inputs = []

    # Split text into sentences with max_length characters
    sentences = [text[i:i + max_length] for i in range(0, len(text), max_length)]

    # Process each sentence
    for sentence in sentences:
        # Tokenize and add special tokens
        inputs = tokenizer(
            sentence,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            padding='longest'
        ).to(device)

        processed_inputs.append(inputs)

    return processed_inputs

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mgm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('itm_acc', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('itm_pos_acc', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('ita_acc', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 500
    step_size = 100
    warmup_iterations = warmup_steps*step_size
    accumulation_steps = 1  # 累积4个batch再更新
    pos_rst = []

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (graph, text) in tqdm(enumerate(metric_logger.log_every(data_loader, print_freq, header))):

        optimizer.zero_grad()  # 清空梯度
        # 计算alpha
        alpha = config['alpha'] if epoch > 0 else config['alpha'] * min(1, ( epoch * len(data_loader) + i) / warmup_iterations)

        loss_mlm, loss_gtc, loss_gtm, loss_mgm, itm_acc, ita_acc, itm_pos_acc = model(graph, text, alpha = alpha)
        loss = (config['mlm_coef'] *loss_mlm + config['gtc_coef'] * loss_gtc + config['gtm_coef'] * loss_gtm + config['mgm_coef']*loss_mgm)/ accumulation_steps  # 注意除以累积步数

        pos_rst.append(itm_pos_acc.item())
        # 反向传播
        loss.backward()
        optimizer.step()

        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_mgm=loss_mgm.item())
        metric_logger.update(loss_ita=loss_gtc.item())
        metric_logger.update(loss_itm=loss_gtm.item())
        metric_logger.update(itm_acc=itm_acc.item())
        metric_logger.update(ita_acc=ita_acc.item())
        metric_logger.update(itm_pos_acc=itm_pos_acc.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

def main(args, config):
    #### initialize the distributed environment ####
    if args.distributed:
        utils.init_distributed_mode(args)

    device = torch.device(f'cuda:{args.gpu}')

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    val_epoch_interval = config['val_epoch_interval']
    bs = config['batch_size']
    # #### Dataset ####
    print("Creating dataset")
    # datasets = [create_dataset('pretrain', config)]
    # load constructed data pairs
    data = load_pairs(args.dataset)
    train_data, validation_data = data[:int(len(data)*0.8)], data[int(len(data)*0.2):]
    graphtext_dataset_train = [GraphTextDataset(train_data)]
    graphtext_dataset_val = [GraphTextDataset(validation_data)]

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(graphtext_dataset_train, [True], num_tasks, global_rank)
    else:
        samplers = [None]

    train_data_loader = create_loader(graphtext_dataset_train,samplers,batch_size=[bs], num_workers=[4], is_trains=[True], collate_fns=[None])[0]
    val_data_loader = create_loader(graphtext_dataset_val,samplers,batch_size=[bs*3], num_workers=[4], is_trains=[True], collate_fns=[None])[0]
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    #### Model ####
    print("Creating model")
    model = CGLP(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)

    model.to(device)

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module

    best_val_acc = 0.9
    print("Start training")
    start_time = time.time()

    test_datasets = {
        'optc': load_test_dataset('optc'),
        'trace': load_test_dataset('trace'),
        'cadets': load_test_dataset('cadets'),
        'theia': load_test_dataset('theia')
    }
    for epoch in range(start_epoch, max_epoch):

        model.train()
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)

        train_stats = train(model, train_data_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)

        if epoch%val_epoch_interval==0 and epoch >= 0:
            model.eval()
            with torch.no_grad():
                head_val_acc, sim_val_acc = validate(model, val_data_loader, tokenizer, device, config)
                # head_val_acc, sim_val_acc = validate(model, train_data_loader, tokenizer, device, config)
            print(f'Validation acc of head:{head_val_acc:.4f}, acc of sim:{sim_val_acc:.4f}.')
            if sim_val_acc > best_val_acc:
                best_val_acc = sim_val_acc
        #         if utils.is_main_process():
        #             # torch.save(model.state_dict(), os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
                print('save best model. Begin to test darpa datasets....')
                datasets = ['optc', 'trace', 'cadets', 'theia']
                for dataset in datasets:
                    print(f"==================================================Testing on {dataset}")
                    # test_dataset = load_test_dataset(dataset)
                    test_dataset = test_datasets[dataset]
                    fp_idx, top1_recall, top3_recall = test(model, test_dataset, tokenizer, device, config)
                    # if epoch > 10:
                    #     if top3_recall == 1:
                    #         print(f"top3 recall is 1, stop training")
                    #         # dump the model
                    #         torch.save(model.state_dict(), os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))
                    #     # print("intersection fp graphs number: ", len(fp_idx))
                    #         test_dataset = delete_fp_samples(test_dataset, fp_idx)
                    #         test_datasets[dataset] = test_dataset
                    #         # dump the test_dataset
                    #         with open(os.path.join(args.output_dir,f'filtered_tp_graphs.pkl'), 'wb') as f:
                    #             pickle.dump(test_dataset, f)
                    #         fp_idx, top1_recall, top3_recall = test(model, test_dataset, tokenizer, device, config)
                    #
                    #     with open(f'../data/darpa/{dataset}/test/filtered_tp_graphs.pkl', 'wb') as f:
                    #         pickle.dump(test_dataset, f)

        torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'checkpoint_%02d.pth' % epoch))
        # if utils.is_main_process():
        #     log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #                  'epoch': epoch,
        #                 }
        #     save_obj = {
        #         'model': model_without_ddp.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'lr_scheduler': lr_scheduler.state_dict(),
        #         'config': config,
        #         'epoch': epoch,
        #     }
        #     torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))
        #
        #     with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
        #         f.write(json.dumps(log_stats) + "\n")
        # Synchronization operation waits until all processes reach this step
        # dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    import sys
    import os

    # Add project root directory to sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, project_root)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--dataset', default='')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='Pretrain/')
    # parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_encoder', default='./bert_files/')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    # parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--train_file', default="")
    args = parser.parse_args()

    config = YAML(typ='rt').load(open(args.config, 'r'))
    args.output_dir = f"dataset/{args.dataset}/"
    args.ckpt_dir = f"ckpts_{config['mlm_coef']}_{config['mgm_coef']}_{config['gtm_coef']}_{config['gtc_coef']}/"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    YAML(typ='unsafe', pure=True).dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    torch.multiprocessing.set_sharing_strategy('file_system')
    main(args, config)