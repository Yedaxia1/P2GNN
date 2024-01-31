import argparse
import atexit
import json
import os
import pickle
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
cpu_num = 8 # 这里设置成你想运行的CPU个数
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
from ProG.utils import seed_everything, seed

seed_everything(seed)

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.data import DenseDataLoader
from ProG.pre_train import PreTrain
from ProG.utils import load_data4pretrain
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
from meta_demo import called4hy

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("create folder {}".format(path))
    else:
        print("folder exists! {}".format(path))
        
def pretrain(args):
    mkdir('./pre_trained_gnn/')

    # pretext = args['pretext']  # 'GraphCL', 'SimGRACE'
    # gnn_type = args['gnn_type']  # 'GAT', ''TransformerConv
    dataname, num_parts = args['dataname'], args['pretrain_num_parts']

    print("load data...")
    graph_list, input_dim = load_data4pretrain(dataname, num_parts)
    
    print("create PreTrain instance...")
    pt = PreTrain(args)
    
    pt.model.cuda()
    print("pre-training...")
    pt.train(dataname, graph_list, batch_size=args['batch_size'],
             aug1=args['aug1'], aug2=args['aug2'], aug_ratio=args['aug_ratio'],
             lr=args['pretrain_lr'], decay=args['pretrain_decay'], epochs=args['pretrain_epoch'], 
             step_size=args['pretrain_step_size'], gamma=args['pretrain_gamma'])

def main(args): 
    # args['out_dim'] = args['hid_dim']
    # pretrain(args)
    acc = called4hy(args)
    
    return {
            "loss": -acc,
            'status': STATUS_OK,
            'params': args
        }
        


if __name__ == '__main__':
    def save_result(result_file,trials):
        print("正在保存结果...")
        with open(result_file, "w+") as f:
            for result in trials.results:
                if 'loss' in result and result['loss'] <= trials.best_trial['result']['loss']:
                    print(result, file=f)
        print("结果已保存 {:s}".format(result_file))
        print(trials.best_trial)
    def initial_hyperopt(trial_file,result_file,max_evals):
        try:
            with open(trial_file, "rb") as f:
                trials = pickle.load(f)
            current_process = len(trials.results)
            print("使用已有的trial记录, 现有进度: {:d}/{:d} {:s}".format(current_process,max_evals,trial_file))
        except:
            trials = Trials()
            print("未找到现有进度, 从0开始训练 0/{:d} {:s}".format(max_evals, trial_file))
        atexit.register(save_result,result_file,trials)
        return trials

    max_evals = 50
    
    args_dataset = 'CiteSeer'  # 'CiteSeer'  # 'PubMed' 'Cora'  Computers
    args_pretext = 'SimGRACE'     # 'SimGRACE'    # 'GraphCL'
    args_gnn_type = 'GAT'    # 'GCN'  # 'GAT'  # 'TransformerConv'
    with open("./config/{}.{}.{}.json".format(args_dataset, args_pretext, args_gnn_type)) as params:
        args = json.load(params)
        
    # args['pos_enc'] = hp.choice('pos_enc',[None,'diffusion','pstep','adj'])
    # args['lap_pe'] = hp.choice('lap_pe',[True, False])
    # args['batch_size'] = hp.choice('batch_size',[64,128,256])
    # args['fusion_layers'] = hp.choice('fusion_layers',[1,2,3,4])
    
    # args['hid_dim'] = hp.choice('hid_dim',[64, 128, 256])
    # args['cross_prune'] = hp.choice('cross_prune',[0.1,0.3,0.5])
    # args['gcn_layer_num'] = hp.choice('gcn_layer_num',[2,4])
    # args['aug_ratio'] = hp.choice('aug_ratio',[0.1,0.2,0.3])
    # args['pretrain_lr'] = hp.choice('pretrain_lr', [0.01,0.005,0.001])
    # args['pretrain_gamma'] = hp.choice('pretrain_gamma',[0.1, 0.3, 0.5, 0.7])

    args['meta_adapt_lr'] = hp.choice('meta_adapt_lr',[0.1, 0.01, 0.001])
    args['meta_lr'] = hp.choice('meta_lr',[0.1, 0.01, 0.001])
    args['meta_train_adapt_steps'] = hp.choice('meta_train_adapt_steps',[1,2])
    args['meta_test_adapt_steps'] = hp.choice('meta_test_adapt_steps',[2,5,10,20])
    args['meta_train_epoch'] = hp.choice('meta_train_epoch',[5,10])
    
    save_root = os.path.join("hyperopt")
    result_file = os.path.join(save_root, f"result.log")
    trial_file = os.path.join(save_root, f"result.trial")

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    trials = initial_hyperopt(trial_file,result_file,max_evals)
    best = fmin(
        fn=main,space=args, algo=tpe.suggest, max_evals=max_evals, 
        trials = trials, trials_save_file=trial_file)
