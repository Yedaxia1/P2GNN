import os
import random 
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
# cpu_num = 8 # 这里设置成你想运行的CPU个数
# os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
# os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
# os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
# os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
# os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)


from torch import nn, optim
import torch
# torch.set_num_threads(cpu_num)
from copy import deepcopy
from ProG.utils import seed, seed_everything
from random import sample, shuffle
from ProG.meta import MAML
from ProG.eva import acc_f1_over_batches
from data_preprocess import load_tasks
from torch_geometric.loader import DataLoader
import json
from imgs import print_loss_line
from itertools import combinations
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
seed_everything(seed)

from ProG.prompt import GNN, FrontAndHead, Prompted_GNN


def meta_test_adam(meta_test_task_id_list,
                   dataname,
                   batch_size,
                   K_shot,
                   seed,
                   maml, gnn,
                   adapt_steps_meta_test,
                   lossfn,
                   args=None):
    # meta-testing
    if len(meta_test_task_id_list) < 2:
        raise AttributeError("\ttask_id_list should contain at leat two tasks!")

    shuffle(meta_test_task_id_list)

    task_pairs = [(meta_test_task_id_list[i], meta_test_task_id_list[i + 1]) for i in
                  range(0, len(meta_test_task_id_list) - 1, 2)]

    for task_1, task_2, support, query, _ in load_tasks('test', task_pairs, dataname, K_shot, seed):

        test_model = deepcopy(maml.module)
        test_opi = optim.Adam(filter(lambda p: p.requires_grad, test_model.parameters()),
                              lr=0.001,
                              weight_decay=0.0001)
        
        test_model.train()

        support_loader = DataLoader(support.to_data_list(), batch_size=batch_size, shuffle=True)
        query_loader = DataLoader(query.to_data_list(), batch_size=batch_size, shuffle=True)

        for _ in range(adapt_steps_meta_test):
            running_loss = 0.
            for batch_id, support_batch in enumerate(support_loader):
                if args['is_cuda']:
                    support_batch.cuda()
                support_preds = test_model(support_batch)
                support_loss = lossfn(support_preds, support_batch.y)
                test_opi.zero_grad()
                support_loss.backward()
                test_opi.step()
                running_loss += support_loss.item()

                if batch_id == len(support_loader) - 1:
                    last_loss = running_loss / len(support_loader)  # loss per batch
                    print('{}/{} training loss: {:.8f}'.format(_, adapt_steps_meta_test, last_loss))
                    running_loss = 0.

        test_model.eval()
        acc, f1, auc = acc_f1_over_batches(args, query_loader, test_model.gnn, test_model.answering, 2, 'multi_class_classification')
        return acc, f1, auc
        ## DO NOT DELETE the following content!
        # metric = torchmetrics.classification.Accuracy(task="binary")  # , num_labels=2)
        # for batch_id, query_batch in enumerate(query_loader):
        #     query_preds = test_model(query_batch,gnn)
        #     pre_class = torch.argmax(query_preds, dim=1)
        #     acc = metric(pre_class, query_batch.y)
        #     # print(f"Accuracy on batch {batch_id}: {acc}")
        #
        # acc = metric.compute()
        # print("""\ttask pair ({}, {}) | Acc: {:.4} """.format(task_1, task_2, acc))
        # metric.reset()


def meta_train_maml(epoch, maml, gnn, lossfn, opt, scheduler, meta_train_task_id_list, dataname, batch_size=10, adapt_steps=2, K_shot=100, args=None):
    if len(meta_train_task_id_list) < 2:
        raise AttributeError("\ttask_id_list should contain at leat two tasks!")

    shuffle(meta_train_task_id_list)

    # task_pairs = [(meta_train_task_id_list[i-1], meta_train_task_id_list[i]) for i in
    #               range(0, len(meta_train_task_id_list), 2)]
    
    task_pairs = [(meta_train_task_id_list[i], meta_train_task_id_list[i + 1]) for i in
                  range(0, len(meta_train_task_id_list) - 1, 2)]

    if len(task_pairs) > 10:
        task_pairs = task_pairs[:5]

    epoch_loss_list = []
    # meta-training
    for ep in range(epoch):
        meta_train_loss = 0.0
        pair_count = 0
        PrintN = 10

        for task_1, task_2, support, query, total_num in load_tasks('train', task_pairs, dataname, K_shot, seed):
            pair_count = pair_count + 1

            learner = maml.clone()
            
            support_loader = DataLoader(support.to_data_list(), batch_size=batch_size, shuffle=False)
            query_loader = DataLoader(query.to_data_list(), batch_size=batch_size, shuffle=False)

            for j in range(adapt_steps):  # adaptation_steps
                running_loss = 0.
                support_loss = 0.
                for batch_id, support_batch in enumerate(support_loader):
                    if args['is_cuda']:
                        support_batch.cuda()

                    support_batch_preds = learner(support_batch)
                    support_batch_loss = lossfn(support_batch_preds, support_batch.y)

                    # learner.adapt(support_batch_loss)
                    running_loss += support_batch_loss.item()
                    support_loss += support_batch_loss

                    # if (batch_id + 1) % PrintN == 0:  # report every PrintN updates
                    #     last_loss = running_loss / PrintN  # loss per batch
                    #     print('adapt {}/{} | batch {}/{} | loss: {:.8f}'.format(j + 1, adapt_steps,
                    #                                                             batch_id + 1,
                    #                                                             len(support_loader),
                    #                                                             last_loss))

                    #     running_loss = 0.
                    
                support_loss = support_loss / len(support_loader)
                learner.adapt(support_loss)
                
                print('adapt {}/{} | loss: {:.8f}'.format(j + 1, adapt_steps, support_loss.item()))
                # opt.zero_grad()
                
                # support_loss.cpu()
                # del support_loss
                # torch.cuda.empty_cache()

            running_loss, query_loss = 0., 0.
            for batch_id, query_batch in enumerate(query_loader):
                if args['is_cuda']:
                    query_batch.cuda()
                    
                query_batch_preds = learner(query_batch)
                query_batch_loss = lossfn(query_batch_preds, query_batch.y)
                query_loss += query_batch_loss
                running_loss += query_batch_loss
                # if (batch_id + 1) % PrintN == 0:
                #     last_loss = running_loss / PrintN
                #     print('query loss batch {}/{} | loss: {:.8f}'.format(batch_id + 1,
                #                                                          len(query_loader),
                #                                                          last_loss))

                #     running_loss = 0.

            query_loss = query_loss / len(query_loader)
            meta_train_loss += query_loss

        print('meta_train_loss @ epoch {}/{}: {}'.format(ep, epoch, meta_train_loss.item()))
        epoch_loss_list.append(meta_train_loss.item())
        meta_train_loss = meta_train_loss / len(meta_train_task_id_list)
        opt.zero_grad()
        meta_train_loss.backward()
        opt.step()
        
        scheduler.step()
        
    # print_loss_line(epoch_loss_list, "{}.{}.{}.meta_train".format(dataname, pre_train_method, gnn_type))


def model_components(args):
    """
    input_dim, dataname, gcn_layer_num=2, hid_dim=16, num_classes=2,
                 task_type="multi_label_classification",
                 token_num=10, cross_prune=0.1, inner_prune=0.3, gnn_type='TransformerConv'

    :param args:
    :param round:
    :param pre_train_path:
    :param gnn_type:
    :param project_head_path:
    :return:
    """
    adapt_lr = args['meta_adapt_lr']
    meta_lr = args['meta_lr']

    # load pre-trained GNN
    # gnn = GNN(input_dim=100, hid_dim=100, out_dim=100, gcn_layer_num=2, gnn_type='TransformerConv')
    gnn = Prompted_GNN(input_dim=args['input_dim'], hid_dim=args['hid_dim'], out_dim=args['out_dim'], gcn_layer_num=args['gcn_layer_num'], pool=None,
                        gnn_type=args['gnn_type'], token_num=args['prompt_token_num'], cross_prune=args['cross_prune'], inner_prune=args['inner_prune'])
    
    pre_train_path = './pre_trained_gnn/{}.{}.{}.pth'.format(args['dataname'], args['pretext'], args['gnn_type'])
    gnn.load_state_dict(torch.load(pre_train_path))
    gnn.frozen_gnn()
    # gnn.eval()
    print("successfully load pre-trained weights for gnn! @ {}".format(pre_train_path))
    
    model = FrontAndHead(hid_dim=args['hid_dim'], num_classes=2,  # 0 or 1
                         task_type=args['task_type'], 
                         gnn=gnn)
    
    # for p in gnn.parameters():
    #     p.requires_grad = False

    maml = MAML(model, lr=adapt_lr, first_order=False, allow_nograd=True)
    
    maml.module.load_state_dict(torch.load(load_path))
    
    lossfn = nn.CrossEntropyLoss(reduction='mean')
    
    if args['is_cuda']:
        maml.cuda()
        lossfn.cuda()
    
    # model_param_group = []
    # model_param_group.append({"params": filter(lambda p: p.requires_grad, gnn.parameters())})

    # model_param_group.append({"params": filter(lambda p: p.requires_grad, maml.parameters())})
    
    # for name, pa in maml.named_parameters():
    #     print(name, pa.requires_grad)
    
    opt = optim.Adam(filter(lambda p: p.requires_grad, maml.parameters()), meta_lr)
    scheduler = lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)
    # scheduler = lr_scheduler.StepLR(opt, step_size=10, gamma=0.3)

    return maml, gnn, opt, scheduler, lossfn

if __name__ == '__main__':    
    
    # args_dataset = 'PubMed'  # 'CiteSeer'  # 'PubMed' 'Cora'  Computers
    # args_pretext = 'SimGRACE'     # 'SimGRACE'    # 'GraphCL'
    # args_gnn_type = 'GCN'    # 'GCN'  # 'GAT'  # 'TransformerConv'
    # with open("./config/{}.{}.{}.json".format(args_dataset, args_pretext, args_gnn_type)) as params:
    #     args = json.load(params)
    
    # dataname = args['dataname']
    # # node-level: 0 1 2 3 4 5
    # # edge-level: 6 7 8 9 10 11
    # # graph-level: 12 13 14 15 16 17
    # meta_train_task_id_list = args['meta_task_id_list'][:-2]
    # meta_test_task_id_list = args['meta_task_id_list'][-2:]
    
    # if len(meta_train_task_id_list) < 2:
    #     meta_train_task_id_list = args['meta_task_id_list'][:2]
        
    # print("meta_train_task_id_list:", meta_train_task_id_list)
    # print("meta_test_task_id_list:", meta_test_task_id_list)

    # pre_train_method = args['pretext']
    # gnn_type = args['gnn_type']

    # maml, gnn, opt, scheduler, lossfn = model_components(args)
    
    # print(opt)

    # # meta training on source tasks
    # meta_train_maml(args['meta_train_epoch'], maml, gnn, lossfn, opt, scheduler, meta_train_task_id_list,
    #                 dataname, args['batch_size'], adapt_steps=args['meta_train_adapt_steps'], K_shot=100, args=args)

    # # meta testing on target tasks
    # adapt_steps_meta_test = args['meta_test_adapt_steps']  # 00  # 50
    # meta_test_adam(meta_test_task_id_list, dataname, args['batch_size'], 100, seed, maml, gnn,
    #                adapt_steps_meta_test, lossfn, args=args)
    
    
    
    args_dataset = 'Computers'  # 'CiteSeer'  # 'PubMed' 'Cora'  Computers 'Reddit' 'ogbn_arxiv'  'ogbn_products'  'dblp'
    args_pretext = 'GraphCL'     # 'SimGRACE'    # 'GraphCL'
    args_gnn_type = 'GCN'    # 'GCN'  # 'GAT'  # 'TransformerConv'
    with open("./config/{}/{}.{}.{}.json".format(args_dataset, args_dataset, args_pretext, args_gnn_type)) as params:
        args = json.load(params)
    
    dataname = args['dataname']
    # node-level: 0 1 2 3 4 5
    # edge-level: 6 7 8 9 10 11
    # graph-level: 12 13 14 15 16 17
    
    num_class = args['num_class']
    
    # edge-level: [num_classes,num_classes*2)
    # graph-level: [num_classes*2,num_classes*3)
    dataset_num_class = {
            'Reddit': 41,
            'ogbn_arxiv': 40,
            'ogbn_products': 47,
            'dblp': 4,
            'Computers': 10,
            'CiteSeer': 6
        }
    
    target_domain = 'Reddit'       # 'CiteSeer'  'Reddit'
    num_class = dataset_num_class[target_domain]
    
    target_level = 'edge_level'     # 'node_level ''graph_level'  'edge_level'
    
    if target_level == 'graph_level':
        meta_task_id_list = list(range(2 * num_class, 3 * num_class))
    elif target_level == 'edge_level':
        meta_task_id_list = list(range(num_class, 2 * num_class))
    elif target_level == 'node_level':
        meta_task_id_list = list(range(num_class))
    
    combinations_list = list(combinations(meta_task_id_list, 2))
    
    test_task_id_list = sample(combinations_list, 5)
    
    load_path = "./Transferability/{}/Computers.GraphCL.GCN.pth".format(target_level)
    
    dataname = target_domain
    
    acc_list, f1_list, auc_list = [], [], []
    for run_time in range(5):
        # combinations_list[run_time % len(combinations_list)]
        # meta_test_task_id_list = list(combinations_list[run_time % len(combinations_list)])
        meta_test_task_id_list = list(test_task_id_list[run_time])
        
        meta_train_task_id_list = list(filter(lambda item: item not in meta_test_task_id_list, meta_task_id_list))
        if len(meta_train_task_id_list) < 2:
            meta_train_task_id_list.append(meta_test_task_id_list[0])
            
        print("meta_train_task_id_list:", meta_train_task_id_list)
        print("meta_test_task_id_list:", meta_test_task_id_list)

        pre_train_method = args['pretext']
        gnn_type = args['gnn_type']

        maml, gnn, opt, scheduler, lossfn = model_components(args)

        # meta training on source tasks
        meta_train_maml(args['meta_train_epoch'], maml, gnn, lossfn, opt, scheduler, meta_train_task_id_list,
                        dataname, args['batch_size'], adapt_steps=args['meta_train_adapt_steps'], K_shot=100, args=args)

        # meta testing on target tasks
        adapt_steps_meta_test = 4  # 00  # 50
        acc, f1, auc = meta_test_adam(meta_test_task_id_list, dataname, args['batch_size'], 100, seed, maml, gnn,
                                        adapt_steps_meta_test, lossfn, args=args)
        
        print("Run {}/5 Acc: {:.4f} | Macro-F1: {:.4f} | Macro-AUROC: {:.4f}".format(run_time+1, acc, f1, auc))
        
        acc_list.append(acc)
        f1_list.append(f1) 
        auc_list.append(auc)
    
    for i, _ in enumerate(zip(test_task_id_list, acc_list)):
        print(_) 
        
    print("Final result, Acc: {:.4f} | Macro-F1: {:.4f} | Macro-AUROC: {:.4f}".format(np.mean(acc_list), np.mean(f1_list), np.mean(auc_list)))
