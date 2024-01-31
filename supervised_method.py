import os
import random 
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
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
from baseline import BaseGNN
import pickle as pk
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import torchmetrics
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
        task_pairs = task_pairs[:10]

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

def load_data_ori(dataname='Reddit', num_class=2):
    
    train_graph_list = []
    valid_graph_list = []
    test_graph_list = []
    
    for class_id in range(num_class):
        
        data_path1 = os.path.join('./dataset/{}/induced_graphs/task{}.meta.train.support'.format(dataname, class_id))
        data_path2 = os.path.join('./dataset/{}/induced_graphs/task{}.meta.train.query'.format(dataname, class_id))
        
        data_path3 = os.path.join('./dataset/{}/induced_graphs/task{}.meta.test.support'.format(dataname, class_id))
        data_path4 = os.path.join('./dataset/{}/induced_graphs/task{}.meta.test.query'.format(dataname, class_id))

        for data_path in [data_path1, data_path2]:
            with open(data_path, 'br') as t1s:
                t1s = pk.load(t1s)
                t1s = t1s['pos']
                
                for data in t1s:
                    x = data.x.detach()
                    edge_index = data.edge_index
                    edge_index = to_undirected(edge_index)
                    data_ = Data(x=x, edge_index=edge_index, y=class_id)
                    
                    train_graph_list.append(data_)
        
        with open(data_path3, 'br') as t1s:
            t1s = pk.load(t1s)
            t1s = t1s['pos']
            
            for data in t1s:
                x = data.x.detach()
                edge_index = data.edge_index
                edge_index = to_undirected(edge_index)
                data_ = Data(x=x, edge_index=edge_index, y=class_id)
                
                test_graph_list.append(data_)
        
        with open(data_path4, 'br') as t1s:
            t1s = pk.load(t1s)
            t1s = t1s['pos']
            
            for data in t1s:
                x = data.x.detach()
                edge_index = data.edge_index
                edge_index = to_undirected(edge_index)
                data_ = Data(x=x, edge_index=edge_index, y=class_id)
                
                test_graph_list.append(data_)
    
    return train_graph_list, valid_graph_list, test_graph_list


def load_data(dataname='Reddit', test_task_id_list=None):
    
    train_graph_list = []
    valid_graph_list = []
    test_graph_list = []
    
    for index_, class_id in enumerate(test_task_id_list):
        
        data_path1 = os.path.join('./dataset/{}/induced_graphs/task{}.meta.train.support'.format(dataname, class_id))
        data_path2 = os.path.join('./dataset/{}/induced_graphs/task{}.meta.train.query'.format(dataname, class_id))
        
        data_path3 = os.path.join('./dataset/{}/induced_graphs/task{}.meta.test.support'.format(dataname, class_id))
        data_path4 = os.path.join('./dataset/{}/induced_graphs/task{}.meta.test.query'.format(dataname, class_id))

        # for data_path in [data_path1, data_path2]:
        #     with open(data_path, 'br') as t1s:
        #         t1s = pk.load(t1s)
        #         t1s = t1s['pos']
                
        #         for data in t1s:
        #             x = data.x.detach()
        #             edge_index = data.edge_index
        #             edge_index = to_undirected(edge_index)
        #             data_ = Data(x=x, edge_index=edge_index, y=index_)
                    
        #             train_graph_list.append(data_)
        
        with open(data_path3, 'br') as t1s:
            t1s = pk.load(t1s)
            t1s = t1s['pos']
            
            for data in t1s:
                x = data.x.detach()
                edge_index = data.edge_index
                edge_index = to_undirected(edge_index)
                data_ = Data(x=x, edge_index=edge_index, y=index_)
                
                train_graph_list.append(data_)
        
        with open(data_path4, 'br') as t1s:
            t1s = pk.load(t1s)
            t1s = t1s['pos']
            
            for data in t1s:
                x = data.x.detach()
                edge_index = data.edge_index
                edge_index = to_undirected(edge_index)
                data_ = Data(x=x, edge_index=edge_index, y=index_)
                
                test_graph_list.append(data_)
    
    return train_graph_list, valid_graph_list, test_graph_list


def eval_(model, loader):
    model.eval()
    accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=2)
    macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=2, average="macro")
    mc_auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=2)


    accuracy = accuracy.cuda()
    macro_f1 = macro_f1.cuda()
    mc_auroc = mc_auroc.cuda()
    

    for batch_id, data in enumerate(loader):
        data.cuda()
        labels = data.y
        
        output = model(data)
        
        output = output.detach()
        pred = torch.argmax(output, dim=1)
    
        acc = accuracy(pred, labels)
        ma_f1 = macro_f1(pred, labels)
        ma_auc = mc_auroc(output, labels)
        
        
    acc = accuracy.compute()
    ma_f1 = macro_f1.compute()
    ma_auc = mc_auroc.compute()
    # print("Eval Acc: {:.4f} | Macro-F1: {:.4f} | Macro-AUROC: {:.4f}".format(acc.item(), ma_f1.item(), ma_auc.item()))
    accuracy.reset()
    macro_f1.reset()
    mc_auroc.reset()
    
    return acc.item(), ma_f1.item(), ma_auc.item()


if __name__ == '__main__':    
       
    dataset = 'Computers'  # 'CiteSeer'  # 'PubMed' 'Cora'  Computers 'Reddit' 'ogbn_arxiv'  'ogbn_products'  'dblp'

    gnn_type = 'TransformerConv'    # 'GCN'  # 'GAT'  # 'TransformerConv'
    dataset_num_class = {
            'Reddit': 41,
            'ogbn_arxiv': 40,
            'ogbn_products': 47,
            'dblp': 4,
            'Computers': 10,
            'CiteSeer': 6
        }
    
    
    num_class = dataset_num_class[dataset]
    input_dim = 100
    hid_dim = 100
    out_dim = 100
    
    batch_size = 10
    max_patience = 20
    
    meta_task_id_list = list(range(num_class))
    
    combinations_list = list(combinations(meta_task_id_list, 2))
    
    test_task_id_list = sample(combinations_list, 5)
    
    acc_list, f1_list, auc_list = [], [], []
    for run_time in range(5): 
        model = BaseGNN(input_dim, hid_dim=hid_dim, out_dim=out_dim, gcn_layer_num=2, pool=None, gnn_type=gnn_type, num_class=2)
        model.cuda()
        
        lossfn = nn.CrossEntropyLoss()
        lossfn.cuda()
        
        opt = optim.Adam(model.parameters(), lr=0.001)
        
        train_graph_list, valid_graph_list, test_graph_list = load_data(dataset, test_task_id_list[run_time])
        
        train_loader = DataLoader(train_graph_list, batch_size=batch_size, shuffle=True)
        # valid_loader = DataLoader(valid_graph_list, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_graph_list, batch_size=batch_size, shuffle=True)
        
        max_acc = 0.0
        best_epoch = 0
        patience =0
        for epoch in range(50):
            train_loss = 0.0
            model.train()
            for batch_id, data in enumerate(train_loader):
                opt.zero_grad()
                
                data.cuda()
                label = data.y
                
                pred = model(data)
                
                loss = lossfn(pred, label)
                loss.backward()
                opt.step()
                
                train_loss += loss.item()
            
            # valid_acc, valid_f1, valid_auc = eval_(model, valid_loader)
            # n_sample_train = len(train_loader.dataset)
            train_loss = train_loss / len(train_loader)
            # print("Epoch: {}, Train loss: {:.6f}, Valid acc: {:.6f}".format(epoch, train_loss, valid_acc))
            print("Epoch: {}, Train loss: {:.6f}".format(epoch, train_loss))
            
            # if valid_acc > max_acc:
            #     best_epoch = epoch
            #     max_acc = valid_acc
            #     torch.save(model.state_dict(), './pre_trained_gnn/{}.{}.pth'.format(dataset, gnn_type))
            #     print("Model {}.{} saved at epoch{}".format(dataset, gnn_type, epoch))
            #     patience = 0
            # else:
            #     patience += 1
                    
            # if patience == max_patience:
            #     break

        # model.load_state_dict(torch.load('./pre_trained_gnn/{}.{}.pth'.format(dataset, gnn_type)))
        model.eval()
        test_acc, test_f1, test_auc = eval_(model, test_loader)
        
        print("Run {}/5 Acc: {:.4f} | Macro-F1: {:.4f} | Macro-AUROC: {:.4f}".format(run_time+1, test_acc, test_f1, test_auc))
        
        acc_list.append(test_acc)
        f1_list.append(test_f1) 
        auc_list.append(test_auc)
    
    # for i, _ in enumerate(zip(test_task_id_list, acc_list)):
    #     print(_) 
        
    print("Final result, Acc: {:.4f} | Macro-F1: {:.4f} | Macro-AUROC: {:.4f}".format(np.mean(acc_list), np.mean(f1_list), np.mean(auc_list)))
