"""
Model training methods
"""
import torch.utils.data
import torch
import numpy as np
import copy
from utils import *
import time
from tqdm import tqdm
from sklearn import metrics
import pandas as pd

def train_model(model,folds_list, test_loader, local_test_loader, n_epochs, split_no , batch_size, log_start, logger, validate, use_extra_features):
    print("\n --------training model----------")
    optimizer = torch.optim.Adam(model.parameters())
    
    step_size = 300
    scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.003,
                         step_size=step_size, mode='triangular2',
                         gamma=0.99994)
    
    binary_cross_entropy = torch.nn.BCEWithLogitsLoss(reduction='mean').cuda()
    #l2_loss = torch.nn.MSELoss().cuda()
    #folds_list = [X_train_fold ,X_val_fold, Y_train_fold, Y_val_fold, train_feat_fold, valid_feat_fold]
    train = torch.utils.data.TensorDataset(folds_list[0],folds_list[4] , folds_list[2])
    valid = torch.utils.data.TensorDataset(folds_list[1], folds_list[5] , folds_list[3])
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size*2, shuffle=False)
    
    for epoch in range(n_epochs):
        start_time = time.time()
        model.train()
        avg_loss = 0.
        
        for (x_full_batch_train, feat_train ,y_batch_train) in tqdm(train_loader, disable = True):
            y_pred_train = model(x_full_batch_train, feat_train)
            scheduler.batch_step()
            loss = binary_cross_entropy(y_pred_train, y_batch_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
            
        model.eval()
        valid_preds = np.zeros((folds_list[1].size(0)))
        if validate == "True":
            avg_val_loss = 0.
            for i, (x_full_batch_val, feat_val ,y_batch_val) in enumerate(valid_loader):
                y_pred_val = model(x_full_batch_val, feat_val).detach()
                avg_val_loss += binary_cross_entropy(y_pred_val, y_batch_val).item() / len(valid_loader)
                valid_preds[i * batch_size*2:(i+1) * batch_size*2] = sigmoid(y_pred_val.cpu().numpy())[:, 0]
            
            search_result = threshold_search(folds_list[3].cpu().numpy(), valid_preds)
            val_f1, val_threshold = search_result['f1'], search_result['threshold']
            logger.loc[epoch + log_start, "fold"+str(split_no)] = val_f1

            elapsed_time = time.time() - start_time
            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t val_f1={:.4f} best_t={:.2f} \t time={:.2f}s'.format(
                epoch + 1, n_epochs, avg_loss, avg_val_loss, val_f1, val_threshold, elapsed_time))
        else:
            elapsed_time = time.time() - start_time
            print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
                epoch + 1, n_epochs, avg_loss, elapsed_time))
    
    model.eval()
    valid_preds = np.zeros((folds_list[1].size(0)))
    avg_val_loss = 0.
    for i, (x_full_batch_val, feat_val,y_batch_val) in enumerate(valid_loader):
        y_pred_val = model(x_full_batch_val, feat_val).detach()
        avg_val_loss += binary_cross_entropy(y_pred_val , y_batch_val).item() / len(valid_loader)
        valid_preds[i * batch_size*2:(i+1) * batch_size*2] = sigmoid(y_pred_val.cpu().numpy())[:, 0]
    print('Validation loss: ', avg_val_loss)

    test_preds = np.zeros((len(test_loader.dataset)))
    for i, (x_full_batch_test, feat_test) in enumerate(test_loader):
        y_pred_test = model(x_full_batch_test, feat_test).detach()
        test_preds[i * batch_size*2:(i+1) * batch_size*2] = sigmoid(y_pred_test.cpu().numpy())[:, 0]
    
    test_preds_local = np.zeros((len(local_test_loader.dataset)))
    if(local_test_loader is not None):   
	    for i, (x_full_batch_local, feat_local) in enumerate(local_test_loader):
	        y_pred_local = model(x_full_batch_local, feat_local).detach()
	        test_preds_local[i * batch_size*2:(i+1) * batch_size*2] = sigmoid(y_pred_local.cpu().numpy())[:, 0]

    return valid_preds, test_preds, test_preds_local

def trainer(splits, model_orig , train_X, train_Y, epochs, test_loader, local_test_loader ,train_preds, test_preds, local_test_preds, train_feat, batch_size,validate ,use_extra_features, logger):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_start = logger.shape[0]
    print("\n ---------splitting----------")
    for split_no, (train_idx, valid_idx) in enumerate(splits):
        
        X_train_fold = torch.tensor(train_X[train_idx], dtype = torch.long).to(device)
        Y_train_fold = torch.tensor(train_Y[train_idx, np.newaxis], dtype = torch.float32).to(device)
        X_val_fold = torch.tensor(train_X[valid_idx], dtype = torch.long).to(device)
        Y_val_fold =torch.tensor(train_Y[valid_idx, np.newaxis], dtype = torch.float32).to(device)
        
        
        train_feat_fold = torch.tensor(train_feat[train_idx], dtype = torch.float32).to(device)
        valid_feat_fold = torch.tensor(train_feat[valid_idx], dtype = torch.float32).to(device)
        
        folds_list = [X_train_fold ,X_val_fold, Y_train_fold, Y_val_fold, train_feat_fold, valid_feat_fold]
        
        model = copy.deepcopy(model_orig)
        model.to(device)
        print("Split {}/{}".format(split_no+1,len(splits)))
        pred_val_fold, pred_test_fold, pred_local_test_fold = train_model(model, folds_list,
                                                                          test_loader, local_test_loader,
                                                                           epochs ,split_no , batch_size, log_start, logger , validate, use_extra_features)
        
        train_preds[valid_idx] = pred_val_fold
        test_preds[:, split_no] = pred_test_fold
        local_test_preds[:, split_no] = pred_local_test_fold
    return train_preds, test_preds, local_test_preds

