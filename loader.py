import torch.utils.data
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np

def GetTestLoaders(test_X, local_test_X, test_feats, local_test_feats, batch_size ,use_extra_features):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_local_loader = None

    x_test_tensor = torch.tensor(test_X ,dtype = torch.long).to(device)
    test_feats_tensor = torch.tensor(test_feats, dtype = torch.float32).to(device)
    x_test_dataset = torch.utils.data.TensorDataset(x_test_tensor, test_feats_tensor)
    test_loader = torch.utils.data.DataLoader(x_test_dataset, batch_size = batch_size*2, shuffle = False)   

    if local_test_X is not None:
    	x_test_local_tensor = torch.tensor(local_test_X, dtype = torch.long).to(device)

    	x_test_local_tensor = torch.tensor(local_test_X, dtype = torch.long).to(device)
    	local_test_feats_tensor = torch.tensor(local_test_feats, dtype = torch.float32).to(device)
    	x_local_test_dataset = torch.utils.data.TensorDataset(x_test_local_tensor,local_test_feats_tensor)

    	test_local_loader = torch.utils.data.DataLoader(x_local_test_dataset, batch_size = batch_size*2, shuffle = False)

    return test_loader, test_local_loader


def GetData(test_X, local_test_X, train_X ,train_Y, test_feats, local_test_feats, n_splits, batch_size ,use_extra_features= False):
    test_loader, local_test_loader = GetTestLoaders(test_X, local_test_X, test_feats, local_test_feats, batch_size ,use_extra_features)
    print(train_X.shape, test_X.shape)
    if n_splits > 1:
        splits = list(StratifiedKFold(n_splits = n_splits, shuffle=True, random_state= 165).split(train_X, train_Y))
    else:
        valid_index = 30000
        splits = [[np.arange(start = 0, stop = train_X.shape[0]-valid_index), np.arange(start = train_X.shape[0] - valid_index, stop = train_X.shape[0])]]
    train_preds = np.zeros((train_X.shape[0], ))
    test_preds = np.zeros((test_X.shape[0], len(splits)))

    local_test_preds = None
    if(local_test_X is not None):
    	local_test_preds = np.zeros((local_test_X.shape[0], len(splits)))
    
    return test_loader, local_test_loader, splits, train_preds, test_preds, local_test_preds

