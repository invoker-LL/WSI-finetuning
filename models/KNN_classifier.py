from sklearn.neighbors import KNeighborsClassifier
import os, pdb
from utils.utils import *
os.environ['CUDA_VISIBLE_DEVICES']='0'
from datasets.dataset_generic import Generic_MIL_Dataset

split_dir = "../splits/task_brca/"
data_root_dir = "../data_feat"
# Is_mean = False # for max-pool KNN
Is_mean = True  # for mean-pool

dataset = Generic_MIL_Dataset(
                            csv_path = '../dataset_csv/brca.csv',
                            data_dir= os.path.join(data_root_dir, 'brca_resnet50_20x_feat'),
                            shuffle = False,
                            seed = 2022,
                            print_info = True,
                            label_dict={'IDC':0, 'ILC':1},
                            patient_strat=False,
                            ignore=[])

f1_list,auc_list = [], []

folds=10

for i in range(folds):
    fold = i
    train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,
                csv_path='{}/splits_{}.csv'.format(split_dir, fold))

    train_loader = get_split_loader(train_dataset)
    test_loader = get_split_loader(test_dataset)
    X_train = []
    y_train = []
    train_pos = []
    for i,(datax, label) in enumerate(train_loader):
        position = datax[:,-2:]
        data = datax[:, :-2]
        if Is_mean:
            features = torch.mean(data, dim=0).view(1,-1)
        else:
            features = torch.max(data, dim=0)[0].view(1,-1)
        X_train.append(features.cpu().detach())
        y_train.append(label)
    # pdb.set_trace()
    X_train = torch.cat(X_train)
    y_train = torch.cat(y_train)

    X_test = []
    y_test = []
    train_pos = []
    for _, (datax, label) in enumerate(test_loader):
        # pdb.set_trace()
        position = datax[:, -2:]
        data = datax[:, :-2]
        # train_pos.append(position)
        if Is_mean:
            features = torch.mean(data, dim=0).view(1,-1)
        else:
            features = torch.max(data, dim=0)[0].view(1,-1)
        X_test.append(features.cpu().detach())
        y_test.append(label)
    X_test = torch.cat(X_test)
    y_test = torch.cat(y_test)
    # pdb.set_trace()
    print('data preparation finished')

    for n_neighb in range(9,10): # also we can use range 5~15 for avearged results
        knn = KNeighborsClassifier(n_neighbors=n_neighb,weights='distance')
        knn.fit(X_train,y_train)
        y_pred = knn.predict(X_test)
        y_prob = knn.predict_proba(X_test)
        from sklearn.metrics import f1_score, roc_auc_score
        f1 = f1_score(y_test,y_pred,average='macro')

        if y_prob.shape[1] == 2:
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr')

        f1_list.append(f1)
        auc_list.append(auc)
        print(auc)

f1_arr = np.asarray(f1_list)
auc_arr = np.asarray(auc_list)
print('f1 list:', f1_list)
print('auc list:', auc_list)
print('f1 mean std:', round(f1_arr.mean(),3),round(f1_arr.std(),3))
print('auc mean std:', round(auc_arr.mean(),3), round(auc_arr.std(),3))
