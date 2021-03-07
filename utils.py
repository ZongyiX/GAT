import numpy as np
import scipy.sparse as sp
import torch

def encode_onehot(labels):

    # set()删除重复的词汇
    classes = set(labels)

    # np.identity(n):创建一个n*n的方阵(7*7)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}

    # 从labels获取字典的索引，然后通过classes_dict.get获取索引内容
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    # np.ones(edges.shape[0])：创造出总连接数那么多个1出来； edges[:,0],edges[:,1]:将产生的1放置到相应的位置
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)


    # adj的归一化
    adj = normalize(adj + sp.eye(adj.shape[0]))
    # adj = adj+sp.eye(adj.shape[0])

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    # 将相应的features和adj变成Tensor矩阵
    features = torch.FloatTensor(np.array(features.todense()))
    adj = torch.FloatTensor(np.array(adj.todense()))

    # np.where(labels) 表示的是labels中每一行1所在行的位置,其中labels(2708*7),正好对应classes中论文分类的位置
    labels = torch.LongTensor(np.where(labels)[1])

    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    # output.max(1)[1]：获取output中最大数据的位置,即预测的类型
    # type_as(labels):转成与labels一样的数据类型，但是这里数据已是该类型，该代码对精确度无影响
    preds = output.max(1)[1].type_as(labels)
    # 根据预测值preds和labels的是否一样返回 True or False
    # 加上了double(), True->1, False->0
    correct = preds.eq(labels).double()
    # 将correct内的所有数据加起来
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    # 换成COOrdinate format
    sparse_mx = sparse_mx.tocoo().astype(np.float32)

    # .row, .col: 分别把sparse_mx中的非零元的行，列坐标得出来
    # tensor([[   0,    8,   14,  ..., 1389, 2344, 2707],
    #         [   0,    0,    0,  ..., 2707, 2707, 2707]])
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))

    # 非零元的数据得出来
    values = torch.from_numpy(sparse_mx.data)
    # print(values)

    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)