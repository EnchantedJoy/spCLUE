import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.functional import normalize


class NoiseLayer(nn.Module):

    def __init__(self, alpha=0.01, dropout=0.5) -> None:
        super().__init__()
        self.alpha = alpha
        self.drop = dropout

    def forward(self, x):
        gauss_x = x + self.alpha * torch.randn_like(x)
        return F.dropout(gauss_x, self.drop, training=self.training)


class AttentionBlock(nn.Module):

    def __init__(self, in_size, hidden_size=16):
        super().__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        # shape of z: [N, n_vision, d]
        w = self.project(z)  # attention weight of each vision, shape: [N, n_vision, 1]
        beta = torch.softmax(w, dim=1)  # [N, n_vision, 1]
        return (beta * z).sum(1), beta


class CCGCN(Module):

    def __init__(self, dims_list, n_clusters, graph_corr=0.4, dropout=0.5) -> None:
        super(CCGCN, self).__init__()
        """
        Args: 
            dims_list (list): dimensions of GCNs.
            n_clusters (int): number of clusters in the cluster-contrastive module.
            graph_corr (float): corruption probability of the graph. (Edge Dropout).
            dropout (float): probability of the dropout of networks.
        """
        self.input_dim = dims_list[0]
        self.hidden_dim = dims_list[1]
        self.z_dim = dims_list[2]
        self.dropout = dropout
        self.n_clusters = n_clusters
        self.graph_corr = graph_corr

        ### + encoders
        self.noiseLayer = NoiseLayer(dropout=self.dropout)
        self.Transform1 = TransForm_W(self.input_dim, self.hidden_dim, self.dropout)
        self.Transform2 = TransForm_W(self.hidden_dim, self.z_dim, self.dropout)

        self.act = nn.ELU()
        self.relu = nn.ReLU()
        self.attention = AttentionBlock(self.z_dim)

        ## + instance projection head
        self.projectInsHead = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim),
            nn.ReLU(),
            nn.Linear(self.z_dim, self.z_dim),
            nn.ReLU(),
        )

        ## + cluster projection head
        self.projectClsHead = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim),
            nn.ReLU(),
            nn.Linear(self.z_dim, self.n_clusters),
            nn.Softmax(dim=1),
        )

    def encoder(self, data, adj):
        feature = self.noiseLayer(data)
        adj1 = torch.sparse_coo_tensor(
            adj._indices(),
            F.dropout(adj._values(), p=self.graph_corr, training=self.training),
            size=adj.size(),
        )
        feature = self.act(torch.spmm(adj1, self.Transform1(feature)))
        adj2 = torch.sparse_coo_tensor(
            adj._indices(),
            F.dropout(adj._values(), p=self.graph_corr, training=self.training),
            size=adj.size(),
        )
        feature = self.act(torch.spmm(adj2, self.Transform2(feature)))
        return feature

    def getCluster(self, embed):
        labels = self.projectClsHead(embed)
        return torch.argmax(labels, dim=1)

    def forward(self, data, adj1, adj2, batch_onehot=None):
        """
        Args:
            data (torch.FloatTensor): pca input of gene expression data.
            adj1 (torch.sparse_coo_tensor): normalized spatial graph.
            adj2 (torch.sparse_coo_tensor): normalized expr graph.
        """
        feature1 = self.encoder(data, adj1)
        feature2 = self.encoder(data, adj2)

        # + L2 normalization
        z1_norm = normalize(feature1, p=2, dim=1)
        z2_norm = normalize(feature2, p=2, dim=1)

        # + instance projection
        h1_norm = normalize(self.projectInsHead(z1_norm), p=2, dim=1)
        h2_norm = normalize(self.projectInsHead(z2_norm), p=2, dim=1)

        # + cluster projection
        label1 = self.projectClsHead(z1_norm)
        label2 = self.projectClsHead(z2_norm)

        # + attention fuse
        z = torch.stack([z1_norm, z2_norm], dim=1)
        z, _ = self.attention(z)

        x_Rec = self.relu(z @ self.Transform2.W.data.T) @ self.Transform1.W.data.T

        return h1_norm, h2_norm, z1_norm, z2_norm, z, label1, label2, x_Rec


class CCGCNs(Module):

    def __init__(
        self,
        dims_list,
        n_clusters,
        n_batches,
        graph_corr=0.4,
        dropout=0.5,
        device="cuda:0"
    ) -> None:
        super(CCGCNs, self).__init__()
        self.input_dim = dims_list[0]
        self.hidden_dim = dims_list[1]
        self.z_dim = dims_list[2]
        self.dropout = dropout
        self.n_clusters = n_clusters
        self.n_batches = n_batches
        self.graph_corr = graph_corr

        ### + encoders
        self.noiseLayer = NoiseLayer()
        self.Transform1 = TransForm_W(self.input_dim, self.hidden_dim, self.dropout)
        self.Transform2 = TransForm_W(self.hidden_dim, self.z_dim, self.dropout)

        self.batchPortion = torch.eye(self.n_batches).to(device)
        self.weightBatch = 0.01

        if self.weightBatch != 0:
            self.batchEmbed = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(self.n_batches, self.z_dim))
            )
            self.batchPCA = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(self.n_batches, self.input_dim))
            )
        else:
            self.batchEmbed = torch.ones(self.n_batches, self.z_dim).to(device)
            self.batchPCA = torch.ones(self.n_batches, self.input_dim).to(device)

        self.act = nn.ELU()
        self.relu = nn.ReLU()
        self.attention = AttentionBlock(self.z_dim)

        ## + instance projection head
        self.projectInsHead = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim),
            nn.ReLU(),
            nn.Linear(self.z_dim, self.z_dim),
            nn.ReLU(),
        )
        ## + cluster projection head
        self.projectClsHead = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim),
            nn.ReLU(),
            nn.Linear(self.z_dim, self.n_clusters),
            nn.Softmax(dim=1),
        )

    def encoder(self, data, adj):
        feature = self.noiseLayer(data)
        adj1 = torch.sparse_coo_tensor(
            adj._indices(),
            F.dropout(adj._values(), p=self.graph_corr, training=self.training),
            size=adj.size(),
        )
        feature = self.act(torch.spmm(adj1, self.Transform1(feature)))
        adj2 = torch.sparse_coo_tensor(
            adj._indices(),
            F.dropout(adj._values(), p=self.graph_corr, training=self.training),
            size=adj.size(),
        )
        feature = self.act(torch.spmm(adj2, self.Transform2(feature)))
        return feature

    def getCluster(self, embed):
        labels = self.projectClsHead(embed)
        return torch.argmax(labels, dim=1)

    def forward(self, data, adj1, adj2, batch_list=None):
        """
        Args:
            data (torch.FloatTensor): pca input of the gene expression data.
            adj1 (torch.sparse_coo_tensor): normalized spatial graph.
            adj2 (torch.sparse_coo_tensor): normalized expr graph.
            batch_list (): list of batch ID.
        """
        batch_noise = self.batchPortion[batch_list] @ self.batchPCA
        data = data - self.weightBatch * batch_noise 
        feature1 = self.encoder(data, adj1)
        feature2 = self.encoder(data, adj2)

        # + L2 normalization
        z1_norm = normalize(feature1, p=2, dim=1)
        z2_norm = normalize(feature2, p=2, dim=1)

        # + instance projection
        h1_norm = normalize(self.projectInsHead(z1_norm), p=2, dim=1)
        h2_norm = normalize(self.projectInsHead(z2_norm), p=2, dim=1)

        # + cluster projection
        label1 = self.projectClsHead(z1_norm)
        label2 = self.projectClsHead(z2_norm)

        # + attention fuse
        z = torch.stack([z1_norm, z2_norm], dim=1)
        z, _ = self.attention(z)

        # + add batch embedding
        norm_batch_embed = normalize(self.batchEmbed, p=2, dim=1)
        z_dec = z + self.weightBatch * self.batchPortion[batch_list] @ norm_batch_embed

        x_Rec = self.relu(z_dec @ self.Transform2.W.data.T) @ self.Transform1.W.data.T

        return h1_norm, h2_norm, z1_norm, z2_norm, z, x_Rec, label1, label2


class InnerProductDec(nn.Module):

    def __init__(self, dropout=0.2) -> None:
        super().__init__()
        self.dropout = dropout

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj_rec = z @ z.T
        return adj_rec


class IdentityMap(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x


class TransForm_W(nn.Module):

    def __init__(self, input_dim, out_dim, dropout=0.5, act=None) -> None:
        super().__init__()
        self.dropout = dropout
        self.W = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(input_dim, out_dim))
        )  ## = initialize weight of transform layer

    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x @ self.W
