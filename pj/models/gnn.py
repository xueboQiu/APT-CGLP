import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention
from torch_geometric.nn.inits import glorot, zeros
from torch_scatter import scatter_add

class GraphSAGEConv(MessagePassing):
    def __init__(self, node_dim, edge_dim, emb_dim, aggr="mean", input_layer=False):
        super(GraphSAGEConv, self).__init__(aggr=aggr)

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_dim = edge_dim
        ### Mapping 0/1 edge features to embedding
        # self.edge_encoder = torch.nn.Linear(self.edge_dim, self.edge_dim)

        ### Mapping uniform input features to embedding.
        self.input_layer = input_layer

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), self.edge_dim)
        self_loop_attr[:, self.edge_dim-1] = 1  # attribute for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # edge_embeddings = self.edge_encoder(edge_attr)
        edge_embeddings = edge_attr

        # if self.input_layer:
        #     x = self.input_node_embeddings(x.to(torch.int64).view(-1, ))

        # x = self.linear(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        input_layer (bool): whether the GIN conv is applied to input layer or not. (Input node labels are uniform...)

    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self,node_dim, edge_dim, emb_dim,graph_width, aggr="add", input_layer=False):
        super(GINConv, self).__init__(aggr=aggr)
        # multi-layer perceptron
        # self.mlp = torch.nn.Sequential(torch.nn.Linear(2 * node_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
        #                                torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, node_dim))
        self.mlp = torch.nn.Sequential(torch.nn.Linear(2 * node_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Dropout(p = 0.2), torch.nn.Linear(2 * emb_dim, graph_width))
        self.edge_dim = edge_dim
        ### Mapping 0/1 edge features to embedding
        ### Mapping uniform input features to embedding.
        # self.input_layer = input_layer
        # if self.input_layer:
        #     self.input_node_embeddings = torch.nn.Embedding(node_dim+1, emb_dim)
        #     torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), self.edge_dim)
        self_loop_attr[:, self.edge_dim-1] = 1  # attribute for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device)
        # edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0).clone().detach().requires_grad_(True).float()
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # if self.input_layer:
        #     x = self.input_node_embeddings(x.to(torch.int64).view(-1,))

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return torch.cat([edge_attr, x_j], dim=1)

    def update(self, aggr_out):
        return self.mlp(aggr_out)
class GNN(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gat, graphsage, gcn

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536

    Output:
        node representations

    """

    def __init__(self,node_dim, edge_dim, num_layer, emb_dim, graph_width, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ###List of message-passing GNN convs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if layer == 0:
                input_layer = True
            else:
                input_layer = False

            # self.gnns.append(GraphSAGEConv(node_dim,edge_dim, emb_dim, input_layer=input_layer))
            self.gnns.append(GINConv(node_dim,edge_dim, emb_dim,graph_width, input_layer=input_layer))

    def forward(self, x, edge_index, edge_attr):
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            if layer == self.num_layer - 1:
                # remove relu from the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list[1:], dim=0), dim=0)[0]

        # mean pool
        # node_representation = torch.mean(node_representation, dim=0, keepdim=True).squeeze(0)
        return node_representation