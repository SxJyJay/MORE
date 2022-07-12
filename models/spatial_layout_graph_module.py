import math
from operator import index
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as GUtils

import numpy as np

from torch_geometric.data import Data as GData
from torch_geometric.data import DataLoader as GDataLoader
from torch_geometric.nn import GCNConv, TransformerConv
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.typing import Adj, Size

from scipy.sparse import coo_matrix

import os
import sys
import ipdb
sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from utils.box_util import box3d_iou_batch_tensor
from lib.config import CONF

class EdgeConv(MessagePassing):
    def __init__(self, in_size, out_size, aggregation="add"):
        super().__init__(aggr=aggregation)
        self.in_size = in_size
        self.out_size = out_size

        self.map_edge = nn.Sequential(
            nn.Linear(2 * in_size, out_size),
            nn.ReLU(),
            nn.Linear(out_size, out_size)
        )
        # self.map_node = nn.Sequential(
        #     nn.Linear(out_size, out_size),
        #     nn.ReLU()
        # )

    def forward(self, x, edge_index):
        # x has shape [N, in_size]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            adj (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
                :obj:`torch_sparse.SparseTensor` that defines the underlying
                graph connectivity/message passing flow.
                :obj:`edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
                If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its
                shape must be defined as :obj:`[2, num_messages]`, where
                messages from nodes in :obj:`edge_index[0]` are sent to
                nodes in :obj:`edge_index[1]`
                (in case :obj:`flow="source_to_target"`).
                If :obj:`edge_index` is of type
                :obj:`torch_sparse.SparseTensor`, its sparse indices
                :obj:`(row, col)` should relate to :obj:`row = edge_index[1]`
                and :obj:`col = edge_index[0]`.
                The major difference between both formats is that we need to
                input the *transposed* sparse adjacency matrix into
                :func:`propagate`.
            size (tuple, optional): The size :obj:`(N, M)` of the assignment
                matrix in case :obj:`edge_index` is a :obj:`LongTensor`.
                If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case :obj:`edge_index` is a
                :obj:`torch_sparse.SparseTensor`. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        size = self.__check_input__(edge_index, size)

        coll_dict = self.__collect__(self.__user_args__, edge_index, size,
                                        kwargs)

        msg_kwargs = self.inspector.distribute('message', coll_dict)
        message = self.message(**msg_kwargs)

        # For `GNNExplainer`, we require a separate message and aggregate
        # procedure since this allows us to inject the `edge_mask` into the
        # message passing computation scheme.
        if self.__explain__:
            edge_mask = self.__edge_mask__.sigmoid()
            # Some ops add self-loops to `edge_index`. We need to do the
            # same for `edge_mask` (but do not train those).
            if message.size(self.node_dim) != edge_mask.size(0):
                loop = edge_mask.new_ones(size[0])
                edge_mask = torch.cat([edge_mask, loop], dim=0)
            assert message.size(self.node_dim) == edge_mask.size(0)
            message = message * edge_mask.view([-1] + [1] * (message.dim() - 1))

        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        out = self.aggregate(message, **aggr_kwargs)

        update_kwargs = self.inspector.distribute('update', coll_dict)
        # import ipdb
        # ipdb.set_trace()
        
        return self.update(out, **update_kwargs), message

    def message(self, x_i, x_j):
        # x_i has shape [E, in_size]
        # x_j has shape [E, in_size]

        edge = torch.cat([x_i, x_j - x_i], dim=1)  # edge has shape [E, 2 * in_size]
        # edge = torch.cat([x_i, x_j], dim=1)  # edge has shape [E, 2 * in_size]
        
        return self.map_edge(edge)

    def update(self, x_i):
        # x has shape [N, out_size]

        # return self.map_node(x_i)
        return x_i

class SpatialLayoutConv(MessagePassing):
    def __init__(self, in_size, out_size, heads=1, concat=True, edge_dim=None, bias=True, root_weight=True, aggregation='add',\
        spatial_vocab_we=None, room_spatial_vocab_we=None):
        super().__init__(node_dim=0, aggr=aggregation)
        self.in_channels = in_size
        self.out_channels = out_size
        self.heads = heads
        self.root_weight = root_weight
        self.concat = concat
        self.edge_dim = edge_dim
        self.spatial_vocab_we = spatial_vocab_we # (7, 300) {0:'top', 1:'bottom', 2:'left', 3:'right', 4:'front', 5:'behind', 6:['next to','near','besides']}
        self.room_spatial_vocab_we = room_spatial_vocab_we # (6, 300) {0:'corner', 1:'middle', 2:'left', 3:'right', 4:'front', 5:'back'}

        self.lin_key = Linear(in_size, heads * out_size)
        self.lin_query = Linear(in_size, heads * out_size)
        self.lin_value = Linear(in_size, heads * out_size)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_size, bias=False)
            self.map_normal = Linear(3, in_size)
            self.map_xy_relative_coord = Linear(2, in_size)
            self.map_back_edge_loc = Linear(in_size*3, in_size)
            self.map_back_edge_obj = Linear(in_size*2, in_size)
            # for cct + conv loc & obj fusion
            # self.edge_loc_obj_fusion = nn.Sequential(
            #     Linear(in_size*2, in_size),
            #     nn.LeakyReLU()
            # )
            self.hv_words_fusion = Linear(300*2, 300)

            self.horizontal_word_cls = Linear(in_size, 5)
        else:
            self.lin_edge = None

        if concat:
            self.lin_skip = Linear(in_size, heads * out_size, bias=bias)
        else:
            self.lin_skip = Linear(in_size, out_size, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()

    def forward(self, x, edge_index, edge_attr, obj_normals):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, obj_normals=obj_normals)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x)
            out = out + x_r

        return out

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            adj (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
                :obj:`torch_sparse.SparseTensor` that defines the underlying
                graph connectivity/message passing flow.
                :obj:`edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
                If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its
                shape must be defined as :obj:`[2, num_messages]`, where
                messages from nodes in :obj:`edge_index[0]` are sent to
                nodes in :obj:`edge_index[1]`
                (in case :obj:`flow="source_to_target"`).
                If :obj:`edge_index` is of type
                :obj:`torch_sparse.SparseTensor`, its sparse indices
                :obj:`(row, col)` should relate to :obj:`row = edge_index[1]`
                and :obj:`col = edge_index[0]`.
                The major difference between both formats is that we need to
                input the *transposed* sparse adjacency matrix into
                :func:`propagate`.
            size (tuple, optional): The size :obj:`(N, M)` of the assignment
                matrix in case :obj:`edge_index` is a :obj:`LongTensor`.
                If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case :obj:`edge_index` is a
                :obj:`torch_sparse.SparseTensor`. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        size = self.__check_input__(edge_index, size)

        coll_dict = self.__collect__(self.__user_args__, edge_index, size,
                                        kwargs)

        msg_kwargs = self.inspector.distribute('message', coll_dict)
        message = self.message(**msg_kwargs)

        # For `GNNExplainer`, we require a separate message and aggregate
        # procedure since this allows us to inject the `edge_mask` into the
        # message passing computation scheme.
        if self.__explain__:
            edge_mask = self.__edge_mask__.sigmoid()
            # Some ops add self-loops to `edge_index`. We need to do the
            # same for `edge_mask` (but do not train those).
            if message.size(self.node_dim) != edge_mask.size(0):
                loop = edge_mask.new_ones(size[0])
                edge_mask = torch.cat([edge_mask, loop], dim=0)
            assert message.size(self.node_dim) == edge_mask.size(0)
            message = message * edge_mask.view([-1] + [1] * (message.dim() - 1))

        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        # ipdb.set_trace()
        out = self.aggregate(message, **aggr_kwargs)

        update_kwargs = self.inspector.distribute('update', coll_dict)
        # import ipdb
        # ipdb.set_trace()
        
        return self.update(out, **update_kwargs)

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i, obj_normals_i, obj_normals_j):
        # x_i, x_j : [E, C]
        # edge_attr : [E, 4] --> [:,0:2] for top & down; [:, 2:4] for xy_relative_coord
        # ipdb.set_trace()
        query = self.lin_query(x_i).view(-1, self.heads, self.out_channels)
        key = self.lin_key(x_j).view(-1, self.heads, self.out_channels)

        # Add code for generating spatial word embeddings (i.e. edge_word)
        if self.lin_edge is not None:

            top_edges = edge_attr[:, 0] > 0
            bottom_edges = edge_attr[:, 1] < 0
            
            # spatial word for vertical relations i.e., top, down
            top_edges = top_edges.unsqueeze(1) * self.spatial_vocab_we[0].unsqueeze(0) # (E, 300) 
            bottom_edges = bottom_edges.unsqueeze(1) * self.spatial_vocab_we[1].unsqueeze(0) # (E, 300)
            vertical_edges = top_edges + bottom_edges
            
            # construct edge spatial features 
            xy_relative_coord = edge_attr[:, 2:4] # (E, 2)
            mapped_obj_normals_i = self.map_normal(obj_normals_i) # (E, 128)
            mapped_obj_normals_j = self.map_normal(obj_normals_j) # (E, 128)
            mapped_xy_relative_coord = self.map_xy_relative_coord(xy_relative_coord) # (E, 128)
            edge_loc_feat = torch.cat([mapped_xy_relative_coord, mapped_obj_normals_i, mapped_obj_normals_j], dim=1)
            mapped_edge_loc_feat = self.map_back_edge_loc(edge_loc_feat) # (E, 128)

            # construct edge object features
            edge_obj_feat = torch.cat([x_i, x_j - x_i], dim=1)
            mapped_edge_obj_feat = self.map_back_edge_obj(edge_obj_feat)

            # classification to 5 types of horizontal spatial words
            # tanh(W1x1+W2x2) fusion
            horizontal_desicion_feat = F.tanh(mapped_edge_obj_feat + mapped_edge_loc_feat)
            # concat+conv fusion
            # horizontal_desicion_feat = self.edge_loc_obj_fusion(torch.cat([mapped_edge_obj_feat, mapped_edge_loc_feat], dim=-1))
            horizontal_words_logit = self.horizontal_word_cls(horizontal_desicion_feat)
            # Soft version
            horizontal_words_prob = F.softmax(horizontal_words_logit, dim=1) # (E, 5)
            # hard version
            # horizontal_words_prob = F.gumbel_softmax(horizontal_words_logit, tau=1, hard=True)
            horizontal_edges = torch.matmul(horizontal_words_prob, self.spatial_vocab_we[2:7, :]) # (E, 300)

            hv_edges = torch.cat([vertical_edges, horizontal_edges], dim=-1)
            edge_word = self.hv_words_fusion(hv_edges)

            edge_spatial = self.lin_edge(edge_word).view(-1, self.heads, self.out_channels)
            key = key + edge_spatial

        alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)

        out = self.lin_value(x_j).view(-1, self.heads, self.out_channels)
        if self.lin_edge is not None:
            out = out + edge_spatial

        out = out * alpha.view(-1, self.heads, 1)
        return out

    def update(self, input):
        return input




class SpatialLayoutGraphModule(nn.Module):
    def __init__(self, in_size, out_size, num_layers, num_proposals, feat_size, num_locals, glove, 
        query_mode="corner", graph_mode="spatial_layout_conv", return_edge=False, graph_aggr="add", 
        return_orientation=False, num_bins=6, return_distance=False, use_color=True, use_multiview=False):
        super().__init__()

        # construct a look up talble: LabelId --> VocabId through LabelCls word
        self.Label_CLS2ID = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
            'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
            'refrigerator':12, 'shower curtain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'others':17}  
        self.Label_ID2CLS = {self.Label_CLS2ID[cls_name]:cls_name for cls_name in self.Label_CLS2ID}

        self.in_size = in_size
        self.out_size = out_size

        self.num_proposals = num_proposals # add the global node
        self.feat_size = feat_size
        
        self.num_locals = num_locals
        self.query_mode = query_mode

        # GLOVE embedding lookup table
        self.glove = glove
        self.glove_embedding_size = 300

        self.spatial_vocab = {0:'top', 1:'bottom', 2:'left', 3:'right', 4:'front', 5:'behind', 6:['next to','near','besides']}
        self.room_spatial_vocab = {0:'corner', 1:'middle', 2:'left', 3:'right', 4:'front', 5:'back'}
        self.spatial_vocab_we = np.zeros((7, 300)) # (7, 300)
        self.room_spatial_vocab_we = np.zeros((6, 300)) # (6, 300)

        for ii in range(7):
            spatial_word = self.spatial_vocab[ii]
            if isinstance(spatial_word, list):
                self.spatial_vocab_we[ii] = ((self.glove['next']+self.glove['to'])/2. + self.glove['near'] + self.glove['besides'])/3.
                # self.spatial_vocab_we[ii] = self.glove['besides']
            else:
                self.spatial_vocab_we[ii] = self.glove[spatial_word]

        for ii in range(6):
            room_spatial_word = self.room_spatial_vocab[ii]
            self.room_spatial_vocab_we[ii] = self.glove[room_spatial_word]
            
        
        self.spatial_vocab_we = torch.from_numpy(self.spatial_vocab_we).to(torch.float32).cuda()
        self.room_spatial_vocab_we = torch.from_numpy(self.room_spatial_vocab_we).to(torch.float32).cuda()

        # obj proposal & class we fusion module
        self.feat_we_fusion_module = nn.Sequential(
            nn.Linear(self.feat_size+self.glove_embedding_size, self.feat_size),
            nn.LeakyReLU()
        )
        # self.obj_proposal_trans = nn.Linear(self.feat_size, self.feat_size)
        # self.we_trans = nn.Linear(self.glove_embedding_size, self.feat_size)
        # self.node_feat_trans= nn.Linear(self.feat_size, self.feat_size)

        # import ipdb
        # ipdb.set_trace()

        # graph layers
        self.graph_mode = graph_mode
        self.gc_layers = nn.ModuleList()
        for _ in range(num_layers):
            if graph_mode == "graph_conv":
                self.gc_layers.append(GCNConv(in_size, out_size))
            elif graph_mode == "transformer_conv":
                self.gc_layers.append(TransformerConv(in_size, out_size))
            elif graph_mode == "spatial_layout_conv":
                self.gc_layers.append(SpatialLayoutConv(in_size, out_size, edge_dim=300,\
                    spatial_vocab_we=self.spatial_vocab_we, room_spatial_vocab_we=self.room_spatial_vocab_we))
            elif graph_mode == "repitition_transformer_conv":
                self.gc_layers.append(SpatialLayoutConv(in_size, out_size))
            elif graph_mode == "edge_conv":
                self.gc_layers.append(EdgeConv(in_size, out_size, graph_aggr))
            else:
                raise ValueError("invalid graph mode, choices: [\"graph_conv\", \"edge_conv\"]")

        # graph edges
        self.return_edge = return_edge
        self.return_orientation = return_orientation
        self.return_distance = return_distance
        self.num_bins = num_bins
        self.use_color = use_color
        self.use_multiview = use_multiview

        # output final edges
        if self.return_orientation: 
            assert self.graph_mode == "edge_conv"
            self.edge_layer = EdgeConv(in_size, out_size, graph_aggr)
            self.edge_predict = nn.Linear(out_size, num_bins + 1)

    def _nn_distance(self, pc1, pc2):
        """
        Input:
            pc1: (B,N,C) torch tensor
            pc2: (B,M,C) torch tensor

        Output:
            dist1: (B,N) torch float32 tensor
            idx1: (B,N) torch int64 tensor
            dist2: (B,M) torch float32 tensor
            idx2: (B,M) torch int64 tensor
        """

        N = pc1.shape[1]
        M = pc2.shape[1]
        pc1_expand_tile = pc1.unsqueeze(2).repeat(1,1,M,1)
        pc2_expand_tile = pc2.unsqueeze(1).repeat(1,N,1,1)
        pc_diff = pc1_expand_tile - pc2_expand_tile
        pc_dist = torch.sqrt(torch.sum(pc_diff**2, dim=-1) + 1e-8) # (B,N,M)

        return pc_dist, pc_diff

    def _nn_corner_z_axis_diff(self, pc1, pc2):
        """
        Input:
            pc1: (B,1,8,3) torch tensor
            pc2: (B,num_proposals,8,3) torch tensor

        Output:
            z_axis_diff: (B,num_proposals,8,8) torch float32 tensor
        """
        batch_size, num_proposals, _, _ = pc2.shape # num_proposals
        # take z coordinates of bbox corners for calculation
        pc1 = pc1[..., 2] # B,1,8
        pc2 = pc2[..., 2] # B,num_propsoals,8
        pc1_expand_tile = pc1.unsqueeze(-1).repeat(1,num_proposals,1,8) # B,num_propsoals,8,8
        pc2_expand_tile = pc2.unsqueeze(2).repeat(1,1,8,1) # B,num_propsoals,8,8
        
        pc_diff = pc1_expand_tile - pc2_expand_tile
        pc_z_axis_diff = pc_diff.view(batch_size, num_proposals, 64)

        return pc_z_axis_diff
    

    def _get_bbox_centers(self, corners):
        coord_min = torch.min(corners, dim=2)[0] # batch_size, num_proposals, 3
        coord_max = torch.max(corners, dim=2)[0] # batch_size, num_proposals, 3

        return (coord_min + coord_max) / 2

    def _query_locals(self, data_dict, target_ids, object_masks, include_self=True, overlay_threshold=CONF.TRAIN.OVERLAID_THRESHOLD):
        corners = data_dict["bbox_corner"] # batch_size, num_proposals, 8, 3
        centers = self._get_bbox_centers(corners) # batch_size, num_proposals, 3
        batch_size, _, _ = centers.shape

        # decode target box info
        target_centers = torch.gather(centers, 1, target_ids.view(-1, 1, 1).repeat(1, 1, 3)) # batch_size, 1, 3
        target_corners = torch.gather(corners, 1, target_ids.view(-1, 1, 1, 1).repeat(1, 1, 8, 3)) # batch_size, 1, 8, 3

        # get the distance
        if self.query_mode == "center":
            pc_dist, pc_diff = self._nn_distance(target_centers, centers) # batch_size, num_proposals
            pc_dist = pc_dist.squeeze(1) # batch_size, num_proposals
            pc_diff = pc_diff.squeeze(1) # batch_size, num_propsoals, 3
        elif self.query_mode == "corner":
            pc_dist, _ = self._nn_distance(target_corners.squeeze(1), centers) # batch_size, 8, num_proposals
            _, pc_diff = self._nn_distance(target_centers, centers)
            pc_diff = pc_diff.squeeze(1)
            pc_dist, _ = torch.min(pc_dist, dim=1) # batch_size, num_proposals
        else:
            raise ValueError("invalid distance mode, choice: [\"center\", \"corner\"]")

        # get the corner diff
        pc_corner_z_axis_diff = self._nn_corner_z_axis_diff(target_corners, corners) # batch_size, num_propsoals, 64
        pc_corner_z_axis_diff_max = pc_corner_z_axis_diff.max(-1)[0]
        pc_corner_z_axis_diff_min = pc_corner_z_axis_diff.min(-1)[0] # batch_size, num_propsals
        # pc_corner_z_axis_diff_max.masked_fill_(object_masks == 0, 0)
        # pc_corner_z_axis_diff_min.masked_fill_(object_masks == 0, 0)

        # mask out invalid objects
        pc_dist.masked_fill_(object_masks == 0, float('1e30')) # distance to invalid objects: infinity

        # exclude overlaid boxes
        tar2neigbor_iou = box3d_iou_batch_tensor(
            target_corners.repeat(1, self.num_proposals, 1, 1).view(-1, 8, 3), corners.view(-1, 8, 3)).view(batch_size, self.num_proposals) # batch_size, num_proposals
        overlaid_masks = tar2neigbor_iou >= overlay_threshold
        pc_dist.masked_fill_(overlaid_masks, float('1e30')) # distance to overlaid objects: infinity
        # pc_corner_z_axis_diff_max.masked_fill_(overlaid_masks, 0)
        # pc_corner_z_axis_diff_min.masked_fill_(overlaid_masks, 0)

        # include the target objects themselves
        self_dist = 0 if include_self else float('1e30')
        self_masks = torch.zeros(batch_size, self.num_proposals).cuda()
        self_masks.scatter_(1, target_ids.view(-1, 1), 1)
        pc_dist.masked_fill_(self_masks == 1, self_dist) # distance to themselves: 0 or infinity

        # get the top-k object ids
        _, topk_ids = torch.topk(pc_dist, self.num_locals, largest=False, dim=1) # batch_size, num_locals

        # construct masks for the local context
        local_masks = torch.zeros(batch_size, self.num_proposals).cuda()
        local_masks.scatter_(1, topk_ids, 1)

        pc_corner_z_axis_diff_max.masked_fill_(local_masks == 0, 0)
        pc_corner_z_axis_diff_min.masked_fill_(local_masks == 0, 0)
        expanded_local_masks = local_masks.unsqueeze(-1).repeat(1, 1, 3)
        pc_diff.masked_fill_(expanded_local_masks == 0, 0)

        return local_masks, pc_corner_z_axis_diff_max, pc_corner_z_axis_diff_min, pc_diff

    def _create_adjacent_mat(self, data_dict, object_masks):
        batch_size, num_objects = object_masks.shape
        adjacent_mat = torch.zeros(batch_size, num_objects, num_objects).cuda()
        z_axis_max = torch.zeros(batch_size, num_objects, num_objects).cuda()
        z_axis_min = torch.zeros(batch_size, num_objects, num_objects).cuda()
        relative_coord = torch.zeros(batch_size, num_objects, num_objects, 3).cuda()

        for obj_id in range(num_objects):
            target_ids = torch.LongTensor([obj_id for _ in range(batch_size)]).cuda()
            adjacent_entry, z_axis_max_entry, z_axis_min_entry, relative_coord_entry = self._query_locals(data_dict, target_ids, object_masks, include_self=False) # batch_size, num_objects
            adjacent_mat[:, obj_id] = adjacent_entry
            z_axis_max[:, obj_id] = z_axis_max_entry
            z_axis_min[:, obj_id] = z_axis_min_entry
            relative_coord[:, obj_id] = relative_coord_entry

        return adjacent_mat, z_axis_max, z_axis_min, relative_coord

    def _feed(self, graph):
        feat, edge, edge_attr, obj_normals = graph.x, graph.edge_index, graph.edge_attr, graph.obj_normals

        for layer in self.gc_layers:
            if self.graph_mode != "edge_conv":
                feat = layer(feat, edge, edge_attr, obj_normals)
                message = None
                
            elif self.graph_mode == "edge_conv":
                feat, message = layer(feat, edge)

        return feat, message

    def _obj_proposal_we_fusion(self, obj_proposal_feat, obj_cls_we):
        # simple concat-convolution fusion
        # if use more complicated fusion module, formulate it to an individual nn.Module class
        multimodal_feat = torch.cat([obj_proposal_feat, obj_cls_we], dim=-1)
        fused_feat = self.feat_we_fusion_module(multimodal_feat)
        # mapped_obj_proposal = self.obj_proposal_trans(obj_proposal_feat)
        # mapped_we_feat = self.we_trans(obj_cls_we)
        # fused_feat = self.node_feat_trans(torch.tanh(torch.add(mapped_obj_proposal, mapped_we_feat)))
        return fused_feat



    def forward(self, data_dict):
        obj_feats = data_dict["bbox_feature"] # batch_size, num_proposals, feat_size
        object_masks = data_dict["bbox_mask"] # batch_size, num_proposals
        object_semantics = data_dict["bbox_sems"] # batch_size, num_proposals
        object_centers = data_dict["center"] # batch_size, num_proposals, 3
        proposalId_2_seedId = data_dict["aggregated_vote_inds"] # batch_size, num_proposals
        seedId_2_pcId = data_dict["seed_inds"] # batch_size, 1024
        batch_size = obj_feats.shape[0]

        # get normals for each proposal
        proposalId_2_pcId = seedId_2_pcId.gather(1, proposalId_2_seedId.long()) # batch_size, num_proposals
        proposalId_2_pcId = proposalId_2_pcId.unsqueeze(-1).repeat(1,1,3) # batch_size, num_proposals, 3
        if self.use_color:
            object_normals = data_dict["point_clouds"][:,:,6:9].gather(1, proposalId_2_pcId.long()) # batch_size, num_proposals, 3
        elif self.use_multiview:
            object_normals = data_dict["point_clouds"][:,:,3:6].gather(1, proposalId_2_pcId.long()) # batch_size, num_proposals, 3

        global_feat = torch.mean(obj_feats, dim=1, keepdim=True) # batch_size, 1, feat_size
        global_centers = torch.mean(data_dict["point_clouds"][:,:,:3], dim=1, keepdim=True) # batch_size, 1, 3
        global_normal = torch.Tensor([0,0,1.0]).cuda()
        global_normal = global_normal.view(1, 3) # 1, 3
        self_relative_coord = torch.zeros(1, 1, 2).cuda()

        global_scene_cls_we = torch.Tensor(self.glove["room"]).unsqueeze_(0).to(torch.float32).cuda() # 1, 300
        
        _, global_relative_coord = self._nn_distance(global_centers, object_centers)
        global_relative_coord = global_relative_coord.squeeze(1) # batch_size, num_proposals, 3

        batch_size, num_objects, _ = obj_feats.shape
        adjacent_mat, z_axis_max, z_axis_min, relative_coord = self._create_adjacent_mat(data_dict, object_masks) # batch_size, num_proposals, num_proposals

        top_mat = z_axis_min > 0 # B, num_proposals, num_proposals
        bottom_mat = z_axis_max < 0 # B, num_proposals, num_proposals
        xy_relative_coord = relative_coord[:,:,:,:2] # B, num_proposals, num_proposals, 2
        # global_mat_row = torch.zeros(batch_size, 1, num_objects).fill_(1)
        # global_mat_col = torch.zeros(batch_size, num_objects+1, 1).fill_(1)
        # global_added_adjacent_mat = torch.cat([adjacent_mat, global_mat_row], dim=1)
        # global_added_adjacent_mat = torch.cat([global_added_adjacent_mat, global_mat_col], dim=-1) # batch_size, num_proposals+1, num_proposals+1

        new_obj_feats = torch.zeros(batch_size, num_objects, self.feat_size).cuda()
        edge_indices = torch.zeros(batch_size, 2, num_objects * self.num_locals).cuda()
        edge_feats = torch.zeros(batch_size, num_objects, self.num_locals, self.out_size).cuda()
        edge_preds = torch.zeros(batch_size, num_objects * self.num_locals, self.num_bins+1).cuda()
        num_sources = torch.zeros(batch_size).long().cuda()
        num_targets = torch.zeros(batch_size).long().cuda()
        for batch_id in range(batch_size):
            # valid object masks
            batch_object_masks = object_masks[batch_id]
            batch_object_cls_id = object_semantics[batch_id, batch_object_masks == 1] # num_valid_objects
            
            batch_object_cls_name = [self.Label_ID2CLS[obj_sem_cls_id.cpu().item()] for obj_sem_cls_id in batch_object_cls_id]
            
            batch_object_cls_we = np.zeros((len(batch_object_cls_name), 300)) # (num_valid_objects, 300)
            for ii in range(len(batch_object_cls_name)):
                # if batch_object_cls_name[ii] != "others":
                #     if batch_object_cls_name[ii] == "shower curtain":
                #         batch_object_cls_we[ii] = self.glove["curtain"]
                #     else:
                #         batch_object_cls_we[ii] = self.glove[batch_object_cls_name[ii]]
                if batch_object_cls_name[ii] != "shower curtain":
                    batch_object_cls_we[ii] = self.glove[batch_object_cls_name[ii]]
                else:
                    batch_object_cls_we[ii] = (self.glove["shower"]+self.glove["curtain"])/2.
            batch_object_cls_we = torch.from_numpy(batch_object_cls_we).to(torch.float32).cuda()

            # get global normals
            batch_object_normals = object_normals[batch_id, batch_object_masks == 1] # num_valid_objects, 3
            batch_global_object_normals = torch.cat([batch_object_normals, global_normal], dim=0) # num_valid_objects+1, 3

            # create adjacent matric for this scene
            batch_adjacent_mat = adjacent_mat[batch_id] # num_objects, num_objects
            batch_adjacent_mat = batch_adjacent_mat[batch_object_masks == 1, :][:, batch_object_masks == 1] # num_valid_objects, num_valid_objects
            batch_obj_feats = obj_feats[batch_id, batch_object_masks == 1] # num_valid_objects, in_size
            num_valid_objects = batch_obj_feats.shape[0]

            # get spatial words indicators
            batch_top_mat = top_mat[batch_id]
            batch_top_mat = batch_top_mat[batch_object_masks == 1, :][:, batch_object_masks == 1] # num_valid_objects, num_valid_objects
            batch_bottom_mat = bottom_mat[batch_id]
            batch_bottom_mat = batch_bottom_mat[batch_object_masks == 1, :][:, batch_object_masks == 1] # num_valid_objects, num_valid_objects
            batch_xy_relative_coord = xy_relative_coord[batch_id]
            batch_xy_relative_coord = batch_xy_relative_coord[batch_object_masks == 1, :][:, batch_object_masks == 1] # (num_valid_objects, num_valid_objects, 2)
            

            # add global node info into every batch's adjacent_mat, obj_feats, object_cls_names
            batch_global_mat_row = torch.zeros(1, num_valid_objects).fill_(1).cuda() # 1, num_valid_objects
            batch_global_mat_col = torch.zeros(num_valid_objects+1, 1).fill_(1).cuda() # num_valid_objects+1, 1
            batch_global_adjacent_mat = torch.cat([batch_adjacent_mat, batch_global_mat_row], dim=0) # num_valid_objects+1, num_valid_objects
            batch_global_adjacent_mat = torch.cat([batch_global_adjacent_mat, batch_global_mat_col], dim=1) # num_valid_objects+1, num_valid_objects+1 
            batch_global_obj_feats = torch.cat([batch_obj_feats, global_feat[batch_id]], dim=0) # num_valid_objects+1, in_size
            batch_global_object_cls_we = torch.cat([batch_object_cls_we, global_scene_cls_we], dim=0) # num_valid_objects+1, 300

            # add global node info into every batch's top_mat, bottom_mat
            batch_global_top_row = torch.zeros(1, num_valid_objects).cuda()
            batch_global_top_col = torch.zeros(num_valid_objects+1, 1).cuda()
            batch_global_bottom_row = torch.zeros(1, num_valid_objects).cuda()
            batch_global_bottom_col = torch.zeros(num_valid_objects+1, 1).cuda()
            batch_global_top_mat = torch.cat([batch_top_mat, batch_global_top_row], dim=0)
            batch_global_top_mat = torch.cat([batch_global_top_mat, batch_global_top_col], dim=1) # num_valid_object+1, num_valid_object+1
            batch_global_bottom_mat = torch.cat([batch_bottom_mat, batch_global_bottom_row], dim=0)
            batch_global_bottom_mat = torch.cat([batch_global_bottom_mat, batch_global_bottom_col], dim=1) # num_valid_object+1, num_valid_object+1
            
            # add global node info into every batch's xy_relative_coord
            batch_global_xy_relative_row = global_relative_coord[batch_id, batch_object_masks == 1].unsqueeze(0) # 1, num_valid_objects, 3
            batch_global_xy_relative_row = batch_global_xy_relative_row[:, :, :2] # 1, num_valid_objects, 2
            
            batch_global_xy_relative_col = torch.cat([batch_global_xy_relative_row, self_relative_coord], dim=1) # 1, num_valid_objects+1, 2
            batch_global_xy_relative_col = -1 * batch_global_xy_relative_col.permute(1, 0, 2) #  num_valid_objects+1, 1, 2
            batch_global_xy_relative_coord = torch.cat([batch_xy_relative_coord, batch_global_xy_relative_row], dim=0)
            batch_global_xy_relative_coord = torch.cat([batch_global_xy_relative_coord, batch_global_xy_relative_col], dim=1) # num_valid_object+1, num_valid_object+1, 2

            batch_spatial_words_indicator = torch.cat([batch_global_top_mat.unsqueeze(-1), batch_global_bottom_mat.unsqueeze(-1), batch_global_xy_relative_coord], dim=-1) # (num_valid_objects+1, num_valid_objects+1, 4)
            # initialize graph for this scene
            # sparse_mat = coo_matrix(batch_adjacent_mat.detach().cpu().numpy())
            sparse_mat = coo_matrix(batch_global_adjacent_mat.detach().cpu().numpy())
            batch_edge_index, edge_attr = GUtils.from_scipy_sparse_matrix(sparse_mat)
            row, col = batch_edge_index
            batch_spatial_words_indicator = batch_spatial_words_indicator[row, col] # num_valid_objects * num_locals, 4
            
            # batch_node_feats = self._obj_proposal_we_fusion(batch_obj_feats, batch_object_cls_we)
            batch_node_feats = self._obj_proposal_we_fusion(batch_global_obj_feats, batch_global_object_cls_we)

            # ipdb.set_trace()
            batch_graph = GData(x=batch_node_feats, edge_index=batch_edge_index.cuda(), edge_attr=batch_spatial_words_indicator, obj_normals=batch_global_object_normals)

            # graph conv
            node_feat, edge_feat = self._feed(batch_graph)

            # # skip connection
            # node_feat += batch_obj_feats
            # new_obj_feats[batch_id, batch_object_masks == 1] = node_feat

            # output last edge
            if self.return_orientation:
                # output edge
                try:
                    num_src_objects = len(set(batch_edge_index[0].cpu().numpy()))
                    num_tar_objects = int(edge_feat.shape[0] / num_src_objects)

                    num_sources[batch_id] = num_src_objects
                    num_targets[batch_id] = num_tar_objects

                    edge_feat = edge_feat[:num_src_objects*num_tar_objects] # in case there are less than 10 neighbors                    
                    edge_feats[batch_id, :num_src_objects, :num_tar_objects] = edge_feat.view(num_src_objects, num_tar_objects, self.out_size)
                    edge_indices[batch_id, :, :num_src_objects*num_tar_objects] = batch_edge_index[:, :num_src_objects*num_tar_objects]
                    
                    _, edge_feat = self.edge_layer(node_feat, batch_edge_index.cuda())
                    edge_pred = self.edge_predict(edge_feat)
                    
                    edge_preds[batch_id, :num_src_objects*num_tar_objects] = edge_pred

                except Exception:
                    print("error occurs when dealing with graph, skipping...")

            # skip connection
            # batch_obj_feats += node_feat
            # batch_obj_feats = batch_obj_feats + node_feat
            batch_obj_feats = batch_global_obj_feats + node_feat
            new_obj_feats[batch_id, batch_object_masks == 1] = batch_obj_feats[:-1,:]

        # store
        data_dict["bbox_feature"] = new_obj_feats
        data_dict["adjacent_mat"] = adjacent_mat
        data_dict["edge_index"] = edge_indices
        data_dict["edge_feature"] = edge_feats
        data_dict["num_edge_source"] = num_sources
        data_dict["num_edge_target"] = num_targets
        data_dict["edge_orientations"] = edge_preds[:, :, :-1]
        data_dict["edge_distances"] = edge_preds[:, :, -1]

        return data_dict
