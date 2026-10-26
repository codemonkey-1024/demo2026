# mypy: ignore-errors
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from . import layers
from .base_nbfnet import BaseNBFNet
from torch import autograd

class EntityNBFNet(BaseNBFNet):
    """Neural Bellman-Ford Network for Entity Prediction.

    This class extends BaseNBFNet to perform entity prediction in knowledge graphs using a neural
    version of the Bellman-Ford algorithm. It learns entity representations through message passing
    over the graph structure.

    Args:
        input_dim (int): Dimension of input node/relation features
        hidden_dims (list): List of hidden dimensions for each layer
        num_relation (int, optional): Number of relation types. Defaults to 1 (dummy value)
        **kwargs: Additional arguments passed to BaseNBFNet

    Attributes:
        layers (nn.ModuleList): List of GeneralizedRelationalConv layers
        mlp (nn.Sequential): Multi-layer perceptron for final prediction
        query (torch.Tensor): Relation type embeddings used as queries

    Methods:
        bellmanford(data, h_index, r_index, separate_grad=False):
            Performs neural Bellman-Ford message passing iterations.

            Args:
                data: Graph data object containing edge information
                h_index (torch.Tensor): Indices of head entities
                r_index (torch.Tensor): Indices of relations
                separate_grad (bool): Whether to use separate gradients for visualization

            Returns:
                dict: Contains node features and edge weights after message passing

        forward(data, relation_representations, batch):
            Forward pass for entity prediction.

            Args:
                data: Graph data object
                relation_representations (torch.Tensor): Embeddings of relations
                batch: Batch of (head, tail, relation) triples

            Returns:
                torch.Tensor: Prediction scores for tail entities
    """
    def __init__(self, input_dim, hidden_dims, num_relation=1, **kwargs):
        # dummy num_relation = 1 as we won't use it in the NBFNet layer
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i],
                    self.dims[i + 1],
                    num_relation,
                    self.dims[0],
                    self.message_func,
                    self.aggregate_func,
                    self.layer_norm,
                    self.activation,
                    dependent=False,
                    project_relations=True,
                )
            )

        feature_dim = (
            sum(hidden_dims) if self.concat_hidden else hidden_dims[-1]
        ) + input_dim
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(self.num_mlp_layers - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    def bellmanford(self, data, h_index, r_index, separate_grad=False):
        batch_size = len(r_index)

        # initialize queries (relation types of the given triples)
        query = self.query[torch.arange(batch_size, device=r_index.device), r_index]
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(
            batch_size, data.num_nodes, self.dims[0], device=h_index.device, dtype=query.dtype
        )
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))

        size = (data.num_nodes, data.num_nodes)
        edge_weight = query.new_ones(data.num_edges)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:
            # for visualization
            if separate_grad:
                edge_weight = edge_weight.detach().clone().requires_grad_()

            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(
                layer_input,
                query,
                boundary,
                data.edge_index,
                data.edge_type,
                size,
                edge_weight,
            )
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(
            -1, data.num_nodes, -1
        )  # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, relation_representations, batch):
        h_index, t_index, r_index = batch.unbind(-1)

        # initial query representations are those from the relation graph
        self.query = relation_representations

        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        # if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            # data = self.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(
            h_index, t_index, r_index, num_direct_rel=data.num_relations // 2
        )
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # message passing and updated node representations
        output = self.bellmanford(
            data, h_index[:, 0], r_index[:, 0]
        )  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        # extract representations of tail entities from the updated node states
        feature = feature.gather(
            1, index
        )  # (batch_size, num_negative + 1, feature_dim)

        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)


class QueryNBFNet(EntityNBFNet):
    """
    The entity-level reasoner for UltraQuery-like complex query answering pipelines.

    This class extends EntityNBFNet to handle query-specific reasoning in knowledge graphs.
    Key differences from EntityNBFNet include:

    1. Initial node features are provided during forward pass rather than read from triples batch
    2. Query comes from outer loop
    3. Returns distribution over all nodes (assuming t_index covers all nodes)

    Attributes:
        layers: List of neural network layers for message passing
        short_cut: Boolean flag for using residual connections
        concat_hidden: Boolean flag for concatenating hidden states
        mlp: Multi-layer perceptron for final scoring
        num_beam: Beam size for path search
        path_topk: Number of top paths to return

    Methods:
        bellmanford(data, node_features, query, separate_grad=False):
            Performs Bellman-Ford message passing iterations.
            Args:
                data: Graph data object containing edge information
                node_features: Initial node representations
                query: Query representation
                separate_grad: Whether to track gradients separately for edges
            Returns:
                dict: Contains node features and edge weights

        forward(data, node_features, relation_representations, query):
            Main forward pass of the model.
            Args:
                data: Graph data object
                node_features: Initial node features
                relation_representations: Representations for relations
                query: Query representation
            Returns:
                torch.Tensor: Scores for each node

        visualize(data, sample, node_features, relation_representations, query):
            Visualizes reasoning paths for given entities.
            Args:
                data: Graph data object
                sample: Dictionary containing entity masks
                node_features: Initial node features
                relation_representations: Representations for relations
                query: Query representation
            Returns:
                dict: Contains paths and weights for target entities
    """

    def bellmanford(self, data, node_features, query, separate_grad=False):
        size = (data.num_nodes, data.num_nodes)
        edge_weight = query.new_ones(data.num_edges)

        hiddens = []
        edge_weights = []
        layer_input = node_features

        for layer in self.layers:
            # for visualization
            if separate_grad:
                edge_weight = edge_weight.detach().clone().requires_grad_()

            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(
                layer_input,
                query,
                node_features,
                data.edge_index,
                data.edge_type,
                size,
                edge_weight,
            )
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(
            -1, data.num_nodes, -1
        )  # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, node_features, relation_representations, query):
        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        # we already did traversal_dropout in the outer loop of UltraQuery
        # if self.training:
        #     # Edge dropout in the training mode
        #     # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
        #     # to make NBFNet iteration learn non-trivial paths
        #     data = self.remove_easy_edges(data, h_index, t_index, r_index)

        # node features arrive in shape (bs, num_nodes, dim)
        # NBFNet needs batch size on the first place
        output = self.bellmanford(
            data, node_features, query
        )  # (num_nodes, batch_size, feature_dim）
        score = self.mlp(output["node_feature"]).squeeze(-1)  # (bs, num_nodes)
        return score

    def visualize(self, data, sample, node_features, relation_representations, query):
        for layer in self.layers:
            layer.relation = relation_representations

        seed_kwargs = ({"question_entities_mask": sample["question_entities_masks"]}
                       if isinstance(self, QueryflowNBFNet) else {})
        output = self.bellmanford(
            data, node_features, query, separate_grad=True, **seed_kwargs
        )  # (num_nodes, batch_size, feature_dim）
        node_feature = output["node_feature"]
        edge_weights = output["edge_weights"]
        question_entities_mask = sample["question_entities_masks"]
        target_entities_mask = sample["supporting_entities_masks"]
        query_entities_index = question_entities_mask.nonzero(as_tuple=True)[1]
        target_entities_index = target_entities_mask.nonzero(as_tuple=True)[1]

        paths_results = {}
        for t_index in target_entities_index:
            index = t_index.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(-1, -1, node_feature.shape[-1])
            feature = node_feature.gather(1, index).squeeze(0)
            score = self.mlp(feature).squeeze(-1)

            edge_grads = autograd.grad(score, edge_weights, retain_graph=True)
            distances, back_edges = self.beam_search_distance(data, edge_grads, query_entities_index, t_index, self.num_beam)
            paths, weights = self.topk_average_length(distances, back_edges, t_index, self.path_topk)
            paths_results[t_index.item()] = (paths, weights)
        return paths_results


class QueryflowNBFNet(QueryNBFNet):
    """QueryNBFNet with a recurrent query state between propagation layers.

    After each layer, the current node
    states are summarized against the active query and a gated GRU transition
    updates the query used by the next layer's relation gate and boundary.
    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        num_relation=1,
        query_shift_dropout=0.0,
        query_shift_temperature=1.0,
        activation_checkpointing=False,
        **kwargs,
    ):
        super().__init__(input_dim, hidden_dims, num_relation=num_relation, **kwargs)
        if any(dim != self.dims[0] for dim in self.dims):
            raise ValueError("QueryflowNBFNet requires equal input and hidden dimensions.")
        if query_shift_temperature <= 0:
            raise ValueError("query_shift_temperature must be positive.")

        hidden_dim = self.dims[0]
        self.query_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.query_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relation_query_gate = nn.Linear(hidden_dim, hidden_dim)
        self.summary_norm = nn.LayerNorm(hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.query_shift_dropout = nn.Dropout(query_shift_dropout)
        self.query_shift_temperature = float(query_shift_temperature)
        self.activation_checkpointing = bool(activation_checkpointing)

    def _summarize_active_nodes(self, hidden, query):
        scale = math.sqrt(hidden.shape[-1]) * self.query_shift_temperature
        attention_scores = (hidden * query.unsqueeze(1)).sum(dim=-1) / scale
        attention = torch.softmax(attention_scores, dim=1)
        summary = torch.einsum("bn,bnd->bd", attention, hidden)
        return self.summary_norm(summary)

    def _update_query(self, query, active_summary, layer_index=None):
        candidate = self.query_update(active_summary, query)
        gate = torch.sigmoid(self.query_gate(torch.cat([query, active_summary], dim=-1)))
        shifted = query + gate * self.query_shift_dropout(candidate - query)
        return self.query_norm(shifted)

    def _compose_node_feature(self, hiddens, query, num_nodes):
        node_query = query.unsqueeze(1).expand(-1, num_nodes, -1)
        if self.concat_hidden:
            return torch.cat(hiddens + [node_query], dim=-1)
        return torch.cat([hiddens[-1], node_query], dim=-1)

    def bellmanford(
        self,
        data,
        node_features,
        query,
        separate_grad=False,
        question_entities_mask=None,
    ):
        size = (data.num_nodes, data.num_nodes)
        edge_weight = query.new_ones(data.num_edges)
        relation_representations = self.layers[0].relation
        if relation_representations is None:
            raise ValueError("QueryflowNBFNet requires relation representations before bellmanford().")

        if question_entities_mask is None:
            query_entity_mask = (
                node_features.detach().abs().sum(dim=-1, keepdim=True) > 0
            ).to(node_features.dtype)
        else:
            query_entity_mask = question_entities_mask.unsqueeze(-1).to(node_features.dtype)
        query_state = query
        hiddens = []
        edge_weights = []
        layer_input = node_features

        for layer_index, layer in enumerate(self.layers):
            if separate_grad:
                edge_weight = edge_weight.detach().clone().requires_grad_()

            relation_gate = torch.sigmoid(self.relation_query_gate(query_state)).unsqueeze(1)
            layer.relation = relation_representations * relation_gate
            boundary = query_entity_mask * query_state.unsqueeze(1)
            layer_args = (
                layer_input,
                query_state,
                boundary,
                data.edge_index,
                data.edge_type,
                size,
                edge_weight,
                layer.relation,
            )
            if self.activation_checkpointing and self.training and not separate_grad:
                hidden = checkpoint(layer, *layer_args, use_reentrant=False)
            else:
                hidden = layer(*layer_args)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            query_state = self._update_query(
                query_state,
                self._summarize_active_nodes(hidden, query_state),
                layer_index=layer_index,
            )
            layer_input = hidden

        output = self._compose_node_feature(hiddens, query, data.num_nodes)
        return {"node_feature": output, "edge_weights": edge_weights}

    def forward(
        self,
        data,
        node_features,
        relation_representations,
        query,
        question_entities_mask=None,
    ):
        for layer in self.layers:
            layer.relation = relation_representations
        output = self.bellmanford(
            data,
            node_features,
            query,
            question_entities_mask=question_entities_mask,
        )
        return self.mlp(output["node_feature"]).squeeze(-1)

