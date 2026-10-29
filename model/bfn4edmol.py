import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

'''
Main BFN Implementation, refer to the paper for details.
@article{qu2024molcraft,
  title={MolCRAFT: Structure-Based Drug Design in Continuous Parameter Space},
  author={Qu, Yanru and Qiu, Keyue and Song, Yuxuan and Gong, Jingjing and Han, Jiawei and Zheng, Mingyue and Zhou, Hao and Ma, Wei-Ying},
  journal={ICML 2024},
  year={2024}
}

@article{song2024unified,
  title={Unified Generative Modeling of 3D Molecules via Bayesian Flow Networks},
  author={Song, Yuxuan and Gong, Jingjing and Qu, Yanru and Zhou, Hao and Zheng, Mingyue and Liu, Jingjing and Ma, Wei-Ying},
  journal={ICLR 2024},
  year={2024}
}

git repo: https://github.com/AlgoMole/MolCRAFT/tree/master/MolCRAFT
'''

# core.model.bfn_base.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum
import torch.distributions as dist

def build_e_gt(edge_type_gt, edge_index_gt, edge_index_pred, batch):
    device = edge_type_gt.device

    # Normalize edge_index_gt to [E_gt, 2]
    if edge_index_gt.dim() == 2 and edge_index_gt.size(0) == 2:
        edge_index_gt = edge_index_gt.t()  # [E_gt, 2]
    elif edge_index_gt.dim() != 2 or edge_index_gt.size(1) != 2:
        raise ValueError("edge_index_gt must be [2, E] or [E, 2]")

    # Normalize edge_index_pred to [E_pred, 2]
    if edge_index_pred.dim() == 2 and edge_index_pred.size(0) == 2:
        edge_index_pred = edge_index_pred.t()  # [E_pred, 2]
    elif edge_index_pred.dim() != 2 or edge_index_pred.size(1) != 2:
        raise ValueError("edge_index_pred must be [2, E] or [E, 2]")

    E_pred = edge_index_pred.size(0)
    if E_pred == 0:
        return torch.empty(0, dtype=edge_type_gt.dtype, device=device), \
               torch.empty(0, dtype=torch.long, device=device)

    N_total = batch.size(0)
    if N_total == 0:
        return torch.zeros(E_pred, dtype=edge_type_gt.dtype, device=device), \
               torch.zeros(E_pred, dtype=torch.long, device=device)

    # Encode edges as unique keys: src * (N_total + 1) + dst
    M = N_total + 1
    edge_key_gt = edge_index_gt[:, 0] * M + edge_index_gt[:, 1]  # [E_gt]
    edge_key_pred = edge_index_pred[:, 0] * M + edge_index_pred[:, 1]  # [E_pred]

    # Build lookup table: key -> edge_type
    max_key = torch.cat([edge_key_gt, edge_key_pred]).max().item() if edge_key_gt.numel() > 0 else edge_key_pred.max().item()
    lookup = torch.zeros(max_key + 1, dtype=edge_type_gt.dtype, device=device)
    if edge_key_gt.numel() > 0:
        lookup[edge_key_gt] = edge_type_gt

    # Get ground-truth types for predicted edges
    e_gt = lookup[edge_key_pred]  # [E_pred]

    # Get batch for each predicted edge (src and dst must be in same graph)
    src_batch = batch[edge_index_pred[:, 0]]
    e_batch = src_batch  # [E_pred]

    return e_gt, e_batch
import numpy as np
from scipy.optimize import linear_sum_assignment

def match_fixed_to_generated(fixed_pos, fixed_types, gen_pos, gen_type_probs, sigma_final=0.1):
    """
    fixed_pos: [m, 3]
    fixed_types: [m] (class indices)
    gen_pos: [N, 3]
    gen_type_probs: [N, K]
    """
    m, N = len(fixed_pos), len(gen_pos)
    assert m <= N

    # Coordinate log-likelihood (up to const)
    diff = fixed_pos.unsqueeze(1) - gen_pos.unsqueeze(0)  # [m, N, 3]
    coord_ll = -0.5 * (diff ** 2).sum(dim=-1) / (sigma_final ** 2)  # [m, N]

    # Type log-likelihood
    type_ll = torch.log(gen_type_probs[:, fixed_types].T + 1e-8)  # [m, N]

    # Total log-likelihood (we minimize negative)
    cost_matrix = -(coord_ll + type_ll).cpu().numpy()  # [m, N]

    # Solve assignment
    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    return col_idx  # matched indices in gen_pos

def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode="protein"):
    if mode == "none":
        offset = 0.0
        pass
    elif mode == "protein":
        offset = scatter_mean(protein_pos, batch_protein, dim=0)
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, offset

LOG2PI = np.log(2 * np.pi)


class BFNBase(nn.Module):
    # this is a general method which could be used for implement vector field in CNF or
    def __init__(self, *args, **kwargs):
        super(BFNBase, self).__init__(*args, **kwargs)

    def get_k_params(self, bins):
        """
        function to get the k parameters for the discretised variable
        """
        list_c = []
        list_l = []
        list_r = []
        for k in range(1, int(bins + 1)):
            k_c = (2 * k - 1) / bins - 1
            k_l = k_c - 1 / bins
            k_r = k_c + 1 / bins
            list_c.append(k_c)
            list_l.append(k_l)
            list_r.append(k_r)

        return list_c, list_l, list_l

    def discretised_cdf(self, mu, sigma, x):
        """
        cdf function for the discretised variable
        """
        # in this case we use the discretised cdf for the discretised output function
        mu = mu.unsqueeze(1)
        sigma = sigma.unsqueeze(1)  # B,1,D

        f_ = 0.5 * (1 + torch.erf((x - mu) / (sigma * np.sqrt(2))))
        flag_upper = torch.ge(x, 1)
        flag_lower = torch.le(x, -1)
        f_ = torch.where(flag_upper, torch.ones_like(f_), f_)
        f_ = torch.where(flag_lower, torch.zeros_like(f_), f_)

        return f_

    def continuous_var_bayesian_update(self, t, sigma1, x):
        """
        x: [N, D]
        """
        """
        TODO: rename this function to bayesian flow
        """
        # Eq.(77): p_F(θ|x;t) ~ N (μ | γ(t)x, γ(t)(1 − γ(t))I)
        gamma = 1 - torch.pow(sigma1, 2 * t)  # [B]
        mu = gamma * x + torch.randn_like(x) * torch.sqrt(gamma * (1 - gamma))
        return mu, gamma

    def discrete_var_bayesian_update(self, t, beta1, x, K):
        """
        x: [N, K]
        """
        # Eq.(182): β(t) = t**2 β(1)
        beta = beta1 * (t**2)  # (B,)

        # Eq.(185): p_F(θ|x;t) = E_{N(y | β(t)(Ke_x−1), β(t)KI)} δ (θ − softmax(y))
        # can be sampled by first drawing y ~ N(y | β(t)(Ke_x−1), β(t)KI)
        # then setting θ = softmax(y)
        one_hot_x = x  # (N, K)
        mean = beta * (K * one_hot_x - 1)
        std = (beta * K).sqrt()
        eps = torch.randn_like(mean)
        y = mean + std * eps
        theta = F.softmax(y, dim=-1)
        return theta

    def discreteised_var_bayesian_update(self, t, sigma1, x):
        """
        x: [N, D]
        Note, this is identical to the continuous_var_bayesian_update
        """
        gamma = 1 - torch.pow(sigma1, 2 * t)
        mu = gamma * x + torch.randn_like(x) * torch.sqrt(gamma * (1 - gamma))
        return mu, gamma

    def ctime4continuous_loss(self, t, sigma1, x_pred, x, segment_ids=None):
        # Eq.(101): L∞(x) = −ln(σ1) * E_{t∼U (0,1), p_F(θ|x;t)} [|x − x_hat(θ,t)|**2 / (σ_1**2)**t]
        if segment_ids is not None:
            loss = scatter_mean(
                torch.pow(sigma1, -2 * t.view(-1))
                * ((x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)),
                segment_ids,
                dim=0,
            )
        else:
            loss = torch.pow(sigma1, -2 * t.view(-1)) * (x_pred - x).view(
                x.shape[0], -1
            ).abs().pow(2).sum(dim=1)
        return -torch.log(sigma1) * loss

    def dtime4continuous_loss(self, i, N, sigma1, x_pred, x, segment_ids=None):
        # TODO not debuged yet
        weight = N * (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2 * i / N))
        # print(x_pred.shape, x.shape , i.shape,weight.shape)
        # print(segment_ids)
        if segment_ids is not None:
            loss = scatter_mean(
                weight.view(-1) * ((x_pred - x) ** 2).sum(-1), segment_ids, dim=0
            )
        else:
            loss = (
                N
                * (1 - torch.pow(sigma1, 2 / N))
                / (2 * torch.pow(sigma1, 2 * i / N))
                * (x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)
            )

        # print(loss.shape)
        return loss

    def ctime4discrete_loss(self, t, beta1, one_hot_x, p_0, K, segment_ids=None):
        # Eq.(205): L∞(x) = Kβ(1) E_{t∼U (0,1), p_F (θ|x,t)} [t|e_x − e_hat(θ, t)|**2,
        # where e_hat(θ, t) = (\sum_k p_O^(1) (k | θ; t)e_k, ..., \sum_k p_O^(D) (k | θ; t)e_k)
        e_x = one_hot_x  # [N, K]
        e_hat = p_0  # (N, K)
        assert e_x.size() == e_hat.size()
        if segment_ids is not None:
            L_infinity = scatter_mean(
                K * beta1 * t.view(-1) * ((e_x - e_hat) ** 2).sum(dim=-1),
                segment_ids,
                dim=0,
            )
        else:
            L_infinity = K * beta1 * t.view(-1) * ((e_x - e_hat) ** 2).sum(dim=-1)
        return L_infinity

    def dtime4discrete_loss_prob(
        self, i, N, beta1, one_hot_x, p_0, K, n_samples=200, segment_ids=None
    ):
        # this is based on the official implementation of BFN.
        # import pdb
        # pdb.set_trace()
        target_x = one_hot_x  # [D, K]
        e_hat = p_0  # (D,  K)
        alpha = beta1 * (2 * i - 1) / N**2  # [D]
        alpha = alpha.view(-1, 1) # [D, 1]
        classes = torch.arange(K, device=target_x.device).long().unsqueeze(0)  # [ 1, K]
        e_x = F.one_hot(classes.long(), K) #[1,K, K]
        # print(e_x.shape)
        receiver_components = dist.Independent(
            dist.Normal(
                alpha.unsqueeze(-1) * ((K * e_x) - 1), # [D K, K]
                (K * alpha.unsqueeze(-1)) ** 0.5, # [D, 1, 1]
            ),
            1,
        )  # [D,T, K, K]
        receiver_mix_distribution = dist.Categorical(probs=e_hat)  # [D, K]
        receiver_dist = dist.MixtureSameFamily(
            receiver_mix_distribution, receiver_components
        )  # [D, K]
        sender_dist = dist.Independent( dist.Normal(
            alpha* ((K * target_x) - 1), ((K * alpha) ** 0.5)
        ),1)  # [D, K]
        y = sender_dist.sample(torch.Size([n_samples])) 
        loss = N * (sender_dist.log_prob(y) - receiver_dist.log_prob(y)).mean(0).mean(
            -1, keepdims=True
        )
        return loss.mean()

    def dtime4discrete_loss(self, i, N, beta1, one_hot_x, p_0, K, segment_ids=None):
        # i in {1,n}
        # Algorithm 7 in BFN
        e_x = one_hot_x  # [D, K]
        e_hat = p_0  # (D, K)
        assert e_x.size() == e_hat.size()
        alpha = beta1 * (2 * i - 1) / N**2  # [D]

        # print(alpha.shape)
        mean_ = alpha * (K * e_x - 1)  # [D, K]
        std_ = torch.sqrt(alpha * K)  # [D,1] TODO check shape
        eps = torch.randn_like(mean_)  # [D,K,]
        y_ = mean_ + std_ * eps
        # modify this line:
        matrix_ek = torch.eye(K, K).unsqueeze(0).to(e_x.device)
        matrix_ek.repeat(alpha.size(0), 1, 1)  # [D,K,K]
        mean_matrix = alpha.unsqueeze(-1) * (K * matrix_ek - 1)  # [D,K,K]
        std_matrix = torch.sqrt(alpha * K).unsqueeze(-1)  #
        likelihood = (
            torch.exp(
                -((y_.unsqueeze(1).repeat(1, K, 1) - mean_matrix) ** 2)
                / (2 * std_matrix**2)
            )
            / (std_matrix * np.sqrt(2 * np.pi))
        ).prod(
            -1
        )  # [D,K]

        if segment_ids is not None:
            L_N = -scatter_mean(
                torch.log((likelihood * e_hat).sum(dim=-1)), segment_ids, dim=0
            )
        else:
            L_N = -torch.log((likelihood * e_hat).sum(dim=-1))  # [D]
        # print(L_N.shape)
        #
        return N * L_N

    def dtime4discrete_loss_gjj(self, i, N, beta1, one_hot_x, p_0, K, segment_ids=None):
        # i in {1,n}
        # Algorithm 7 in BFN
        e_x = one_hot_x  # [D, K]
        e_hat = p_0  # (D, K)
        assert e_x.size() == e_hat.size()
        alpha = beta1 * (2 * i - 1) / N**2  # [D]

        # print(alpha.shape)
        mean_ = alpha * (K * e_x - 1)  # [D, K]
        std_ = torch.sqrt(alpha * K)  # [D,1] TODO check shape
        eps = torch.randn_like(mean_)  # [D,K,]
        y_ = mean_ + std_ * eps  # [D, K]
        # modify this line:
        matrix_ek = torch.eye(K, K).to(e_x.device)  # [K, K]
        mean_matrix = K * matrix_ek - 1  # [K, K]
        std_matrix = torch.sqrt(alpha * K).unsqueeze(-1)  #
        _log_gaussians = (  # [D, K]
            (-0.5 * LOG2PI - torch.log(std_matrix))
            - (y_.unsqueeze(1) - mean_matrix) ** 2 / (2 * std_matrix**2)
        ).sum(-1)

        _inner_log_likelihood = torch.log(
            torch.sum(e_hat * torch.exp(_log_gaussians), dim=-1)
        )  # (D,)

        _inner_log_likelihood = torch.log(e_hat) + _log_gaussians  # [D, K]
        log_likelihood = torch.logsumexp(_inner_log_likelihood, dim=-1)  # [D]

        if segment_ids is not None:
            L_N = -scatter_mean(log_likelihood, segment_ids, dim=0)
        else:
            L_N = -log_likelihood.sum(dim=-1)  # [D]
        # print(L_N.shape)
        #
        return N * L_N

    def ctime4discreteised_loss(self, t, sigma1, x_pred, x, segment_ids=None):
        if segment_ids is not None:
            loss = scatter_sum(
                (x_pred - x).view(x.shape[0], -1).abs().pow(2), segment_ids, dim=0
            )
        else:
            raise NotImplementedError
            loss = (x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)
        return -torch.log(sigma1) * loss * torch.pow(sigma1, -2 * t.view(-1))

    def interdependency_modeling(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def loss_one_step(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

# core.models.common.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, fixed_offset=True):
        super(GaussianSmearing, self).__init__()
        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians
        if fixed_offset:
            # customized offset
            offset = torch.tensor([0, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10])
        else:
            offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def __repr__(self):
        return f'GaussianSmearing(start={self.start}, stop={self.stop}, num_gaussians={self.num_gaussians})'

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class AngleExpansion(nn.Module):
    def __init__(self, start=1.0, stop=5.0, half_expansion=10):
        super(AngleExpansion, self).__init__()
        l_mul = 1. / torch.linspace(stop, start, half_expansion)
        r_mul = torch.linspace(start, stop, half_expansion)
        coeff = torch.cat([l_mul, r_mul], dim=-1)
        self.register_buffer('coeff', coeff)

    def forward(self, angle):
        return torch.cos(angle.view(-1, 1) * self.coeff.view(1, -1))


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    'silu': nn.SiLU()
}


class MLP(nn.Module):
    """MLP with the same hidden dim across all layers."""

    def __init__(self, in_dim, out_dim, hidden_dim, num_layer=2, norm=True, act_fn='relu', act_last=False):
        super().__init__()
        layers = []
        for layer_idx in range(num_layer):
            if layer_idx == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif layer_idx == num_layer - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layer_idx < num_layer - 1 or act_last:
                if norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(NONLINEARITIES[act_fn])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)



def outer_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    E, C1 = a.shape
    C2 = b.shape[1]
    # [E, C1, 1] * [E, 1, C2] -> [E, C1, C2] -> [E, C1*C2]
    return (a.unsqueeze(-1) * b.unsqueeze(-2)).flatten(1)

def get_h_dist(dist_metric, hi, hj):
    if dist_metric == 'euclidean':
        h_dist = torch.sum((hi - hj) ** 2, -1, keepdim=True)
        return h_dist
    elif dist_metric == 'cos_sim':
        hi_norm = torch.norm(hi, p=2, dim=-1, keepdim=True)
        hj_norm = torch.norm(hj, p=2, dim=-1, keepdim=True)
        h_dist = torch.sum(hi * hj, -1, keepdim=True) / (hi_norm * hj_norm)
        return h_dist, hj_norm


def get_r_feat(r, r_exp_func, node_type=None, edge_index=None, mode='basic'):
    if mode == 'origin':
        r_feat = r
    elif mode == 'basic':
        r_feat = r_exp_func(r)
    elif mode == 'sparse':
        src, dst = edge_index
        nt_src = node_type[src]  # [n_edges, 8]
        nt_dst = node_type[dst]
        r_exp = r_exp_func(r)
        r_feat = outer_product(nt_src, nt_dst, r_exp)
    else:
        raise ValueError(mode)
    return r_feat


def compose_context(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand):
    # previous version has problems when ligand atom types are fixed
    # (due to sorting randomly in case of same element)

    # TODO: check if this runs on cpu
    batch_ctx = torch.cat([batch_protein, batch_ligand], dim=0)
    # sort_idx = batch_ctx.argsort()
    sort_idx = torch.sort(batch_ctx, stable=True).indices

    mask_ligand = torch.cat([
        torch.zeros([batch_protein.size(0)], device=batch_protein.device).bool(),
        torch.ones([batch_ligand.size(0)], device=batch_ligand.device).bool(),
    ], dim=0)[sort_idx]

    batch_ctx = batch_ctx[sort_idx]
    h_ctx = torch.cat([h_protein, h_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, H)
    pos_ctx = torch.cat([pos_protein, pos_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, 3)

    return h_ctx, pos_ctx, batch_ctx, mask_ligand


def compose_context_prop(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand):
    batch_ctx = torch.cat([batch_protein, batch_ligand], dim=0)
    sort_idx = batch_ctx.argsort()

    mask_protein = torch.cat([
        torch.ones([batch_protein.size(0)], device=batch_protein.device).bool(),
        torch.zeros([batch_ligand.size(0)], device=batch_ligand.device).bool(),
    ], dim=0)[sort_idx]

    batch_ctx = batch_ctx[sort_idx]
    h_ctx = torch.cat([h_protein, h_ligand], dim=0)[sort_idx]       # (N_protein+N_ligand, H)
    pos_ctx = torch.cat([pos_protein, pos_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, 3)

    return h_ctx, pos_ctx, batch_ctx

## new function for edmol - glt
def compose_context_edmol(h_pocket, h_ed, h_mol, pos_pocket, pos_ed, pos_mol, batch_pocket, batch_ed, batch_mol):
    batch_ctx = torch.cat([batch_pocket, batch_ed, batch_mol], dim=0)
    sort_idx = batch_ctx.argsort()

    mask_pocket = torch.cat([
        torch.ones([batch_pocket.size(0)], device=batch_pocket.device).bool(),
        torch.zeros([batch_ed.size(0)], device=batch_ed.device).bool(),
        torch.zeros([batch_mol.size(0)], device=batch_mol.device).bool(),
    ], dim=0)[sort_idx]

    mask_ed = torch.cat([
        torch.zeros([batch_pocket.size(0)], device=batch_pocket.device).bool(),
        torch.ones([batch_ed.size(0)], device=batch_ed.device).bool(),
        torch.zeros([batch_mol.size(0)], device=batch_mol.device).bool(),
    ], dim=0)[sort_idx]

    mask_mol = torch.cat([
        torch.zeros([batch_pocket.size(0)], device=batch_pocket.device).bool(),
        torch.zeros([batch_ed.size(0)], device=batch_ed.device).bool(),
        torch.ones([batch_mol.size(0)], device=batch_mol.device).bool(),
    ], dim=0)[sort_idx]

    batch_ctx = batch_ctx[sort_idx]
    h_ctx = torch.cat([h_pocket, h_ed, h_mol], dim=0)[sort_idx]  # (N_pocket+N_ed+N_mol, H)
    pos_ctx = torch.cat([pos_pocket, pos_ed, pos_mol], dim=0)[sort_idx]  # (N_pocket+N_ed+N_mol, 3)

    return h_ctx, pos_ctx, batch_ctx, mask_pocket, mask_ed, mask_mol


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


# core.models.unitransformer.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
from torch_scatter import scatter_softmax, scatter_sum



class BaseX2HAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r', out_fc=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.act_fn = act_fn
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.ew_net_type = ew_net_type
        self.out_fc = out_fc

        # attention key func
        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim
        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention value func
        self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention query func
        self.hq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())
        elif ew_net_type == 'm':
            self.ew_net = nn.Sequential(nn.Linear(output_dim, 1), nn.Sigmoid())

        if self.out_fc:
            self.node_output = MLP(2 * hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)

    def forward(self, h, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        # multi-head attention
        # decide inputs of k_func and v_func
        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        # compute k
        k = self.hk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        # compute v
        v = self.hv_func(kv_input)

        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = self.ew_net(v[..., :self.hidden_dim])
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w
        v = v.view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute q
        q = self.hq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0,
                                dim_size=N)  # [num_edges, n_heads]

        # perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, H_per_head)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, H_per_head)
        output = output.view(-1, self.output_dim)
        if self.out_fc:
            output = self.node_output(torch.cat([output, h], -1))

        output = output + h
        
        return output, m.view(-1, self.output_dim)


class BaseH2XAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.act_fn = act_fn
        self.ew_net_type = ew_net_type

        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim

        self.xk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.xv_func = MLP(kv_input_dim, self.n_heads, hidden_dim, norm=norm, act_fn=act_fn)
        self.xq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())

    def forward(self, h, rel_x, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        # multi-head attention
        # decide inputs of k_func and v_func
        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        k = self.xk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        v = self.xv_func(kv_input)
        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = 1.
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w

        v = v.unsqueeze(-1) * rel_x.unsqueeze(1)  # (xi - xj) [n_edges, n_heads, 3]
        q = self.xq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # Compute attention weights
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0, dim_size=N)  # (E, heads)

        # Perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, 3)
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, 3)

        return output.mean(1)  # [num_nodes, 3]


class AttentionLayerO2TwoUpdateNodeGeneral(nn.Module):
    def __init__(self, hidden_dim, n_heads, num_r_gaussian, edge_feat_dim, act_fn='relu', norm=True,
                 num_x2h=1, num_h2x=1, r_min=0., r_max=10., num_node_types=8,
                 ew_net_type='r', x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.norm = norm
        self.act_fn = act_fn
        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.r_min, self.r_max = r_min, r_max
        self.num_node_types = num_node_types
        self.ew_net_type = ew_net_type
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup

        self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)

        self.x2h_layers = nn.ModuleList()
        for i in range(self.num_x2h):
            self.x2h_layers.append(
                BaseX2HAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type, out_fc=self.x2h_out_fc)
            )
        self.h2x_layers = nn.ModuleList()
        for i in range(self.num_h2x):
            self.h2x_layers.append(
                BaseH2XAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type)
            )

    def forward(self, h, x, edge_attr, edge_index, mask_ligand, e_w=None, fix_x=False):
        src, dst = edge_index
        if self.edge_feat_dim > 0:
            edge_feat = edge_attr  # shape: [#edges_in_batch, #bond_types]
        else:
            edge_feat = None

        rel_x = x[dst] - x[src]
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        h_in = h
        # 4 separate distance embedding for p-p, p-l, l-p, l-l
        for i in range(self.num_x2h):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)
            h_out, e_attr = self.x2h_layers[i](h_in, dist_feat, edge_feat, edge_index, e_w=e_w)
            h_in = h_out
        x2h_out = h_in

        new_h = h if self.sync_twoup else x2h_out
        for i in range(self.num_h2x):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)
            delta_x = self.h2x_layers[i](new_h, rel_x, dist_feat, edge_feat, edge_index, e_w=e_w)
            if not fix_x:
                x = x + delta_x * mask_ligand[:, None]  # only ligand positions will be updated
            rel_x = x[dst] - x[src]
            dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)
        return x2h_out, x, e_attr

class UniTransformerO2TwoUpdateGeneral(nn.Module):
    def __init__(self, num_blocks, num_layers, hidden_dim, n_heads=1, knn=32,
                 num_r_gaussian=50, edge_feat_dim=0, num_node_types=8, act_fn='relu', norm=True,
                 ew_net_type='r',
                 num_init_x2h=1, num_init_h2x=0, num_x2h=1, num_h2x=1, r_max=10., x2h_out_fc=True, sync_twoup=False, name='unio2net'):
        super().__init__()
        self.name = name
        # Build the network
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.act_fn = act_fn
        self.norm = norm
        self.num_node_types = num_node_types
        # radius graph / knn graph
        self.knn = knn
        self.ew_net_type = ew_net_type  # [r, m, none]

        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.num_init_x2h = num_init_x2h
        self.num_init_h2x = num_init_h2x
        self.r_max = r_max
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup
        self.distance_expansion = GaussianSmearing(0., r_max, num_gaussians=num_r_gaussian)
        if self.ew_net_type == 'global':
            self.edge_pred_layer = MLP(num_r_gaussian, 1, hidden_dim)

        self.init_h_emb_layer = self._build_init_h_layer()
        self.base_block = self._build_share_blocks()
        self.cached_edge_index = None  # 添加缓存
        self._current_step = 0  # 记录缓存是在哪一步构建的
        
        self.peb_layer = AttentionLayerO2TwoUpdateNodeGeneral(
            self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn,
            norm=self.norm,
            num_x2h=self.num_x2h, num_h2x=self.num_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
            ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
        )

        self.old_edge_build_mode = False
    
    def __repr__(self):
        return f'UniTransformerO2(num_blocks={self.num_blocks}, num_layers={self.num_layers}, n_heads={self.n_heads}, ' \
               f'act_fn={self.act_fn}, norm={self.norm}, ew_net_type={self.ew_net_type}, ' \
               f'init h emb: {self.init_h_emb_layer.__repr__()} \n' \
               f'base block: {self.base_block.__repr__()} \n' \
               f'edge pred layer: {self.edge_pred_layer.__repr__() if hasattr(self, "edge_pred_layer") else "None"}) '

    def _build_init_h_layer(self):
        layer = AttentionLayerO2TwoUpdateNodeGeneral(
            self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn, norm=self.norm,
            num_x2h=self.num_init_x2h, num_h2x=self.num_init_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
            ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
        )
        return layer

    def _build_share_blocks(self):
        # Equivariant layers
        base_block = []
        for l_idx in range(self.num_layers):
            layer = AttentionLayerO2TwoUpdateNodeGeneral(
                self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn,
                norm=self.norm,
                num_x2h=self.num_x2h, num_h2x=self.num_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
                ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
            )
            base_block.append(layer)
        return nn.ModuleList(base_block)

    def _connect_edge(self, x, mask_mol, batch):
        return knn_graph(x, k=self.knn, batch=batch, flow='source_to_target')
    
    @staticmethod
    def _build_edge_type(edge_index, mask_mol):
        src, dst = edge_index
        edge_type = 3 - 2 * mask_mol[src] - 1 * mask_mol[dst].long()
        edge_type = F.one_hot(edge_type, num_classes=4)
        return edge_type
    
    @staticmethod
    def _build_potential_bond(x, batch, k_threshold=6, r_threshold=1.6):
        
        edge_index_k = knn_graph(x, k=k_threshold, batch=batch, flow='source_to_target')
        # remove edges with distance > r_threshold
        src, dst = edge_index_k
        dist = torch.norm(x[dst] - x[src], p=2, dim=-1)
        mask = dist < r_threshold
        edge_index = edge_index_k[:, mask]
        return edge_index
    
    def forward(self, h, x, mask_pocket, mask_ed, mask_mol, batch, return_all=False, fix_x=False):
        all_x = [x]
        all_h = [h]
    
        for b_idx in range(self.num_blocks):
            if self.old_edge_build_mode:
                # Step 1: Build mol-mol edges (only once!)
                x_temp_mol = x.clone()
                x_temp_mol[~mask_mol] += 1e6  # Hide non-mol atoms
                edge_index_mol_mol = self._connect_edge(x_temp_mol, mask_mol, batch)
                is_mol_mol = mask_mol[edge_index_mol_mol[0]] & mask_mol[edge_index_mol_mol[1]]
                edge_index_mol_mol = edge_index_mol_mol[:, is_mol_mol]
            
                # Step 2: Build pocket-mol edges
                x_temp_pocket_mol = x.clone()
                x_temp_pocket_mol[mask_ed] += 1e6
                edge_index_pocket_mol = self._connect_edge(x_temp_pocket_mol, mask_mol, batch)
                # Remove mol-mol edges from pocket-mol result
                is_mol_mol = mask_mol[edge_index_pocket_mol[0]] & mask_mol[edge_index_pocket_mol[1]]
                edge_index_pocket_mol = edge_index_pocket_mol[:, ~is_mol_mol]
            
                # Step 3: Build ed-mol edges
                x_temp_ed_mol = x.clone()
                x_temp_ed_mol[mask_pocket] += 1e6
                edge_index_ed_mol = self._connect_edge(x_temp_ed_mol, mask_mol, batch)
                is_mol_mol = mask_mol[edge_index_ed_mol[0]] & mask_mol[edge_index_ed_mol[1]]
                edge_index_ed_mol = edge_index_ed_mol[:, ~is_mol_mol]
            
                # Combine without duplicates
                edge_index = torch.cat([
                    edge_index_mol_mol,
                    edge_index_pocket_mol,
                    edge_index_ed_mol
                ], dim=1)
    
            else:
                edge_index = self._connect_edge(x, mask_mol, batch)  # 或直接 radius/knn on x
        
            src, dst = edge_index
            edge_type = self._build_edge_type(edge_index, mask_mol)
            
            # # Edge type (dim: 4)
            # src, dst = edge_index
            # edge_type = self._build_edge_type(edge_index, mask_mol)
    
            if self.ew_net_type == 'global':
                dist = torch.norm(x[dst] - x[src], p=2, dim=-1, keepdim=True)
                dist_feat = self.distance_expansion(dist)
                logits = self.edge_pred_layer(dist_feat)
                e_w = torch.sigmoid(logits)
            else:
                e_w = None
    
            for l_idx, layer in enumerate(self.base_block):
                h, x, _ = layer(h, x, edge_type, edge_index, mask_mol, e_w=e_w, fix_x=fix_x)
            all_x.append(x)
            all_h.append(h)
    
        outputs = {'x': x, 'h': h}
        if return_all:
            outputs.update({'all_x': all_x, 'all_h': all_h})
        
        # x_temp_mol = x.clone()
        # x_temp_mol = x_temp_mol[mask_mol]
        # batch_mol = batch[mask_mol]
        # peb_edge_index = self._build_potential_bond(x_temp_mol, batch_mol)
        peb_edge_index = None
        # mask_mol_ = torch.ones(x_temp_mol.shape[0], dtype=torch.bool, device=x_temp_mol.device)
        # src, dst = peb_edge_index
        # edge_type = self._build_edge_type(peb_edge_index, mask_mol_)
        # if self.ew_net_type == 'global':
        #     dist = torch.norm(x_temp_mol[dst] - x_temp_mol[src], p=2, dim=-1, keepdim=True)
        #     dist_feat = self.distance_expansion(dist)
        #     logits = self.edge_pred_layer(dist_feat)
        #     e_w = torch.sigmoid(logits)
        # else:
        #     e_w = None
        
        # _,_,e = self.peb_layer(h[mask_mol], x[mask_mol], edge_type, peb_edge_index, mask_mol_, e_w=e_w, fix_x=fix_x)
        outputs.update({'e': None, 'ei': peb_edge_index})
        
        return outputs
# core.config.config.py

import yaml
import os
import re
from ast import literal_eval
import json
import copy


class Struct:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                self.__dict__[key] = Struct(**value)
            else:
                self.__dict__[key] = value

    def todict(self):
        # recursively convert to dict
        return {
            k: v.todict() if isinstance(v, Struct) else v
            for k, v in self.__dict__.items()
        }

    def __getitem__(self, index):
        return self.__dict__[index]


class Config:
    def __init__(self, config_file, **kwargs):
        _config = parse_config(path=config_file, subs_dict=kwargs)
        for key, value in _config.items():
            if isinstance(value, dict):
                self.__dict__[key] = Struct(**value)
            else:
                self.__dict__[key] = value

    def __getitem__(self, index):
        return self.__dict__[index]

    def todict(self):
        # recursively convert to dict
        return {
            k: v.todict() if isinstance(v, Struct) else v
            for k, v in self.__dict__.items()
        }

    def save2yaml(self, path):
        with open(path, "w") as f:
            yaml.dump(self.todict(), f, default_flow_style=False)

    def __str__(self):
        def prepare_dict4print(dict_):
            tmp_dict = copy.deepcopy(dict_)

            def recursive_change_list_to_string(d, summarize=16):
                for k, v in d.items():
                    if isinstance(v, dict):
                        recursive_change_list_to_string(v)
                    elif isinstance(v, list):
                        d[k] = (
                            (
                                str(
                                    v[: summarize // 2] + ["..."] + v[-summarize // 2 :]
                                )
                                + f" (len={len(v)})"
                            )
                            if len(v) > summarize
                            else str(v) + f" (len={len(v)})"
                        )
                    else:
                        pass

            recursive_change_list_to_string(tmp_dict)
            return tmp_dict

        return json.dumps(prepare_dict4print(self.todict()), indent=4, sort_keys=False)


def simplest_type(s):
    try:
        return literal_eval(s)
    except:
        return s


# enable parsing of environment variables in yaml files
def parse_config(
    path=None,
    data=None,
    subs_dict={},
    envtag="!ENV",
    substag="!SUB",
    envsubstag="!ENVSUB",
):
    # pattern for global vars: look for ${word}
    pattern = re.compile(".*?\${([\w:\-\.]+)}.*?")
    loader = yaml.FullLoader

    # the tag will be used to mark where to start searching for the pattern
    # e.g. somekey: !ENV somestring${MYENVVAR}blah blah blah
    loader.add_implicit_resolver(envtag, pattern, None)
    loader.add_implicit_resolver(substag, pattern, None)
    loader.add_implicit_resolver(envsubstag, pattern, None)

    def constructor_env_variables(loader, node):
        """
        Extracts the environment variable from the node's value
        :param yaml.Loader loader: the yaml loader
        :param node: the current node in the yaml
        :return: the parsed string that contains the value of the environment
        variable
        """
        value = loader.construct_scalar(node)
        match = pattern.findall(value)  # to find all env variables in line
        if match:
            full_value = value
            for g in match:
                vname, sep, default_val = g.partition(":-")
                full_value = full_value.replace(
                    f"${{{g}}}", os.environ.get(vname, default_val)
                )
            return simplest_type(full_value)
        return simplest_type(value)

    def constructor_subs_variables(loader, node):
        """
        Extracts the environment variable from the node's value
        :param yaml.Loader loader: the yaml loader
        :param node: the current node in the yaml
        :return: the parsed string that contains the value of the environment
        variable
        """
        value = loader.construct_scalar(node)
        match = pattern.findall(value)  # to find all env variables in line
        # print(value)
        # print(match)
        if match:
            full_value = value
            for g in match:
                vname, sep, default_val = g.partition(":-")
                if len(str(subs_dict.get(vname, default_val))) == 0:
                    raise ValueError(
                        f"""argument `{vname}` required, 
                        should be specified from command line with: --{vname} <value>,
                        or set a default value for `{vname}` in `{path}` file.
                        """
                    )
                full_value = full_value.replace(
                    f"${{{g}}}", str(subs_dict.get(vname, default_val))
                )
            return simplest_type(full_value)
        return simplest_type(value)

    def constructor_envsubs_variables(loader, node):
        """
        Extracts the environment variable from the node's value
        :param yaml.Loader loader: the yaml loader
        :param node: the current node in the yaml
        :return: the parsed string that contains the value of the environment
        variable
        """
        value = loader.construct_scalar(node)
        match = pattern.findall(value)  # to find all env variables in line
        # print(value)
        # print(match)
        if match:
            full_value = value
            for g in match:
                vname, sep, default_val = g.partition(":-")
                if (
                    len(str(subs_dict.get(vname, default_val))) == 0
                    and len(str(os.environ.get(vname, default_val))) == 0
                ):
                    raise ValueError(
                        f"""argument `{vname}` required, 
                        should be specified from command line with: --{vname} <value>,
                        or set a default value for `{vname}` in `{path}` file.
                        """
                    )
                full_value = full_value.replace(
                    f"${{{g}}}",
                    str(subs_dict.get(vname, os.environ.get(vname, default_val))),
                )
            return simplest_type(full_value)
        return simplest_type(value)

    def constructor_pathjoin(loader, node):
        value = loader.construct_sequence(node)
        value = [str(v) for v in value]
        return os.path.join(*value)

    def constructor_strjoin(loader, node):
        value = loader.construct_sequence(node)
        value = [str(v) for v in value]
        return "".join(value)

    def constructor_listadd(loader, node):
        value = loader.construct_sequence(node)
        value = sum([simplest_type(v) for v in value])
        return value

    def constructor_listmul(loader, node):
        value = loader.construct_sequence(node)
        ret = 1
        for v in value:
            ret *= simplest_type(v)
        return ret

    loader.add_constructor(envtag, constructor_env_variables)
    loader.add_constructor(substag, constructor_subs_variables)
    loader.add_constructor(envsubstag, constructor_envsubs_variables)
    loader.add_constructor("!PATHJOIN", constructor_pathjoin)
    loader.add_constructor("!STRJOIN", constructor_strjoin)
    loader.add_constructor("!LISTADD", constructor_listadd)
    loader.add_constructor("!LISTMUL", constructor_listmul)

    if path:
        with open(path) as conf_data:
            return yaml.load(conf_data, Loader=loader)
    elif data:
        return yaml.load(data, Loader=loader)
    else:
        raise ValueError("Either a path or data should be defined as input")

# core.models.bfn4sbdd.py

from absl import logging

import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
        
class SinusoidalPosEmbWithLinear(nn.Module):
    def __init__(self, dim_in, dim_out=None):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out or dim_in

        # 原 sinusoidal 编码（不可学习）
        self.register_buffer('freqs', 
            torch.exp(-np.log(10000) * torch.arange(0, dim_in // 2) / (dim_in // 2 - 1))
        )  # 缓存到 buffer，不参与训练，也不保存在 state_dict（但 register_buffer 会保存）

        # 可学习的线性映射
        self.proj = nn.Linear(dim_in, self.dim_out)

    def forward(self, x):
        # x: [..., 1] 或 [... ]，通常为标量位置（如 timestep）
        # 支持任意 batch_shape
        x = x.unsqueeze(-1) if x.ndim == 1 else x  # 确保最后是 [..., 1]
        assert x.shape[-1] == 1, "Last dim must be 1 (scalar per sample)"

        freqs = self.freqs.to(x.device)  # [half_dim]
        emb = x * freqs  # [..., half_dim]

        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # [..., dim_in]
        return self.proj(emb)  # [..., dim_out]

class RBF(nn.Module):
    def __init__(self, start, end, n_center):
        super().__init__()
        self.start = start
        self.end = end
        self.n_center = n_center
        self.centers = torch.linspace(start, end, n_center)
        self.width = (end - start) / n_center

    def forward(self, x):
        assert x.ndim >= 2
        out = (x - self.centers.to(x.device)) / self.width
        ret = torch.exp(-0.5 * out**2)
        return F.normalize(ret, dim=-1, p=1) * 2 - 1


class TimeEmbedLayer(nn.Module):
    def __init__(self, time_emb_mode, time_emb_dim):
        super().__init__()
        self.time_emb_mode = time_emb_mode
        self.time_emb_dim = time_emb_dim

        if self.time_emb_mode == "simple":
            assert self.time_emb_dim == 1
            self.time_emb = lambda x: x
        elif self.time_emb_mode == "sin":
            self.time_emb = nn.Sequential(
                SinusoidalPosEmb(self.time_emb_dim),
                nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                nn.GELU(),
                nn.Linear(self.time_emb_dim * 4, self.time_emb_dim),
            )
        elif self.time_emb_mode == "rbf":
            self.time_emb = RBF(0, 1, self.time_emb_dim)
        elif self.time_emb_mode == "rbfnn":
            self.time_emb = nn.Sequential(
                RBF(0, 1, self.time_emb_dim),
                nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                nn.GELU(),
                nn.Linear(self.time_emb_dim * 4, self.time_emb_dim),
            )
        else:
            raise NotImplementedError

    def forward(self, t):
        return self.time_emb(t)


class BFN4SBDDScoreModel(BFNBase):
    def __init__(
        self,
        # in_node_nf,
        # hidden_nf=64,
        net_config,
        # point_transformer_config,
        protein_atom_feature_dim,
        ligand_atom_feature_dim,
        ed_feature_dim,
        device="cuda",
        condition_time=True,
        sigma1_coord=0.03,
        beta1=1.5,
        use_discrete_t=False,
        discrete_steps=1000,
        t_min=0.0001,
        # no_diff_coord=False,
        node_indicator=True,
        # charge_discretised_loss = False
        time_emb_mode='simple',
        time_emb_dim=0,
        center_pos_mode='protein',
        pos_init_mode='zero',
        destination_prediction = False,
        sampling_strategy = "end_back_pmf"
    ):
        super(BFN4SBDDScoreModel, self).__init__()
        # if include_charge:
        #     out_node_nf = in_node_nf + 2
        # else:
        #     out_node_nf = in_node_nf + 1
        # net_config = Struct(**net_config)
        self.config = net_config

        if net_config.name == 'unio2net':
            self.unio2net = UniTransformerO2TwoUpdateGeneral(**net_config.todict())
        else:
            raise NotImplementedError
        
        self.hidden_dim = net_config.hidden_dim
        self.num_classes = ligand_atom_feature_dim

        self.node_indicator = node_indicator

        if self.node_indicator:
            emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        # just for abl
        self.res_emb = SinusoidalPosEmbWithLinear(26, emb_dim)
        self.ele_emb = SinusoidalPosEmbWithLinear(118, emb_dim)
        
        self.res_scale = nn.Parameter(torch.tensor(0.0))
        self.ele_scale = nn.Parameter(torch.tensor(0.0))
        self.v_scale = nn.Parameter(torch.tensor(0.0))

        # Training-only contrastive heads (kept so training checkpoints load
        # completely; not used by the sampling/inference paths).
        self.global_pool_att = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.use_contrastive_queue = False
        self.queue_size = 4096
        self.register_buffer("rep_queue", torch.randn(self.hidden_dim, self.queue_size))
        self.rep_queue = F.normalize(self.rep_queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(5.0))
        # atom embedding
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)
        self.ed_emb = nn.Linear(ed_feature_dim, emb_dim)
        self.center_pos_mode = center_pos_mode  # ['none', 'protein']

        self.time_emb_mode = time_emb_mode
        self.time_emb_dim = time_emb_dim
        if self.time_emb_dim > 0:
            self.time_emb_layer = TimeEmbedLayer(self.time_emb_mode, self.time_emb_dim)
        self.ligand_atom_emb = nn.Linear(
            ligand_atom_feature_dim + self.time_emb_dim, emb_dim
        )

        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, ligand_atom_feature_dim),
        )  # [hidden to 13]

        
        self.e_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, 5),
        )  # [hidden to 5]

        self.device = device
        self._edges_dict = {}
        self.condition_time = condition_time
        self.sigma1_coord = torch.tensor(sigma1_coord, dtype=torch.float32)  # coordinate sigma1, a schedule for bfn
        self.beta1 = torch.tensor(beta1, dtype=torch.float32)  # type beta, a schedule for types.
        self.use_discrete_t = use_discrete_t  # whether to use discrete t
        self.discrete_steps = discrete_steps
        self.t_min = t_min
        self.pos_init_mode = pos_init_mode
        self.destination_prediction = destination_prediction
        self.sampling_strategy = sampling_strategy


    def interdependency_modeling(
        self,
        time,
        pocket_pos,  # transform from the orginal BFN codebase
        pocket_v,  # transform from
        batch_pocket,  # index for pocket
        ed_pos,
        ed_v,
        batch_ed,
        theta_h_t,
        mu_pos_t,
        batch_ligand,  # index for ligand
        gamma_coord,
        return_all=False,  # legacy from targetdiff
        fix_x=False,
        return_logits=False,
        pocket_res=None,
        pocket_ele=None,
        onlygeneral=False
    ):
        """
        Compute output distribution parameters for p_O (x' | θ; t) (x_hat or k^(d) logits).
        Draw output_sample = x' ~ p_O (x' | θ; t).
            continuous x ~ δ(x - x_hat(θ, t))
            discrete k^(d) ~ softmax(Ψ^(d)(θ, t))_k
        Args:
            time: [node_num x batch_size, 1] := [N_ligand, 1]
            pocket_pos: [node_num x batch_size, 3] := [N_pocket, 3]
            pocket_v: [node_num x batch_size, protein_atom_feature_dim] := [N_pocket, 27]
            batch_pocket: [node_num x batch_size] := [N_pocket]
            theta_h_t: [node_num x batch_size, atom_type] := [N_ligand, 13]
            mu_pos_t: [node_num x batch_size, 3] := [N_ligand, 3]
            batch_ligand: [node_num x batch_size] := [N_ligand]
            gamma_coord: [node_num x batch_size, 1] := [N_ligand, 1]
        """
        K = self.num_classes  # ligand_atom_feature_dim

        theta_h_t = 2 * theta_h_t - 1  # from 1/K \in [0,1] to 2/K-1 \in [-1,1]

        # ---------for targetdiff-----------
        try:
            batch_size = batch_pocket.max().item() + 1
        except:
            try:
                batch_size = batch_ed.max().item() + 1
            except Exception:
                # cold start: the editor-molecule block is empty
                batch_size = batch_ligand.max().item() + 1
        # init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()
        init_ligand_v = theta_h_t
        # time embedding [simple, sin, rbf, learn]
        if self.time_emb_dim > 0:
            time_emb = self.time_emb_layer(time)
            input_ligand_feat = torch.cat([init_ligand_v, time_emb], -1)
        else:
            input_ligand_feat = init_ligand_v

        w_v_raw   = torch.sigmoid(self.v_scale)
        w_res_raw = torch.sigmoid(self.res_scale) if pocket_res is not None else 0.0
        w_ele_raw = torch.sigmoid(self.ele_scale) if pocket_ele is not None else 0.0
        
        h_pocket = 0
        w_tot = 0
        if onlygeneral:
            w_v, w_res, w_ele = w_v_raw, 0, 0
        else:
            if self.training:
                drop1_rate = 0.2
                drop2_rate = 0.2
                total_drop_rate = drop1_rate + drop2_rate  # 0.15
        
                # 生成全局随机数（per-batch 决策，更稳定；也可 per-atom，见下方注释）
                r = torch.rand(1, device=w_v_raw.device).item()
        
                # 初始化 mask（1 = keep, 0 = drop）
                keep_v, keep_res, keep_ele = 1.0, 1.0, 1.0
        
                if r < drop1_rate:
                    # ✅ 随机 drop 1 个（均匀选）
                    which = torch.randint(0, 3, (1,)).item()
                    if which == 0 and pocket_v is not None:       keep_v = 0.0
                    elif which == 1 and pocket_res is not None:   keep_res = 0.0
                    elif which == 2 and pocket_ele is not None:   keep_ele = 0.0
                    # 若某特征本就 None，自动跳过（如 no_res 时不会 drop_res）
        
                elif r < total_drop_rate:
                    # ✅ 随机 drop 2 个（等概率选组合）
                    pairs = [(0,1), (0,2), (1,2)]
                    i, j = pairs[torch.randint(0, 3, (1,)).item()]
                    if i == 0 and pocket_v is not None:       keep_v = 0.0
                    if i == 1 and pocket_res is not None:     keep_res = 0.0
                    if i == 2 and pocket_ele is not None:     keep_ele = 0.0
                    if j == 0 and pocket_v is not None:       keep_v = 0.0
                    if j == 1 and pocket_res is not None:     keep_res = 0.0
                    if j == 2 and pocket_ele is not None:     keep_ele = 0.0
        
                # 应用 mask
                w_v   = w_v_raw   * keep_v
                w_res = w_res_raw * keep_res
                w_ele = w_ele_raw * keep_ele
        
            else:
                # 推理时：全保留（或按 eval_mode 控制，见后文扩展）
                w_v, w_res, w_ele = w_v_raw, w_res_raw, w_ele_raw
            
        if pocket_v is not None:
            h_v = self.protein_atom_emb(pocket_v)  # [N_protein, self.hidden_dim - 1]
            h_pocket = h_pocket + h_v * w_v
            w_tot += w_v
        if pocket_res is not None:
            h_res = self.res_emb(pocket_res)
            h_pocket = h_pocket + h_res * w_res
            w_tot += w_res
        if pocket_ele is not None:
            h_ele = self.ele_emb(pocket_ele)
            h_pocket = h_pocket + h_ele * w_ele
            w_tot += w_ele
        h_pocket = h_pocket / w_tot
        h_ed = self.ed_emb(ed_v.float())
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)  # [N_ligand, self.hidden_dim - 1]
        
        if self.node_indicator:
            h_pocket = torch.cat(
                [h_pocket, torch.zeros(len(h_pocket), 1, device=h_pocket.device)], -1
            )  # [N_ligand, self.hidden_dim ]
            h_ed = torch.cat(
                [h_ed, torch.zeros(len(h_ed),1,device=h_ed.device)] , -1
            )
            init_ligand_h = torch.cat(
                [init_ligand_h, torch.ones(len(init_ligand_h), 1, device=h_pocket.device)], -1
            )  # [N_ligand, self.hidden_dim]

        
        h_all, pos_all, batch_all, mask_pocket, mask_ed, mask_ligand = compose_context_edmol(
            h_pocket=h_pocket,
            h_ed=h_ed,
            h_mol=init_ligand_h,
            pos_pocket=pocket_pos,
            pos_ed=ed_pos,
            pos_mol=mu_pos_t,
            batch_pocket=batch_pocket,
            batch_ed=batch_ed,
            batch_mol=batch_ligand
        )
        
        # get the context for the protein and ligand, while the ligand is h is noisy (h_t)/ pos is also the noise version. (pos_t)

        # ---------------------

        # time = 2 * time - 1
        outputs = self.unio2net(
            h_all, pos_all, mask_pocket, mask_ed, mask_ligand, batch_all, return_all=return_all, fix_x=fix_x
        )
        final_pos, final_h, final_e, final_ei = (
            outputs["x"],
            outputs["h"],
            outputs["e"],
            outputs["ei"]
        )  # shape of the pos and shape of h
        final_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand]
        final_ligand_v = self.v_inference(final_ligand_h)  # [N_ligand, 13]
        # final_e = self.e_inference(final_e) # [N_edge, 5]

        if not self.destination_prediction:
            coord_pred = (
                mu_pos_t / gamma_coord
                - torch.sqrt((1 - gamma_coord) / gamma_coord) * final_ligand_pos
            )
            coord_pred = torch.where(
                time < self.t_min, torch.zeros_like(mu_pos_t), coord_pred
            )
        else:
            coord_pred = final_ligand_pos 

        k_hat = torch.zeros_like(mu_pos_t)  
        if K == 2:
            p0_1 = torch.sigmoid(final_ligand_v)  
            p0_2 = 1 - p0_1
            p0_h = torch.cat((p0_1, p0_2), dim=-1)  
        else:
            p0_h = torch.nn.functional.softmax(final_ligand_v, dim=-1)  # [N_ligand, 13]
        """
        for discretised variable, we return p_o
        """
                    
        # e = torch.nn.functional.softmax(final_e, dim=-1)
        if return_logits:
            return coord_pred, p0_h, final_ligand_v, k_hat, final_e, final_ei
        return coord_pred, p0_h, k_hat, final_e, final_ei
    
    def loss_one_step(
        self,
        t,  # [N_ligand, 1]
        pocket_pos,
        pocket_v,
        batch_pocket,
        ed_pos,
        ed_v,
        batch_ed,
        ligand_pos,
        ligand_v,
        batch_ligand=None,
    ):
        """Primary Bayesian-flow losses plus self-supervised contrastive learning.

        The two views are the same molecule at the flow time t and at a perturbed
        time t_aug, so no teacher model is required.
        """
        K = self.num_classes
        device = ligand_v.device

        assert ligand_v.max().item() < K, f"Error: ligand_v.max().item() >= {K}"
        ligand_v_onehot = F.one_hot(ligand_v, K).float()  # [N, K]
        logit_scale = 5

        # ========== 1. Bayesian Flow: sample theta_t = (mu_pos_t, theta_h_t) ==========
        mu_coord, gamma_coord = self.continuous_var_bayesian_update(
            t, sigma1=self.sigma1_coord, x=ligand_pos
        )  # [N, 3], [N, 1]

        theta = self.discrete_var_bayesian_update(
            t, beta1=self.beta1, x=ligand_v_onehot, K=K
        )  # [N, K]

        # ========== 2. Forward pass ==========
        coord_pred, p0_h, _, final_ligand_h, *_ = self.interdependency_modeling(
            time=t,
            pocket_pos=pocket_pos,
            pocket_v=pocket_v,
            batch_pocket=batch_pocket,
            ed_pos=ed_pos,
            ed_v=ed_v,
            batch_ed=batch_ed,
            theta_h_t=theta,
            mu_pos_t=mu_coord,
            batch_ligand=batch_ligand,
            gamma_coord=gamma_coord,
            return_logits=True
        )

        # ========== 3. Primary diffusion losses (continuous + discrete) ==========
        if not self.use_discrete_t:
            closs = self.ctime4continuous_loss(
                t=t,
                sigma1=self.sigma1_coord,
                x_pred=coord_pred,
                x=ligand_pos,
                segment_ids=batch_ligand,
            )  # [B,]
            dloss = self.ctime4discrete_loss(
                t=t,
                beta1=self.beta1,
                one_hot_x=ligand_v_onehot,
                p_0=p0_h,
                K=K,
                segment_ids=batch_ligand,
            )  # [B,]
        else:
            i = (t * self.discrete_steps).int() + 1
            closs = self.dtime4continuous_loss(
                i=i,
                N=self.discrete_steps,
                sigma1=self.sigma1_coord,
                x_pred=coord_pred,
                x=ligand_pos,
                segment_ids=batch_ligand,
            )
            dloss = self.dtime4discrete_loss_prob(
                i=i,
                N=self.discrete_steps,
                beta1=self.beta1,
                one_hot_x=ligand_v_onehot,
                p_0=p0_h,
                K=K,
                segment_ids=batch_ligand,
            )

        # ========== 4. Self-supervised contrastive loss ==========

        # 4.1 molecule-level representation via attention pooling
        att_scores = self.global_pool_att(final_ligand_h)  # [N_ligand, 1]
        att_weights = scatter_softmax(att_scores, batch_ligand, dim=0)  # [N_ligand, 1]
        mol_rep = scatter_sum(final_ligand_h * att_weights, batch_ligand, dim=0)  # [Batch, hidden]
        mol_rep_proj = self.projection_head(mol_rep)
        mol_rep_proj = F.normalize(mol_rep_proj, p=2, dim=-1)

        # 4.2 positive pairs: the same molecule under a small time perturbation
        with torch.no_grad():
            t_aug = t + torch.randn_like(t) * 0.1
            t_aug = torch.clamp(t_aug, min=self.t_min, max=1.0)

            mu_coord_aug, gamma_coord_aug = self.continuous_var_bayesian_update(
                t_aug, sigma1=self.sigma1_coord, x=ligand_pos
            )
            theta_aug = self.discrete_var_bayesian_update(
                t_aug, beta1=self.beta1, x=ligand_v_onehot, K=K
            )

            _, _, _, final_ligand_h_aug, *_ = self.interdependency_modeling(
                time=t_aug,
                pocket_pos=pocket_pos,
                pocket_v=pocket_v,
                batch_pocket=batch_pocket,
                ed_pos=ed_pos,
                ed_v=ed_v,
                batch_ed=batch_ed,
                theta_h_t=theta_aug,
                mu_pos_t=mu_coord_aug,
                batch_ligand=batch_ligand,
                gamma_coord=gamma_coord_aug,
                return_logits=True
            )

            att_scores_aug = self.global_pool_att(final_ligand_h_aug)
            att_weights_aug = scatter_softmax(att_scores_aug, batch_ligand, dim=0)
            mol_rep_aug = scatter_sum(final_ligand_h_aug * att_weights_aug, batch_ligand, dim=0)
            mol_rep_aug_proj = self.projection_head(mol_rep_aug)
            mol_rep_aug_proj = F.normalize(mol_rep_aug_proj, p=2, dim=-1)

        # 4.5 InfoNCE over the two views
        if self.use_contrastive_queue:
            contrastive_loss = self.contrastive_loss_with_queue(mol_rep_proj, mol_rep_aug_proj)
        else:
            sim_matrix = torch.matmul(mol_rep_proj, mol_rep_aug_proj.T) * torch.exp(self.logit_scale)
            labels = torch.arange(mol_rep_proj.size(0), device=device).long()
            loss_s2t = F.cross_entropy(sim_matrix, labels)
            loss_t2s = F.cross_entropy(sim_matrix.T, labels)
            contrastive_loss = (loss_s2t + loss_t2s) / 2.0

        eloss = torch.zeros_like(closs)
        discretized_loss = torch.zeros_like(closs)

        return (
            closs,              # [B] continuous coordinate loss
            dloss,              # [B] discrete type loss
            eloss,              # [B] reserved
            discretized_loss,   # [B] reserved
            contrastive_loss,   # scalar self-supervised contrastive loss
        )
    def baseline_loss_one_step(
        self,
        t,  # [N_ligand, 1]
        pocket_pos,
        pocket_v,
        batch_pocket,
        ed_pos,
        ed_v,
        batch_ed,
        ligand_pos,
        ligand_v,
        batch_ligand=None,
    ):
        """Primary Bayesian-flow losses only.

        Used by the two configurations that apply no auxiliary objective: the
        no-pretraining baseline and Pretrain Only. The two differ solely in their
        starting checkpoint, so they share this function.

        Returns (continuous, discrete, reserved, reserved), which is the 4-tuple
        baseline_scratch.py unpacks.
        """
        K = self.num_classes
        device = ligand_v.device

        assert ligand_v.max().item() < K, f"Error: ligand_v.max().item() >= {K}"
        ligand_v_onehot = F.one_hot(ligand_v, K).float()  # [N, K]

        # --- Bayesian Flow: sample theta_t = (mu_pos_t, theta_h_t) ---
        mu_coord, gamma_coord = self.continuous_var_bayesian_update(
            t, sigma1=self.sigma1_coord, x=ligand_pos
        )  # [N, 3], [N, 1]
        theta = self.discrete_var_bayesian_update(
            t, beta1=self.beta1, x=ligand_v_onehot, K=K
        )  # [N, K]

        # --- Forward pass ---
        coord_pred, p0_h, _, _final_h, *_ = self.interdependency_modeling(
            time=t,
            pocket_pos=pocket_pos,
            pocket_v=pocket_v,
            batch_pocket=batch_pocket,
            ed_pos=ed_pos,
            ed_v=ed_v,
            batch_ed=batch_ed,
            theta_h_t=theta,
            mu_pos_t=mu_coord,
            batch_ligand=batch_ligand,
            gamma_coord=gamma_coord,
            return_logits=True
        )

        # --- Primary diffusion losses (continuous + discrete) ---
        if not self.use_discrete_t:
            closs = self.ctime4continuous_loss(
                t=t,
                sigma1=self.sigma1_coord,
                x_pred=coord_pred,
                x=ligand_pos,
                segment_ids=batch_ligand,
            )  # [B,]
            dloss = self.ctime4discrete_loss(
                t=t,
                beta1=self.beta1,
                one_hot_x=ligand_v_onehot,
                p_0=p0_h,
                K=K,
                segment_ids=batch_ligand,
            )  # [B,]
        else:
            i = (t * self.discrete_steps).int() + 1
            closs = self.dtime4continuous_loss(
                i=i,
                N=self.discrete_steps,
                sigma1=self.sigma1_coord,
                x_pred=coord_pred,
                x=ligand_pos,
                segment_ids=batch_ligand,
            )
            dloss = self.dtime4discrete_loss_prob(
                i=i,
                N=self.discrete_steps,
                beta1=self.beta1,
                one_hot_x=ligand_v_onehot,
                p_0=p0_h,
                K=K,
                segment_ids=batch_ligand,
            )

        eloss = torch.zeros_like(closs)
        discretized_loss = torch.zeros_like(closs)

        return (
            closs,              # [B] continuous coordinate loss
            dloss,              # [B] discrete type loss
            eloss,              # [B] reserved
            discretized_loss,   # [B] reserved
        )

    def contrastive_loss_with_queue(self, z_a, z_b):
        """InfoNCE with a FIFO memory bank of past representations.

        Positive pairs are (z_a[i], z_b[i]); negatives are the other views in the
        batch plus the queued representations from previous steps. z_b is enqueued
        (detached) after the loss is computed. Selected by `use_contrastive_queue`;
        the plain in-batch path is used otherwise.
        """
        z_a = F.normalize(z_a, p=2, dim=-1)
        z_b = F.normalize(z_b, p=2, dim=-1)
        queue = self.rep_queue.t().detach()          # [queue_size, dim]
        scale = torch.exp(self.logit_scale)

        def _nce(anchor, positive):
            negatives = torch.cat([positive, queue], dim=0)   # [B + queue_size, dim]
            logits = anchor @ negatives.t() * scale
            labels = torch.arange(anchor.size(0), device=anchor.device)
            return F.cross_entropy(logits, labels)

        loss = 0.5 * (_nce(z_a, z_b) + _nce(z_b, z_a))

        with torch.no_grad():
            n = z_b.size(0)
            ptr = int(self.queue_ptr)
            if n >= self.queue_size:
                self.rep_queue.copy_(z_b[-self.queue_size:].t())
                ptr = 0
            else:
                end = ptr + n
                if end <= self.queue_size:
                    self.rep_queue[:, ptr:end] = z_b.t()
                else:
                    head = self.queue_size - ptr
                    self.rep_queue[:, ptr:] = z_b[:head].t()
                    self.rep_queue[:, :end - self.queue_size] = z_b[head:].t()
                ptr = end % self.queue_size
            self.queue_ptr[0] = ptr
        return loss

    def constrain_loss_one_step(
        self,
        rep_model,
        t,  # [N_ligand, 1]
        pocket_pos,
        pocket_v,
        batch_pocket,
        ed_pos,
        ed_v,
        batch_ed,
        ligand_pos,
        ligand_v,
        batch_ligand=None,
        pkt_model=None,
        pocket_res=None,
        pocket_ele=None,
        onlygeneral=False
    ):
        from torch_scatter import scatter_mean

        K = self.num_classes
        device = ligand_v.device

        # --- 0. Preprocess ligand atom types ---
        assert ligand_v.max().item() < K, f"Error: ligand_v.max() = {ligand_v.max().item()} >= num_classes {K}"
        ligand_v_onehot = F.one_hot(ligand_v, K).float()  # [N, K]
        logit_scale = 5
        # --- 1. Bayesian Flow: Sample theta_t = (mu_pos_t, theta_h_t) from p_F(theta|x; t) ---
        mu_coord, gamma_coord = self.continuous_var_bayesian_update(
            t, sigma1=self.sigma1_coord, x=ligand_pos
        )  # [N, 3], [N, 1]
        theta = self.discrete_var_bayesian_update(
            t, beta1=self.beta1, x=ligand_v_onehot, K=K
        )  # [N, K]

        # --- 2. Forward passes (teacher models in no_grad) ---
        with torch.no_grad():
            # ED-conditioned teacher
            coord_pred_ed, p0_h_ed, logits_ed, h_ed, *_ = rep_model.interdependency_modeling(
                time=t,
                pocket_pos=torch.empty((0, 3), device=device),
                pocket_v=torch.empty((0, 11), device=device),
                batch_pocket=torch.empty((0,), device=device),
                ed_pos=ed_pos,
                ed_v=ed_v,
                batch_ed=batch_ed,
                theta_h_t=theta,
                mu_pos_t=mu_coord,
                batch_ligand=batch_ligand,
                gamma_coord=gamma_coord,
                fix_x=False,
                return_logits=True,
                pocket_res=None,
                pocket_ele=None,
                onlygeneral=True
            )

            # Optional Pocket-conditioned teacher
            if pkt_model is not None:
                coord_pred_pkt, p0_h_pkt, logits_pkt, h_pkt, *_ = pkt_model.interdependency_modeling(
                    time=t,
                    pocket_pos=pocket_pos,
                    pocket_v=pocket_v,
                    batch_pocket=batch_pocket,
                    ed_pos=torch.empty((0, 3), device=device),
                    ed_v=torch.empty((0, 4), device=device),
                    batch_ed=torch.empty((0,), device=device),
                    theta_h_t=theta,
                    mu_pos_t=mu_coord,
                    batch_ligand=batch_ligand,
                    gamma_coord=gamma_coord,
                    fix_x=False,
                    return_logits=True,
                    pocket_res=pocket_res,
                    pocket_ele=pocket_ele,
                    onlygeneral=True
                )
            else:
                coord_pred_pkt = p0_h_pkt = logits_pkt = None

        # Student forward (current model, Pocket-conditioned)
        coord_pred, p0_h, logits_self, h_self, *_ = self.interdependency_modeling(
            time=t,
            pocket_pos=pocket_pos,
            pocket_v=pocket_v,
            batch_pocket=batch_pocket,
            ed_pos=torch.empty((0, 3), device=device),
            ed_v=torch.empty((0, 4), device=device),
            batch_ed=torch.empty((0,), device=device),
            theta_h_t=theta,
            mu_pos_t=mu_coord,
            batch_ligand=batch_ligand,
            gamma_coord=gamma_coord,
            fix_x=False,
            return_logits=True,
            pocket_res=pocket_res,
            pocket_ele=pocket_ele,
            onlygeneral=onlygeneral
        )

        # --- 3. Alignment losses: ED -> Student (output space) ---
        # Coordinates and atom-type distributions are what the two paths predict;
        # this is the "d(z_ED, z_lig)" of the Fig. 1c schematic.
        coord_align_loss = torch.mean(torch.sum((coord_pred_ed - coord_pred) ** 2, dim=-1))

        p_ed = p0_h_ed + 1e-8
        p_self = p0_h + 1e-8
        log_p_ed = torch.log(p_ed)
        log_p_self = torch.log(p_self)
        KL_ed2self = (p_ed * (log_p_ed - log_p_self)).sum(dim=-1)
        KL_self2ed = (p_self * (log_p_self - log_p_ed)).sum(dim=-1)
        type_align_loss = 0.5 * (KL_ed2self + KL_self2ed).mean()

        # --- 4. Alignment losses: PKT -> Student (if pkt_model provided) ---
        coord_align_pkt_loss = torch.tensor(0.0, device=device)
        type_align_pkt_loss = torch.tensor(0.0, device=device)

        # --- 5. Time-dependent weighting: focus on mid-training (t ~ 0.5) ---
        weight_t = torch.sigmoid(10 * (t - 0.5))  # [N, 1] or scalar if t is scalar

        coord_align_loss = coord_align_loss * weight_t
        type_align_loss = type_align_loss * weight_t

        # --- 6. Contrastive learning: cross-view InfoNCE, ED teacher vs student ---
        from torch_scatter import scatter_softmax, scatter_sum
        att_scores = self.global_pool_att(h_self)  # [N_total, 1]
        att_weights = scatter_softmax(att_scores, batch_ligand, dim=0)  # [N_total, 1]

        global_rep = scatter_sum(h_self * att_weights, batch_ligand, dim=0)  # [Batch, Hidden]
        mol_rep = self.projection_head(global_rep)  # [Batch, Hidden]
        mol_rep = F.normalize(mol_rep, p=2, dim=-1)

        with torch.no_grad():
            t_att_scores = self.global_pool_att(h_ed)
            t_att_weights = scatter_softmax(t_att_scores, batch_ligand, dim=0)
            t_global_rep = scatter_sum(h_ed * t_att_weights, batch_ligand, dim=0)
            t_mol_rep = self.projection_head(t_global_rep)
            t_mol_rep = F.normalize(t_mol_rep, p=2, dim=-1)

        # Cross-view Contrastive Loss (supports Memory Bank + All-Gather)
        if self.use_contrastive_queue:
            contrastive_loss = self.contrastive_loss_with_queue(mol_rep, t_mol_rep)
        else:
            sim_matrix_st = torch.matmul(mol_rep, t_mol_rep.T) * torch.exp(self.logit_scale)
            labels = torch.arange(mol_rep.size(0), device=mol_rep.device).long()
            loss_s2t = F.cross_entropy(sim_matrix_st, labels)
            loss_t2s = F.cross_entropy(sim_matrix_st.T, labels)
            contrastive_loss = (loss_s2t + loss_t2s) / 2.0

        # --- 7. Primary diffusion losses (continuous + discrete) ---
        if not self.use_discrete_t:
            closs = self.ctime4continuous_loss(
                t=t, sigma1=self.sigma1_coord,
                x_pred=coord_pred, x=ligand_pos,
                segment_ids=batch_ligand
            )  # [B]
            dloss = self.ctime4discrete_loss(
                t=t, beta1=self.beta1,
                one_hot_x=ligand_v_onehot, p_0=p0_h, K=K,
                segment_ids=batch_ligand
            )  # [B]
        else:
            i = (t * self.discrete_steps).int() + 1
            closs = self.dtime4continuous_loss(
                i=i, N=self.discrete_steps,
                sigma1=self.sigma1_coord,
                x_pred=coord_pred, x=ligand_pos,
                segment_ids=batch_ligand
            )
            dloss = self.dtime4discrete_loss_prob(
                i=i, N=self.discrete_steps,
                beta1=self.beta1,
                one_hot_x=ligand_v_onehot, p_0=p0_h, K=K,
                segment_ids=batch_ligand
            )

        eloss = torch.zeros_like(closs)
        discretized_loss = torch.zeros_like(closs)

        # --- Return all loss components ---
        return (
            closs,                   # [B] main continuous loss
            dloss,                   # [B] main discrete loss
            eloss,                   # [B] (reserved)
            discretized_loss,        # [B] (reserved)
            coord_align_loss,        # scalar, ED -> self
            type_align_loss,         # scalar, ED -> self
            coord_align_pkt_loss,    # scalar, PKT -> self (0 if pkt_model=None)
            type_align_pkt_loss,     # scalar, PKT -> self (0 if pkt_model=None)
            contrastive_loss         # scalar, cross-view contrast on molecular reps
        )
    def sample(
        self,
        pocket_pos,
        pocket_v,
        batch_pocket,
        ed_pos,
        ed_v,
        batch_ed,
        batch_ligand,
        n_nodes,  # B
        sample_steps=1000,
        desc='',
        ligand_pos=None,  # for debug
        pocket_res=None,
        pocket_ele=None,
        fix_atom_types=True,  # if False, only fix coordinates, allow random atom types
        return_traj=False,    # reserved: full trajectory dumps are not implemented
    ):
        """
        The function implements a sampling procedure for BFN
        Args:
            t: should be a scalar tensor or the shape of [node_num x batch_size, 1] := [N, 1]
            protein_pos: [node_num x batch_size, 3] := [N_protein, 3]
            protein_v: [N_protein, protein_atom_feature_dim] := [N_protein, 27]
            batch_ligand / protein: segment_ids for ligand / protein
        """
        if return_traj:
            raise NotImplementedError('return_traj=True is not implemented')

        K = self.num_classes


        # TODO test
        if self.pos_init_mode == 'zero':
            mu_pos_t = torch.zeros((n_nodes, 3),device=self.device)
        elif self.pos_init_mode == 'randn':
            mu_pos_t = torch.randn((n_nodes, 3),device=self.device)

        theta_h_t = (
            torch.ones((n_nodes, K)).to(self.device) / K
        )  # [N, K] discrete prior (uniform 1/K)
        ro_coord = 1
        ro_charge = 1

        sample_traj = []
        theta_traj = []
        y_traj = []

        # TODO: debug
        # print(batch_ligand)
        mu_pos_t = mu_pos_t[batch_ligand]
        theta_h_t = theta_h_t[batch_ligand]
        for i in trange(1, sample_steps + 1, desc=f'{desc}'):
            t = torch.ones((n_nodes, 1),device=self.device) * (i - 1) / sample_steps
            if not self.use_discrete_t and not self.destination_prediction:
                t = torch.clamp(t, min=self.t_min)

            t = t[batch_ligand]
            # Eq.(84): γ(t) = σ1^(2t)
            gamma_coord = 1 - torch.pow(self.sigma1_coord, 2 * t)

            coord_pred, p0_h_pred, k_hat, _, _ = self.interdependency_modeling(
                time=t,
                pocket_pos=pocket_pos,
                pocket_v=pocket_v,
                batch_pocket=batch_pocket,
                ed_pos=ed_pos,
                ed_v=ed_v,
                batch_ed=batch_ed,
                batch_ligand=batch_ligand,
                theta_h_t=theta_h_t,
                mu_pos_t=mu_pos_t,  # no guidance
                gamma_coord=gamma_coord,
                pocket_res=pocket_res,
                pocket_ele=pocket_ele
            )

            # maintain theta_traj
            theta_traj.append((mu_pos_t, theta_h_t, k_hat))
            # TODO delete the following condition
            if not torch.all(p0_h_pred.isfinite()):
                p0_h_pred = torch.where(
                    p0_h_pred.isfinite(), p0_h_pred, torch.zeros_like(p0_h_pred)
                )
                logging.warn("p0_h_pred is not finite")

            # ========== 数值稳定（融合成单个 op） ==========
            p0_h_pred = torch.clamp(p0_h_pred, min=1e-6)
            sample_pred = torch.distributions.Categorical(p0_h_pred).sample()
            sample_pred = F.one_hot(sample_pred, num_classes=K)



            # Eq.(95) α_i = σ1 ** (−2i/n) * (1 − σ1 ** (2/n))
            alpha_coord = torch.pow(self.sigma1_coord, -2 * i / sample_steps) * (
                1 - torch.pow(self.sigma1_coord, 2 / sample_steps)
            )
            # Eq.(86): p_S (y | x'; α) = N(y | x', 1/α*I)
            # (meaning that y ∼ p_R(· | θ_{i−1}; t_{i−1}, α_i) — see Eq. 4)
            y_coord = coord_pred + torch.randn_like(coord_pred) * torch.sqrt(
                1 / alpha_coord
            )
            # Algorithm (9)
            # for discrete, y \in R^K, while data \in {1,K}, cf. Eq.(141)
            # where e_k is network output p0_h_pred
            # Eq.(193): α_i = β(1) * (2i − 1) / n**2
            alpha_h = self.beta1 * (2 * i - 1) / (sample_steps**2)
            k = torch.distributions.Categorical(probs=p0_h_pred).sample()


            
            e_k = F.one_hot(k, num_classes=K).float()
            # y ~ N(α(Ke_k − 1) , αKI)
            mean = alpha_h * (K * e_k - 1)
            var = alpha_h * K
            std = torch.full_like(mean, fill_value=var).sqrt()
            y_h = mean + std * torch.randn_like(e_k)
            y_traj.append((y_coord, y_h, k_hat))

            if self.sampling_strategy == "vanilla":
                sample_traj.append((coord_pred, sample_pred, k_hat))

                # 4. Bayesian update input parameters θ_i = h(θ_{i−1}, y) for p_I(x | θ_i; t_i)
                # for continuous, Eq.(49): ρi = ρ_{i−1} + α,
                # Eq.(50): μi = [μ_{i−1}ρ_{i−1} + yα] / ρi
                mu_pos_t = (ro_coord * mu_pos_t + alpha_coord * y_coord) / (
                    ro_coord + alpha_coord
                )
                ro_coord = ro_coord + alpha_coord

                # for discrete, Eq.(171): h(θi−1, y, α) := e**y * θ_{i−1} / \sum_{k=1}^K e**y_k (θ_{i−1})_k
                theta_prime = torch.exp(y_h) * theta_h_t  # e^y * θ_{i−1}
                theta_h_t = theta_prime / theta_prime.sum(dim=-1, keepdim=True)

            elif "end_back" in self.sampling_strategy:
                t = torch.ones((n_nodes, 1)).to(self.device) * i  / sample_steps #next time step
                t = t[batch_ligand]
                if self.sampling_strategy == "end_back":
                    theta_h_t = self.discrete_var_bayesian_update(t, beta1=self.beta1, x=sample_pred, K=K)
                elif self.sampling_strategy == "end_back_pmf":
                    theta_h_t = self.discrete_var_bayesian_update(t, beta1=self.beta1, x=p0_h_pred, K=K)
                elif self.sampling_strategy == "end_back_mode":
                    p0_mode = torch.argmax(p0_h_pred, dim=-1)
                    mode_pred = F.one_hot(p0_mode, num_classes=K).float()
                    theta_h_t = self.discrete_var_bayesian_update(t, beta1=self.beta1, x=mode_pred, K=K)
                else:
                    raise NotImplementedError(f"sampling strategy {self.sampling_strategy} not implemented")
                mu_pos_t = self.continuous_var_bayesian_update(t, sigma1=self.sigma1_coord, x=coord_pred)[0]
                sample_traj.append((coord_pred, sample_pred, k_hat))
            else:
                raise NotImplementedError
            

        # 5. Compute final output distribution parameters for p_O (x' | θ; t)
        mu_pos_final, p0_h_final, k_hat_final, final_e, final_ei = self.interdependency_modeling(
            time=torch.ones((n_nodes, 1),device=self.device)[batch_ligand],
            pocket_pos=pocket_pos,
            pocket_v=pocket_v,
            batch_pocket=batch_pocket,
            ed_pos=ed_pos,
            ed_v=ed_v,
            batch_ed=batch_ed,
            batch_ligand=batch_ligand,
            theta_h_t=theta_h_t,
            mu_pos_t=mu_pos_t,
            gamma_coord=1 - self.sigma1_coord**2,  # γ(t) = 1 − (σ1**2) ** t
            pocket_res=pocket_res,
            pocket_ele=pocket_ele
        )
        # TODO delete the following condition
        if not torch.all(p0_h_final.isfinite()):
            p0_h_final = torch.where(
                p0_h_final.isfinite(), p0_h_final, torch.zeros_like(p0_h_final)
            )
            logging.warn("p0_h_pred is not finite")
        p0_h_final = torch.clamp(p0_h_final, min=1e-6)
        theta_traj.append((mu_pos_final, p0_h_final, k_hat_final))

        # 6. Draw final sample from p_O (· | θ_n, 1)
        # Update: directly take the mode of categorical distribution (as done in BFN paper)
        k_final = p0_h_final

        sample_traj.append((mu_pos_final, k_final, k_hat_final))

        return theta_traj, sample_traj, y_traj, final_e, final_ei

    def sample_by_seed(
        self,
        pocket_pos,
        pocket_v,
        batch_pocket,
        ed_pos,
        ed_v,
        batch_ed,
        batch_ligand,
        n_nodes,  # total number of nodes (for time t creation)
        fixed_atom_mask,  # [N_ligand_total], bool
        fixed_atom_pos,   # [N_fixed, 3]
        fixed_atom_types, # [N_fixed] (class indices) or [N_fixed, K] (one-hot)
        sample_steps=1000,
        desc='',
        ligand_pos=None,  # for debug
        pocket_res=None,
        pocket_ele=None,
        fix_atom_types=True,
        return_traj=False,    # reserved: full trajectory dumps are not implemented
        given_traj=None,      # (mu_pos [T+1, N, 3], theta_h [T+1, N, K]) from a real run
    ):
        """
        Conditional sampling with fixed (seeded) atoms by constructing pseudo forward trajectories
        for fixed atoms that are consistent with BFN's training noise schedule.
        
        When fix_atom_types=False, only coordinates are fixed; atom types of seeded atoms
        are generated freely by the model.

        given_traj supplies the intermediate states of the fixed atoms from an actual free
        generation run (the trajectory `sample` already accumulates). The default
        constructs them analytically from the final answer, mu(t) = gamma(t) * x_true,
        which is a state the network never visits and which pulls the trajectory out of
        distribution. Passing the real path keeps the conditioning in distribution.
        """
        if return_traj:
            raise NotImplementedError('return_traj=True is not implemented')
        K = self.num_classes
        device = self.device
    
        # --- Validate input shapes ---
        N_ligand_total = batch_ligand.shape[0]
        assert fixed_atom_mask.dtype == torch.bool
        assert fixed_atom_mask.shape[0] == N_ligand_total
        n_fixed = fixed_atom_mask.sum().item()
        n_free = (~fixed_atom_mask).sum().item()
    
        assert fixed_atom_pos.shape[0] == n_fixed, (fixed_atom_pos.shape[0], n_fixed)
        if fixed_atom_types.dim() == 2:
            assert fixed_atom_types.shape == (n_fixed, K)
            fixed_atom_types_indices = fixed_atom_types.argmax(dim=-1)  # for discrete prior
            fixed_atom_types_onehot = fixed_atom_types.float()
        elif fixed_atom_types.dim() == 1:
            assert fixed_atom_types.shape[0] == n_fixed
            assert fixed_atom_types.max() < K and fixed_atom_types.min() >= 0
            fixed_atom_types_indices = fixed_atom_types.long()
            fixed_atom_types_onehot = F.one_hot(fixed_atom_types_indices, num_classes=K).float()
        else:
            raise ValueError("fixed_atom_types must be 1D (indices) or 2D (one-hot).")
    
        # --- Expand fixed values to full ligand size (for masking) ---
        fixed_atom_pos_full = torch.zeros(N_ligand_total, 3, device=device)
        fixed_atom_pos_full[fixed_atom_mask] = fixed_atom_pos
    
        fixed_atom_types_full = torch.zeros(N_ligand_total, K, device=device)
        fixed_atom_types_full[fixed_atom_mask] = fixed_atom_types_onehot
    
        fixed_atom_indices_full = torch.full((N_ligand_total,), -1, dtype=torch.long, device=device)
        fixed_atom_indices_full[fixed_atom_mask] = fixed_atom_types_indices
    
        # --- Fixed-atom states over time ---
        # Either the caller's real trajectory, or an analytic pseudo path from the final
        # answer. Shape: [sample_steps + 1, N_ligand_total, ...]
        use_real = given_traj is not None
        if use_real:
            gm, gt = given_traj
            pseudo_mu_pos = gm.to(device).float()
            pseudo_theta_h = gt.to(device).float()
            if pseudo_mu_pos.shape[0] != sample_steps + 1:
                raise ValueError(f'given_traj has {pseudo_mu_pos.shape[0]} frames, '
                                 f'need {sample_steps + 1}')
            if pseudo_mu_pos.shape[1] != N_ligand_total:
                raise ValueError('given_traj atom count does not match the batch')
            print(f'[sample_by_seed] REAL trajectory for {int(n_fixed)} fixed atoms '
                  f'({pseudo_mu_pos.shape[0]} frames)', flush=True)
        else:
            pseudo_mu_pos = torch.zeros(sample_steps + 1, N_ligand_total, 3, device=device)
            pseudo_theta_h = torch.ones(sample_steps + 1, N_ligand_total, K,
                                        device=device) / K
    
        # Time points: t_i = i / sample_steps for i in 0..sample_steps
        t_all = torch.linspace(0, 1, steps=sample_steps + 1, device=device)  # [T+1]
        if not self.use_discrete_t and not self.destination_prediction:
            t_all = torch.clamp(t_all, min=self.t_min)
    
        for i, t_val in enumerate(t_all):
            t = t_val.repeat(N_ligand_total, 1)  # [N_ligand, 1]
    
            # Continuous: mu(t) = gamma(t) * x_true, gamma(t) = 1 - sigma1^(2t)
            if use_real:
                continue          # keep the harvested trajectory verbatim
            gamma_coord = 1 - torch.pow(self.sigma1_coord, 2 * t)  # [N, 1]
            pseudo_mu_pos[i] = gamma_coord * fixed_atom_pos_full  # [N, 3]
    
            # Discrete: p(k|t) ∝ exp(beta(t) * 1_{k=k_true}), beta(t) = beta1 * t
            beta_t = self.beta1 * t.squeeze(-1)  # [N]
            log_p = torch.full((N_ligand_total, K), -1e6, device=device)
            valid_mask = (fixed_atom_indices_full >= 0)
            log_p[valid_mask, fixed_atom_indices_full[valid_mask]] = beta_t[valid_mask]
            pseudo_theta_h[i] = torch.softmax(log_p, dim=-1)  # [N, K]
    
        # --- Initialize belief for free atoms (fixed atoms will be overwritten by pseudo) ---
        if self.pos_init_mode == 'zero':
            mu_pos_t = torch.zeros(N_ligand_total, 3, device=device)
        elif self.pos_init_mode == 'randn':
            mu_pos_t = torch.randn(N_ligand_total, 3, device=device)
        else:
            raise ValueError(f"Unknown pos_init_mode: {self.pos_init_mode}")
    
        theta_h_t = torch.ones(N_ligand_total, K, device=device) / K  # uniform prior
    
        # Overwrite initial belief for fixed atoms with the trajectory state at t=0, so the
        # sampler genuinely starts from that probability rather than from zeros/uniform.
        if fixed_atom_mask.any():
            mu_pos_t = torch.where(fixed_atom_mask.unsqueeze(-1), pseudo_mu_pos[0], mu_pos_t)
            theta_h_t = torch.where(fixed_atom_mask.unsqueeze(-1), pseudo_theta_h[0], theta_h_t)

        ro_coord = 1.0  # scalar precision accumulator
        sample_traj = []
        theta_traj = []
        y_traj = []
    
        free_mask = ~fixed_atom_mask  # [N_ligand_total]
    
        for i in trange(1, sample_steps + 1, desc=f'{desc}'):
            # Current time: t = (i-1)/sample_steps
            t_val = (i - 1) / sample_steps
            if not self.use_discrete_t and not self.destination_prediction:
                t_val = max(t_val, self.t_min)
            t = torch.full((N_ligand_total, 1), t_val, device=device)
    
            gamma_coord = 1 - torch.pow(self.sigma1_coord, 2 * t)  # [N, 1]
    
            # --- Prepare input to model: fixed atoms use pseudo trajectory ---
            if i != 0:
                mu_pos_input = torch.where(fixed_atom_mask.unsqueeze(-1), pseudo_mu_pos[i - 1], mu_pos_t)
                if fix_atom_types:
                    theta_h_input = torch.where(fixed_atom_mask.unsqueeze(-1), pseudo_theta_h[i - 1], theta_h_t)
                else:
                    theta_h_input = theta_h_t
            else:
                theta_h_input = theta_h_t
    
            coord_pred, p0_h_pred, k_hat,_,_ = self.interdependency_modeling(
                time=t,
                pocket_pos=pocket_pos,
                pocket_v=pocket_v,
                batch_pocket=batch_pocket,
                ed_pos=ed_pos,
                ed_v=ed_v,
                batch_ed=batch_ed,
                batch_ligand=batch_ligand,
                theta_h_t=theta_h_input,
                mu_pos_t=mu_pos_input,
                gamma_coord=gamma_coord,
                pocket_res=None,
                pocket_ele=None
            )
    
            theta_traj.append((mu_pos_t.clone(), theta_h_t.clone(), k_hat))
    
            if not torch.all(p0_h_pred.isfinite()):
                p0_h_pred = torch.where(p0_h_pred.isfinite(), p0_h_pred, torch.zeros_like(p0_h_pred))
                logging.warn("p0_h_pred is not finite")
    
            # --- Sample for trajectory (mask fixed atoms) ---
            p0_h_pred_clamped = torch.clamp(p0_h_pred, min=1e-6)
            sample_indices = torch.distributions.Categorical(p0_h_pred_clamped).sample()  # [N]
            if fix_atom_types:
                sample_indices[fixed_atom_mask] = fixed_atom_indices_full[fixed_atom_mask]
            sample_onehot = F.one_hot(sample_indices, num_classes=K).float()
    
            # --- Generate y (sender noise) ---
            alpha_coord = torch.pow(self.sigma1_coord, -2 * i / sample_steps) * (
                1 - torch.pow(self.sigma1_coord, 2 / sample_steps)
            )  # scalar
    
            # Continuous y: y = coord_pred + N(0, 1/alpha)
            noise_coord = torch.randn_like(coord_pred) * torch.sqrt(1.0 / alpha_coord)
            noise_coord[fixed_atom_mask] = 0.0  # no noise on fixed atoms
            y_coord = coord_pred + noise_coord
            y_coord[fixed_atom_mask] = fixed_atom_pos_full[fixed_atom_mask]  # enforce exact
    
            # Discrete y
            alpha_h = self.beta1 * (2 * i - 1) / (sample_steps ** 2)  # scalar
            k_sampled = torch.distributions.Categorical(probs=p0_h_pred_clamped).sample()
            e_k = F.one_hot(k_sampled, num_classes=K).float()
            mean_h = alpha_h * (K * e_k - 1)
            std_h = torch.sqrt(alpha_h * K)
            noise_h = torch.randn_like(e_k) * std_h
            y_h = mean_h + noise_h
    
            y_traj.append((y_coord, y_h, k_hat))
    
            # --- Bayesian update for FREE atoms only ---
            if self.sampling_strategy == "vanilla":
                if free_mask.any() or not fix_atom_types:
                    # Continuous update (only free atoms move)
                    mu_pos_t_free = (ro_coord * mu_pos_t + alpha_coord * y_coord) / (ro_coord + alpha_coord)
                    # Discrete update (all atoms if fix_atom_types=False)
                    theta_prime = torch.exp(y_h) * theta_h_t  # [N, K]
                    theta_h_t_free = theta_prime / theta_prime.sum(dim=-1, keepdim=True)
    
                    mu_pos_t = torch.where(fixed_atom_mask.unsqueeze(-1), mu_pos_t, mu_pos_t_free)
                    if fix_atom_types:
                        theta_h_t = torch.where(fixed_atom_mask.unsqueeze(-1), theta_h_t, theta_h_t_free)
                    else:
                        theta_h_t = theta_h_t_free
    
                ro_coord = ro_coord + alpha_coord  # scalar update
    
                # Fixed atoms' belief follows the supplied path exactly; only the free
                # atoms carry the model's own posterior.
                if fixed_atom_mask.any():
                    j = min(i, pseudo_mu_pos.shape[0] - 1)
                    mu_pos_t = torch.where(fixed_atom_mask.unsqueeze(-1), pseudo_mu_pos[j], mu_pos_t)
                    theta_h_t = torch.where(fixed_atom_mask.unsqueeze(-1), pseudo_theta_h[j], theta_h_t)

                sample_traj.append((coord_pred, sample_onehot, k_hat))
    
            elif "end_back" in self.sampling_strategy:
                t_next = torch.full((N_ligand_total, 1), i / sample_steps, device=device)
                if not self.use_discrete_t:
                    t_next = torch.clamp(t_next, min=self.t_min)
    
                if free_mask.any() or not fix_atom_types:
                    # Discrete update
                    if self.sampling_strategy == "end_back":
                        x_discrete = sample_onehot
                    elif self.sampling_strategy == "end_back_pmf":
                        x_discrete = p0_h_pred_clamped
                    elif self.sampling_strategy == "end_back_mode":
                        mode_idx = torch.argmax(p0_h_pred_clamped, dim=-1)
                        x_discrete = F.one_hot(mode_idx, num_classes=K).float()
                    else:
                        raise NotImplementedError
    
                    if fix_atom_types:
                        x_discrete = x_discrete[free_mask]
                        t_next_discrete = t_next[free_mask]
                    else:
                        t_next_discrete = t_next
    
                    theta_h_t_free = self.discrete_var_bayesian_update(
                        t_next_discrete, beta1=self.beta1, x=x_discrete, K=K
                    )  # [N_free or N_total, K]
    
                    # Continuous update for free atoms
                    mu_pos_t_free, _ = self.continuous_var_bayesian_update(
                        t_next[free_mask], sigma1=self.sigma1_coord, x=coord_pred[free_mask]
                    )  # [N_free, 3]
    
                    # Reconstruct full tensors
                    theta_h_t_new = theta_h_t.clone()
                    mu_pos_t_new = mu_pos_t.clone()
                    if fix_atom_types:
                        theta_h_t_new[free_mask] = theta_h_t_free
                    else:
                        theta_h_t_new = theta_h_t_free
                    mu_pos_t_new[free_mask] = mu_pos_t_free
    
                    theta_h_t = theta_h_t_new
                    mu_pos_t = mu_pos_t_new
    
                # Fixed atoms' belief follows the supplied path exactly; only the free
                # atoms carry the model's own posterior.
                if fixed_atom_mask.any():
                    j = min(i, pseudo_mu_pos.shape[0] - 1)
                    mu_pos_t = torch.where(fixed_atom_mask.unsqueeze(-1), pseudo_mu_pos[j], mu_pos_t)
                    theta_h_t = torch.where(fixed_atom_mask.unsqueeze(-1), pseudo_theta_h[j], theta_h_t)

                sample_traj.append((coord_pred, sample_onehot, k_hat))
    
            else:
                raise NotImplementedError
    
        # --- Final prediction ---
        t_final = torch.ones(N_ligand_total, 1, device=device)
        if not self.use_discrete_t:
            t_final = torch.clamp(t_final, min=self.t_min)
        gamma_final = 1 - self.sigma1_coord ** 2
        
        mu_pos_input_final = torch.where(fixed_atom_mask.unsqueeze(-1), pseudo_mu_pos[-1], mu_pos_t)
        if fix_atom_types:
            theta_h_input_final = torch.where(fixed_atom_mask.unsqueeze(-1), pseudo_theta_h[-1], theta_h_t)
        else:
            theta_h_input_final = theta_h_t
    
        mu_pos_final, p0_h_final, k_hat_final, final_e, final_ei = self.interdependency_modeling(
            time=t_final,
            pocket_pos=pocket_pos,
            pocket_v=pocket_v,
            batch_pocket=batch_pocket,
            ed_pos=ed_pos,
            ed_v=ed_v,
            batch_ed=batch_ed,
            batch_ligand=batch_ligand,
            theta_h_t=theta_h_input_final,
            mu_pos_t=mu_pos_input_final,
            gamma_coord=gamma_final,
            pocket_res=None,
            pocket_ele=None
        )
    
        if not torch.all(p0_h_final.isfinite()):
            p0_h_final = torch.where(p0_h_final.isfinite(), p0_h_final, torch.zeros_like(p0_h_final))
            logging.warn("p0_h_final is not finite")
        p0_h_final = torch.clamp(p0_h_final, min=1e-6)
    

    
        # Final output: enforce fixed atoms
        mu_pos_final = torch.where(fixed_atom_mask.unsqueeze(-1), fixed_atom_pos_full, mu_pos_final)
        if fix_atom_types:
            p0_h_final = torch.where(fixed_atom_mask.unsqueeze(-1), fixed_atom_types_full, p0_h_final)
        theta_traj.append((mu_pos_final, p0_h_final, k_hat_final))    
        sample_traj.append((mu_pos_final, p0_h_final, k_hat_final))
    
        return theta_traj, sample_traj, y_traj, final_e, final_ei
