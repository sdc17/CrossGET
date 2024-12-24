import torch
import torch.nn.functional as F
from torch import nn

import math

class TokenReduction(nn.Module):
    def __init__(self, batch_first=False, protected=None):
        super().__init__()
        self.batch_first=batch_first
        self.protected=protected

    def match(self, metric, r, class_token=True, eot_token=None, query_token=None):
        if self.protected is not None:
            protected = self.protected
        else:
            protected = 1
            if class_token:
                protected += 1
            if eot_token is not None:
                protected += 1
        n, t, c = metric.shape
        r = min(r, (t - protected) // 2)
        with torch.no_grad():
            metric = metric / metric.norm(dim=-1, keepdim=True)
            similarity = metric @ metric.transpose(-1, -2) 
            similarity += torch.empty_like(similarity).fill_(-math.inf).tril_().triu_()
            if class_token:
                similarity[:, 0, :] = similarity[:, :, 0] = -math.inf
                similarity[:, -1, :] = similarity[:, :, -1] = -math.inf
            else:
                similarity[:, 0, :] = similarity[:, :, 0] = -math.inf
            if eot_token is not None:
                eot_mask = torch.full((n, similarity.shape[-1]), -math.inf, device=similarity.device, dtype=similarity.dtype)
                similarity[torch.arange(n), eot_token, :] = similarity[torch.arange(n), :, eot_token] = eot_mask
            edge_idx = torch.max(similarity, dim=-1, keepdim=True)[0].argsort(dim=1, descending=True).expand(-1, -1, t)
            mask = similarity.scatter(1, edge_idx, torch.empty_like(similarity).fill_(-math.inf).tril_())
            mask = similarity.scatter(-1, edge_idx.transpose(-1, -2), mask)

            importance = torch.mean(query_token[..., :-1], dim=-1, keepdim=True) + query_token[..., -1, None] 

            edge_idx = (importance - torch.max(similarity + mask, dim=-1, keepdim=True)[0]).argsort(dim=1, descending=False)
            src_idx, dst_all_idx = edge_idx[..., :r, :], edge_idx[..., r:, :]  

            similarity = similarity.gather(dim=1, index=src_idx.expand(-1, -1, t)).\
                gather(dim=-1, index=dst_all_idx.transpose(-1, -2).expand(-1, r, -1))
            dst_idx = similarity.argmax(dim=-1)[..., None]

            weight_src = importance.gather(dim=1, index=src_idx)
            weight_dst = importance.gather(dim=1, index=dst_all_idx).gather(dim=1, index=dst_idx)
            weight = F.softmax(torch.cat([weight_src, weight_dst], dim=-1), dim=-1)
            
            return src_idx, dst_all_idx, dst_idx, weight

    def forward(self, x, metric, class_token=True, token_id=None, token_size=None):
        if self.token_status["r"] <= 0:
            return x, token_id, token_size
        query, metric = metric
        if not self.batch_first:
            x, query, metric = x.permute(1, 0, 2), query.permute(1, 0, 2), metric.permute(1, 0, 2)

        src_idx, dst_all_idx, dst_idx, weight = self.match(metric, self.token_status["r"], class_token, \
            eot_token=token_id if token_id is None else token_id.argmax(dim=-1), 
            query_token=query if token_id is None else query[torch.arange(query.shape[0]), token_id.argmax(dim=-1), None, :])

        def merge(x, weight=None):
            c = x.shape[-1]
            src = x.gather(dim=1, index=src_idx.expand(-1, -1, c))
            dst = x.gather(dim=1, index=dst_all_idx.expand(-1, -1, c))
            if weight is not None:
                weight *= weight.shape[-1]
                dst = dst.scatter_add(1, dst_idx.expand(-1, -1, c), weight[..., 0, None] * src + \
                        (weight[..., 1, None] - 1) * dst.gather(dim=1, index=dst_idx.expand(-1, -1, c)))
            else:
                dst = dst.scatter_add(1, dst_idx.expand(-1, -1, c), src)
            out = dst.gather(dim=1, index=dst_all_idx.argsort(dim=1).expand(-1, -1, c)) # gather for keeping order
            return out
        x = merge(x * token_size, weight)

        if token_id is not None:
            token_id = merge(token_id[..., None] * token_size)
        token_size = merge(token_size)
        x = x / token_size
        if not self.batch_first:
            x = x.permute(1, 0, 2)
        if token_id is not None:
            token_id = (token_id / token_size)[..., 0]
            return x, token_id, token_size
        return x, None, token_size


