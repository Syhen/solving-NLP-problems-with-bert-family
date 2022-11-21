import torch
import torch.nn as nn


class BasePooling(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, mask=None):
        raise NotImplementedError()


class IdentityPooling(BasePooling):
    def __init__(self, **kwargs):
        super(IdentityPooling, self).__init__()

    def forward(self, x, mask=None):
        return x


class AttentionPooling(BasePooling):
    def __init__(self, *, hidden_size, **kwargs):
        super(AttentionPooling, self).__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x, mask=None):
        w = self.attention(x).float()
        if mask is not None:
            w[mask == 0] = float('-inf')
        w = torch.softmax(w, 1)
        x = torch.sum(w * x, dim=1)
        return x
        

class MeanPooling(BasePooling):
    def __init__(self, **kwargs):
        super(MeanPooling, self).__init__()

    def forward(self, x, mask=None):
        if mask is not None:
            input_mask_expanded = mask.unsqueeze(-1).expand(x.size()).float()
            sum_embeddings = torch.sum(x * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
        else:
            mean_embeddings = torch.mean(x, dim=1)
        return mean_embeddings


class MaxPooling(BasePooling):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return torch.max(x, dim=1)[0]
        input_mask_expanded = mask.unsqueeze(-1).expand(x.size()).float()
        max_embeddings = torch.max(x * input_mask_expanded, dim=1)[0]
        return max_embeddings


class MeanMaxPooling(BasePooling):
    def __init__(self, **kwargs):
        super().__init__()
        self.mean_pooling = MeanPooling()
        self.max_pooling = MaxPooling()

    def forward(self, x, mask=None):
        return torch.cat([self.mean_pooling(x, mask), self.max_pooling(x, mask)], dim=1)


class LSTMPooling(BasePooling):
    def __init__(self, *, hidden_size, double_hidden_size=False, **kwargs):
        super().__init__()
        hidden_cells = hidden_size if double_hidden_size else hidden_size // 2
        self.lstm = nn.LSTM(hidden_size, hidden_cells, bidirectional=True, batch_first=True)

    def forward(self, x, mask=None):
        feature, _ = self.lstm(x)
        return feature


class GRUPooling(BasePooling):
    def __init__(self, *, hidden_size, double_hidden_size=False, **kwargs):
        super().__init__()
        hidden_cells = hidden_size if double_hidden_size else hidden_size // 2
        self.lstm = nn.GRU(hidden_size, hidden_cells, bidirectional=True, batch_first=True)

    def forward(self, x, mask=None):
        feature, _ = self.lstm(x)
        return feature
        
        
class WeightedLayerPooling(BasePooling):
    def __init__(self, *, num_hidden_layers, layer_start: int = 4, layer_weights=None, **kwargs):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None else nn.Parameter(
            torch.tensor([1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float))

    def forward(self, x, mask=None):
        x = torch.cat([i.unsqueeze(0) for i in x])
        all_layer_embedding = x[self.layer_start:, ...]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average
