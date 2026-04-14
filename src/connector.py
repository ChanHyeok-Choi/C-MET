import torch
import torch.nn as nn
import re
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from huggingface_hub import PyTorchModelHubMixin



class PositionalEmbedding(nn.Module):
    def __init__(self, d_model=512, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class Conv1d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, act='ReLU', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_block = nn.Sequential(
            nn.Conv1d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm1d(cout)
        )

        self.residual = residual and (cin == cout and stride == 1)
        self.shortcut = None
        if self.residual and cin != cout:
            self.shortcut = nn.Conv1d(cin, cout, 1, 1, 0)  # 1x1 Conv1d로 차원 맞추기

        self.act = {
            'ReLU': nn.ReLU(),
            'Tanh': nn.Tanh(),
            None: nn.Identity()
        }[act]

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            shortcut = x if self.shortcut is None else self.shortcut(x)
            out += shortcut
        return self.act(out)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    

class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', config['mm_projector_type'])
        
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config['mm_hidden_size'], config['hidden_size'])]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config['hidden_size'], config['hidden_size']))
        self.projector = nn.Sequential(*modules)
        
    def forward(self, x):  # (bs, 1, 256)
        return self.projector(x)  # (bs, 1, 10)
    
    
class Fusion_transformer_encoder(nn.Module):
    def __init__(self, T, d_model=512, nlayers=4, nhead=4, dim_feedforward=1024,  # T, 512, 4, 4, 1024
                 dropout=0.1, device='cuda'):
        super().__init__()
        self.T=T
        self.position_r = PositionalEmbedding(d_model=d_model)  # for ref embedding
        self.position_e = PositionalEmbedding(d_model=d_model)  # for ED embedding
        self.modality = nn.Embedding(4, d_model, padding_idx=0)  # 1 for ref, 2 for speaker, 3 for exp
        self.dropout = nn.Dropout(p=dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.device = device

    def forward(self, ref_emb, speaker_embedding, ED_exp_embedding):  # (bs, T, hs), (bs, 1, hs) (bs, T, hs)
        # 01. positional(temporal) encoding
        position_r_encoding = self.position_r(ref_emb)  # (bs, T, hs)
        position_e_encoding = self.position_e(ED_exp_embedding)

        #(2)  modality encoding
        modality_r = self.modality(1 * torch.ones((ref_emb.size(0), self.T), dtype=torch.int).to(self.device))  # (bs, T, hs)
        modality_s = self.modality(2 * torch.ones((speaker_embedding.size(0), 1), dtype=torch.int).to(self.device))  # (bs, 1, hs)
        modality_e = self.modality(3 * torch.ones((ED_exp_embedding.size(0),  self.T), dtype=torch.int).to(self.device))  # (bs, T, hs)

        ref_tokens = ref_emb + position_r_encoding + modality_r  # (bs, T, hs)
        MV_tokens = speaker_embedding + modality_s  # (bs, 1, hs)
        ED_tokens = ED_exp_embedding + position_e_encoding + modality_e    # (bs, T, hs)

        #(3) concat tokens
        input_tokens = torch.cat((ref_tokens, MV_tokens, ED_tokens), dim=1)  # (B, T+1+T, hs)
        input_tokens = self.dropout(input_tokens)

        #(4) input to transformer
        output = self.transformer_encoder(input_tokens)
        return output


class Speaker_encoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=1024):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.apply(weight_init)
    
    def forward(self, x): #  (bs, 1, 768)
        x = x.squeeze(1)
        out = self.proj(x)  # (bs, 1024)
        out = self.act(out)
        out = self.norm(out)  # (bs, 1024)
        return out.unsqueeze(1)  # (bs, 1, 1024)
    

class Connector_exp(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config_projector, config_transformer, device='cuda'):
        super().__init__()
        self.device = device
        self.fusion_transformer_encoder = Fusion_transformer_encoder(config_transformer['T'], config_transformer['hidden_size'], 
                                                                     config_transformer['nlayer'], config_transformer['nhead'], 
                                                                     config_transformer['dim_feedforward'], config_transformer['dropout'], device)
        self.exp_encoder = nn.Sequential(  # (B*T,1,10)
                    Conv1d(1, 4, 3, 1, 1),  # 10
                    Conv1d(4, 8, 3, 2, 1),  # 5
                    Conv1d(8, 8, 3, 1, 1, residual=True),
                    Conv1d(8, 8, 3, 1, 1, residual=True),

                    Conv1d(8, 16, 3, 2, 1),  # 3
                    Conv1d(16, 16, 3, 1, 1, residual=True),
                    Conv1d(16, 16, 3, 1, 1, residual=True),

                    Conv1d(16, 32, 2, 2, 0),  # 1  ← 커널 크기 변경 (3 → 2)
                    Conv1d(32, 64, 1, 1, 0),  # 1
                    Conv1d(64, 128, 1, 1, 0),  # 1
                    Conv1d(128, 256, 1, 1, 0),  # 1
                    Conv1d(256, 512, 1, 1, 0),  # 1
                    Conv1d(512, 1024, 1, 1, 0, act='Tanh')  # 1
                )
        self.ref_encoder = nn.Sequential(  # (B*T,1,10)
                    Conv1d(1, 4, 3, 1, 1),  # 10
                    Conv1d(4, 8, 3, 2, 1),  # 5
                    Conv1d(8, 8, 3, 1, 1, residual=True),
                    Conv1d(8, 8, 3, 1, 1, residual=True),

                    Conv1d(8, 16, 3, 2, 1),  # 3
                    Conv1d(16, 16, 3, 1, 1, residual=True),
                    Conv1d(16, 16, 3, 1, 1, residual=True),

                    Conv1d(16, 32, 2, 2, 0),  # 1  ← 커널 크기 변경 (3 → 2)
                    Conv1d(32, 64, 1, 1, 0),  # 1
                    Conv1d(64, 128, 1, 1, 0),  # 1
                    Conv1d(128, 256, 1, 1, 0),  # 1
                    Conv1d(256, 512, 1, 1, 0),  # 1
                    Conv1d(512, 1024, 1, 1, 0, act='Tanh')  # 1
                )
        self.hs = config_transformer['hidden_size']
        self.speaker_encoder = Speaker_encoder(input_dim=config_projector['speaker']['mm_hidden_size'], hidden_dim=config_projector['speaker']['hidden_size'])
        self.projector = Projector(config_projector['exp'])
        
        self.apply(weight_init)
        self.Norm=nn.LayerNorm(self.hs)

        if config_projector['path'] is not None:
            self.load_encoders(config_projector['path'])

    def load_encoders(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.speaker_encoder.load_state_dict(checkpoint['speaker_encoder'])
        self.exp_encoder.load_state_dict(checkpoint['exp_encoder'])

        
    def forward(self, ref_emb, e2v, ED_neu):  # (bs, T, 10) (bs, 1, 768) (bs, T, 10)
        T = ED_neu.size(1)

        ref_emb = ref_emb.reshape(-1, 1, 10)  # (bs*T, 1, 10)
        ref_emb = self.ref_encoder(ref_emb).squeeze(-1)
        ref_emb = self.Norm(ref_emb)
        ref_emb = ref_emb.reshape(-1, T, self.hs)

        e2v = self.speaker_encoder(e2v)  # (bs, 1, hs)

        ED_neu = ED_neu.reshape(-1, 1, 10)  # (bs*T, 1, 10)
        ED_neu = self.exp_encoder(ED_neu).squeeze(-1)  # (bs*T, hs)
        ED_neu = self.Norm(ED_neu)  # (bs*T, hs)
        ED_neu = ED_neu.reshape(-1, T, self.hs)  # (bs, T, hs)

        fusion_embedding = self.fusion_transformer_encoder(ref_emb, e2v, ED_neu)  # (bs, T+1, hs)
        fusion_embedding = self.projector(fusion_embedding[:, T+1:, :])  # (bs, T, 10)

        out = fusion_embedding.reshape(-1, fusion_embedding.size(2))  # (bs*T, 10)
        return out, (ref_emb, e2v, ED_neu)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    bs, T = 1, 5
    ref_emb = torch.randn(bs, T, 10).float().to(device)
    e2v = torch.randn(bs, 1, 1024).float().to(device)
    ED_neu = torch.randn(bs, T, 10).float().to(device)
    # Example configuration
    config_projector = {
        'path': None,
        'exp': {
            'mm_projector_type': 'mlp2x_gelu',
            'mm_hidden_size': 1024,
            'hidden_size': 10
        },
        'speaker': {
            'mm_projector_type': 'mlp2x_gelu',
            'mm_hidden_size': 1024,
            'hidden_size': 1024
        }
    }
    connector_exp = Connector_exp(config_projector, 
                        {'T': T, 'hidden_size': 1024, 'nlayer': 4, 'nhead': 4, 'dim_feedforward': 1024, 'dropout': 0.1}, device).to(device)
    
    out, (ref_emb, e2v, ED_neuo) = connector_exp(ref_emb, e2v, ED_neu)

    print(out.size(), ref_emb.size(), e2v.size(), ED_neu.size())  # torch.Size([bs*T, 10]) 
