"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal as MVN

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)  # 작은 값 추가


        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.tensor(1.0, device=mask_pos_pairs.device), mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class AngleLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(AngleLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, feature1, feature2):
        cos = torch.nn.CosineSimilarity(dim=-1)
        cos_sim = cos(feature1, feature2) / self.temperature
        cos_sim = torch.clamp(cos_sim, -0.99999, 0.99999)
        angle = torch.acos(cos_sim)
        angle = torch.mean(angle)
        return angle


class EmotionConsistencyLoss(nn.Module):
    def __init__(self):
        super(EmotionConsistencyLoss, self).__init__()
    
    def forward(self, emotion_pred, emotion_tgt):
        emotion_pred = F.normalize(emotion_pred, p=2, dim=-1)
        emotion_tgt = F.normalize(emotion_tgt, p=2, dim=-1)
        emotion_consistency_loss = 1 - F.cosine_similarity(emotion_pred, emotion_tgt, dim=-1).mean()
        return emotion_consistency_loss

class VelocityLoss(nn.Module):
    def __init__(self):
        super(VelocityLoss, self).__init__()
    
    def forward(self, pred, gt):
        pred_spiky = pred[:, 1:, :] - pred[:, :-1, :]  # (B, T-1, 10)
        gt_spiky = gt[:, 1:, :] - gt[:, :-1, :]  # (B, T-1, 10)

        pairwise_distance = torch.nn.functional.pairwise_distance(
            pred_spiky.view(-1, pred_spiky.size(-1)), 
            gt_spiky.view(-1, gt_spiky.size(-1))
        )
        return pairwise_distance.mean()


class WeightedMSELoss(nn.Module):
    def __init__(self, class_weights=None):
        super(WeightedMSELoss, self).__init__()
        self.class_weights = class_weights  # dict {class_idx: weight}

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, emotion_class: torch.Tensor=None):
        """
        y_pred: (B, D)
        y_true: (B, D)
        emotion_class: (B,)
        """
        loss = (y_pred - y_true) ** 2  # (B, D)
        loss = loss.mean(dim=-1)       # (B,)  — mean per sample

        if self.class_weights is not None and emotion_class is not None:
            weights = torch.tensor(
                [self.class_weights[int(c.item())] for c in emotion_class],
                device=y_pred.device
            )
        else:
            weights = torch.ones_like(loss)
        
        weighted_loss = loss * weights
        return weighted_loss.mean()


class EmotionConsistencyLoss(nn.Module):
    def __init__(self, class_weights=None):
        super(EmotionConsistencyLoss, self).__init__()
        self.class_weights = class_weights  # dict {class_idx: weight}

    def forward(self, emotion_pred: torch.Tensor, emotion_tgt: torch.Tensor, emotion_class: torch.Tensor=None):
        """
        emotion_pred: (B, D)
        emotion_tgt:  (B, D)
        emotion_class: (B,) — LongTensor with class indices
        """
        emotion_pred = F.normalize(emotion_pred, p=2, dim=-1)
        emotion_tgt = F.normalize(emotion_tgt, p=2, dim=-1)

        # cosine similarity: (B,)
        cos_sim = F.cosine_similarity(emotion_pred, emotion_tgt, dim=-1)

        # weight for each sample based on its class
        if self.class_weights is not None and emotion_class is not None:
            weights = torch.tensor(
                [self.class_weights[int(c.item())] for c in emotion_class],
                device=emotion_pred.device
            )
        else:
            weights = torch.ones_like(cos_sim)

        loss = 1 - cos_sim
        weighted_loss = loss * weights
        return weighted_loss.mean()


class DirectionalLoss(nn.Module):
    def __init__(self, class_weights=None):
        super(DirectionalLoss, self).__init__()
        self.class_weights = class_weights

    def forward(self, dir_12: torch.Tensor, dir_21: torch.Tensor, emotion_class: torch.Tensor=None) -> torch.Tensor:
        cos_sim = F.cosine_similarity(dir_12, dir_21, dim=-1)
        # cosine_loss = torch.mean((cos_sim + 1) ** 2)

        if self.class_weights is not None and emotion_class is not None:
            weights = torch.tensor(
                [self.class_weights[int(c.item())] for c in emotion_class],
                device=dir_12.device
            )
        else:
            weights = torch.ones_like(cos_sim)

        cosine_loss = 1 + cos_sim  # We want the cosine similarity to be close to -1
        weighted_cosine_loss = cosine_loss * weights
        weighted_cosine_loss = torch.mean(weighted_cosine_loss ** 2)  # Squared loss
        return weighted_cosine_loss


class CntLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(CntLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, visual_feat, audio_feat, emo_label):
        '''
        visual_feat: (B, D)
        audio_feat: (B, D)
        emo_label: (B,) — LongTensor with class indices
        0: angry, 1: contempt, 2: disgusted, 3: fear, 4: happy, 5: neutral, 6: sad, 7: surprised
        '''
        B = visual_feat.size(0)
        device = visual_feat.device

        # Normalize features
        visual_feat = F.normalize(visual_feat, dim=1)
        audio_feat = F.normalize(audio_feat, dim=1)

        # Combine visual and audio: 2B x D
        feats = torch.cat([visual_feat, audio_feat], dim=0)  # (2B, D)
        labels = emo_label.repeat(2)  # (2B,)

        # Compute similarity matrix
        sim_matrix = torch.div(torch.matmul(feats, feats.T), self.temperature)  # (2B, 2B)
        logits_mask = ~torch.eye(2*B, dtype=torch.bool, device=device)

        # Create mask for positive pairs (same label, not itself)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask * logits_mask.float()  # exclude self-comparisons

        # Compute log-softmax
        sim_matrix = sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0]  # stability
        exp_sim = torch.exp(sim_matrix) * logits_mask.float()
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        # Compute loss
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -mean_log_prob_pos.mean()
        return loss


def weighted_focal_mse_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def bmc_loss_md(pred, target, noise_var):
    """Compute the Multidimensional Balanced MSE Loss (BMC) between `pred` and the ground truth `target`.
    Args:
      pred:   FloatTensor of shape [B, D].
      target: FloatTensor of shape [B, D].
      noise_var: scalar tensor (sigma^2).
    Returns:
      loss: scalar tensor.
    """

    B, D = pred.shape

    # --- 핵심 수정: pred와 동일한 device/dtype으로 단위행렬 만들기 ---
    I = torch.eye(D, device=pred.device, dtype=pred.dtype)

    # MVN expects (B, 1, D) loc and (D,D) covariance that can broadcast.
    logits = MVN(
        loc=pred.unsqueeze(1),                 # [B,1,D]
        covariance_matrix=noise_var * I        # [D,D], broadcasts
    ).log_prob(
        target.unsqueeze(0)                    # [1,B,D] -> logits [B,B]
    )

    # target for CE must be LongTensor on same device
    ce = F.cross_entropy(logits, torch.arange(B, device=pred.device))

    # optional scale restore, detach on noise_var like in the repo
    # loss = ce * (2 * noise_var).detach()
    loss = ce

    return loss


class BMCLoss(nn.Module):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss_md(pred, target, noise_var)


class_weights = {
    0: 0.5,  # angry
    1: 1.0,  # contempt
    2: 0.5,  # disgusted
    3: 1.0,  # fear
    4: 2.0,  # happy
    5: 0.1,  # neutral
    6: 1.0,  # sad
    7: 2.0  # surprised
}


criterion_Huber = torch.nn.SmoothL1Loss(beta=1.0)  # beta 값은 실험적으로 조정
criterion_MSE = torch.nn.MSELoss()
criterion_CE = torch.nn.CrossEntropyLoss()
criterion_SupCon = SupConLoss()
criterion_Angle = AngleLoss()
criterion_EC = EmotionConsistencyLoss()
criterion_Vel = VelocityLoss()
criterion_wMSE = WeightedMSELoss(class_weights=class_weights)
criterion_wEC = EmotionConsistencyLoss(class_weights=class_weights)
criterion_Dir = DirectionalLoss()
criterion_wDir = DirectionalLoss(class_weights=class_weights)
criterion_Cnt = CntLoss()
criterion_BMC = BMCLoss(init_noise_sigma=8.0)


if __name__ == '__main__':
    bs = 32
    T = 5

    out = torch.randn([bs * T, 10])
    y = torch.randn([bs * T, 10])
    MSE_loss = criterion_MSE(out, y)
    Angle_loss = criterion_Angle(out, y)
    CE_loss = criterion_CE(out, y)
    Vel_loss = criterion_Vel(out.reshape(bs, T, out.size(1)), y.reshape(bs, T, y.size(1)))
    print("MSE Loss:", MSE_loss)
    print("Angle Loss:", Angle_loss)
    print("CE Loss:", CE_loss)
    print("Vel Loss:", Vel_loss)

    emo_label = torch.tensor([0]) * (bs * T)
    wMSE_loss = criterion_wMSE(out, y, emo_label)
    EC_loss = criterion_EC(out, y, emo_label)
    print("Weighted MSE Loss:", wMSE_loss)
    print("Emotion Consistency Loss:", EC_loss)
    
    emotion_labels = torch.randint(0, 7, (bs*T,))
    out_cnt = out.unsqueeze(1)
    SupCon_loss = criterion_SupCon(out_cnt, emotion_labels)
    print("Supervised Contrastive Loss:", SupCon_loss)

    Dir_loss = criterion_Dir(out, -out)
    print("Directional Loss (out, -out):", Dir_loss, " (should be close to 0 if directions are similar)")
    Dir_loss = criterion_Dir(out, y)
    print("Directional Loss (out, y):", Dir_loss)
    wDir_loss = criterion_wDir(out, -out, emotion_labels)
    print("Weighted Directional Loss (out, -out):", wDir_loss, "(should be close to 0 if directions are similar)")
    wDir_loss = criterion_wDir(out, y, emotion_labels)
    print("Weighted Directional Loss (out, y):", wDir_loss)

    visual_feat = torch.randn(bs*T, 1024)
    audio_feat = torch.randn(bs*T, 1024)
    Cnt_loss = criterion_Cnt(visual_feat, audio_feat, emotion_labels)
    print("Contrastive Loss (visual_feat, audio_feat):", Cnt_loss)

    focal_mse_loss = weighted_focal_mse_loss(out, y, weights=None, activate='sigmoid', beta=.2, gamma=1)
    print("Weighted Focal MSE Loss (out, y):", focal_mse_loss)

    BMC_loss = criterion_BMC(out, y)
    print("Balanced MSE Loss (out, y):", BMC_loss)