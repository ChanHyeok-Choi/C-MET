from torch import nn
from .encoder import *
from .styledecoder import Synthesis
import torch
import time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_module_time(fn, warmup=10, iters=100, use_cuda=None):
    """
    fn: 인자로 아무것도 받지 않고, forward 한 번을 수행하는 함수 (lambda로 감싸서 전달)
    """
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()

    # warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = fn()
        if use_cuda:
            torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = fn()
        if use_cuda:
            torch.cuda.synchronize()
    end = time.perf_counter()

    avg_ms = (end - start) / iters * 1000.0
    return avg_ms


class Direction(nn.Module):
    def __init__(self, lip_dim, pose_dim):
        super(Direction, self).__init__()
        self.lip_dim = lip_dim
        self.pose_dim = pose_dim
        self.weight = nn.Parameter(torch.randn(512, lip_dim+pose_dim))

    def forward(self, input):
        # input: (bs*t) x 512

        weight = self.weight + 1e-8
        Q, R = torch.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]

        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix
            out = torch.matmul(input_diag, Q.T)
            out = torch.sum(out, dim=1)

            return out
    def get_shared_out(self, input):
        # input: (bs*t) x 512

        weight = self.weight + 1e-8
        Q, R = torch.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]

        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix
            out = torch.matmul(input_diag, Q.T)  # torch.Size([1, 20, 512])
            return out
            # out = torch.sum(out, dim=1)

            # return out
    def get_lip_latent(self, out):
        lip_latent = torch.sum(out[:,:self.lip_dim], dim=1)
        return lip_latent
    def get_pose_latent(self, out):
        pose_latent = torch.sum(out[:,self.lip_dim:], dim=1)
        return pose_latent

class Direction_exp(nn.Module):
    def __init__(self, lip_dim, pose_dim, exp_dim):
        super(Direction_exp, self).__init__()
        self.lip_dim = lip_dim
        self.pose_dim = pose_dim
        self.exp_dim = exp_dim
        self.weight = nn.Parameter(torch.randn(512, exp_dim))

    def forward(self, input, lipnonlip_weight):
        # input: (bs*t) x 512
        weight = torch.cat([lipnonlip_weight, self.weight], -1)
        weight = weight + 1e-8 # torch.Size([512, 36])
        Q, R = torch.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]

        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix torch.Size([1, 36]) torch.Size([1, 36, 36])
            out = torch.matmul(input_diag, Q.T) # Q torch.Size([512, 36]) OUT torch.Size([1, 36, 512])
            out = torch.sum(out, dim=1)

            return out

    def only_exp(self, input):
        # input: (bs*t) x 512

        weight = self.weight + 1e-8
        Q, R = torch.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]

        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix torch.Size([1, 40, 40])
            out = torch.matmul(input_diag, Q.T)
            out = torch.sum(out, dim=1)

            return out

    def get_shared_out(self, input, lipnonlip_weight):
        # input: (bs*t) x 512
        weight = torch.cat([lipnonlip_weight, self.weight], -1)
        weight = weight + 1e-8
        Q, R = torch.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]

        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix
            out = torch.matmul(input_diag, Q.T)  # torch.Size([1, 20, 512])
            return out
            # out = torch.sum(out, dim=1)

            # return out
    def get_lip_latent(self, out):
        lip_latent = torch.sum(out[:,:self.lip_dim], dim=1)
        return lip_latent
    def get_pose_latent(self, out):
        pose_latent = torch.sum(out[:,self.lip_dim:self.lip_dim+self.pose_dim], dim=1)
        return pose_latent

    def get_exp_latent(self, out):
        exp_latent = torch.sum(out[:,self.lip_dim+self.pose_dim:], dim=1)
        return exp_latent


class Generator(nn.Module):
    def __init__(self, size, style_dim=512, lip_dim=20, pose_dim=6, exp_dim = 10, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super(Generator, self).__init__()

        # encoder
        self.lip_dim = lip_dim
        self.pose_dim = pose_dim
        self.exp_dim = exp_dim
        self.enc = Encoder(size, style_dim)
        self.dec = Synthesis(size, style_dim, lip_dim+pose_dim, blur_kernel, channel_multiplier)
        # self.direction = Direction(motion_dim)
        self.direction_lipnonlip = Direction(lip_dim, pose_dim)
        self.direction_exp = Direction_exp(lip_dim, pose_dim, exp_dim)
        # motion network
        fc = [EqualLinear(style_dim, style_dim)]
        for i in range(3):
            fc.append(EqualLinear(style_dim, style_dim))
        self.fc = nn.Sequential(*fc)
        # self.source_fc = EqualLinear(style_dim, motion_dim)

        lip_fc = [EqualLinear(style_dim, style_dim)]
        lip_fc.append(EqualLinear(style_dim, style_dim))
        lip_fc.append(EqualLinear(style_dim, lip_dim))
        self.lip_fc = nn.Sequential(*lip_fc)

        pose_fc = [EqualLinear(style_dim, style_dim)]
        pose_fc.append(EqualLinear(style_dim, style_dim))
        pose_fc.append(EqualLinear(style_dim, pose_dim))
        self.pose_fc = nn.Sequential(*pose_fc)

        exp_fc = [EqualLinear(style_dim, style_dim)]
        exp_fc.append(EqualLinear(style_dim, style_dim))
        exp_fc.append(EqualLinear(style_dim, exp_dim))
        self.exp_fc = nn.Sequential(*exp_fc)


    def test_EDTalk_V(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

        wa, wa_t, feats, feats_t = self.enc(img_source, lip_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        wa_t_p,wa_t_exp, feats_t_p,feats_t_exp = self.enc(pose_img_drive, exp_img_drive) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        shared_fc = self.fc(wa_t)
        alpha_D_lip = self.lip_fc(shared_fc)

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_exp)

        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a = self.direction_exp.get_shared_out(alpha_D, self.direction_lipnonlip.weight)
        e = self.direction_exp.get_exp_latent(a)
        directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + directions_D 
        img_recon = self.dec(latent_poseD, feats, e)
        return img_recon

    def test_EDTalk_V_use_exp_weight(self, img_source, lip_img_drive, pose_img_drive, alpha_D_exp, h_start=None):

        wa, wa_t, feats, _ = self.enc(img_source, lip_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        wa_t_p,_, _,_ = self.enc(pose_img_drive, None) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        shared_fc = self.fc(wa_t)
        alpha_D_lip = self.lip_fc(shared_fc)

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)
        
        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a = self.direction_exp.get_shared_out(alpha_D, self.direction_lipnonlip.weight)
        e = self.direction_exp.get_exp_latent(a)
        directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + directions_D 
        img_recon = self.dec(latent_poseD, feats, e)
        return img_recon

    def test_EDTalk_A(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, h_start=None):

        wa, wa_t_exp, feats, feats_t = self.enc(img_source, exp_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        wa_t_p,_, _,_ = self.enc(pose_img_drive, None) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # shared_fc = self.fc(wa_t)
        alpha_D_lip = lip_img_drive

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_exp)

        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a = self.direction_exp.get_shared_out(alpha_D, self.direction_lipnonlip.weight)
        e = self.direction_exp.get_exp_latent(a)
        directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + directions_D
        img_recon = self.dec(latent_poseD, feats, e)
        return img_recon


    def test_EDTalk_A_use_exp_weight(self, img_source, lip_img_drive, pose_img_drive, alpha_D_exp, h_start=None):

        wa, wa_t_p, feats, _ = self.enc(img_source, pose_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # shared_fc = self.fc(wa_t)
        alpha_D_lip = lip_img_drive

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a = self.direction_exp.get_shared_out(alpha_D, self.direction_lipnonlip.weight)
        e = self.direction_exp.get_exp_latent(a)
        directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + directions_D
        img_recon = self.dec(latent_poseD, feats, e)
        return img_recon
    
    
    def test_EDTalk_A_vector_operation(self, img_source, lip_img_drive, pose_img_drive, exp_img_drive, neu_img_drive, h_start=None):

        wa, wa_t_exp, feats, feats_t = self.enc(img_source, exp_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        _, wa_t_neu, _, _ = self.enc(img_source, neu_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        wa_t_p,_, _,_ = self.enc(pose_img_drive, None) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # shared_fc = self.fc(wa_t)
        alpha_D_lip = lip_img_drive

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)

        shared_fc_exp = self.fc(wa_t_exp)
        alpha_D_exp = self.exp_fc(shared_fc_exp)
        
        shared_fc_neu = self.fc(wa_t_neu)
        alpha_D_neu = self.exp_fc(shared_fc_neu)
        
        shared_fc_src = self.fc(wa)
        alpha_D_src = self.exp_fc(shared_fc_src)
        
        strength = 1.0
        alpha_D_exp = strength * (alpha_D_exp - alpha_D_neu) + alpha_D_src
        
        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a = self.direction_exp.get_shared_out(alpha_D, self.direction_lipnonlip.weight)
        e = self.direction_exp.get_exp_latent(a)
        directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + directions_D
        img_recon = self.dec(latent_poseD, feats, e)
        return img_recon
    
    
    def compute_alpha_D(self, img_source):
        wa, _, _, _ = self.enc(img_source, None)
        shared_fc = self.fc(wa)
        alpha_D_exp = self.exp_fc(shared_fc)
        alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D_lip = self.lip_fc(shared_fc)
        return alpha_D_exp, alpha_D_pose, alpha_D_lip

    
    def test_EDTalk_predicted_D(self, img_source, alpha_D_lip, alpha_D_pose, alpha_D_exp, h_start=None):
        wa, _, feats,_ = self.enc(img_source, h_start)

        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a = self.direction_exp.get_shared_out(alpha_D, self.direction_lipnonlip.weight)
        e = self.direction_exp.get_exp_latent(a)
        directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + directions_D 
        img_recon = self.dec(latent_poseD, feats, e)
        return img_recon


    def test_EDTalk_predicted_D_exp_lip(self, img_source, pose_img_drive, alpha_D_exp, alpha_D_lip, h_start=None):
        wa, wa_t_p, feats, _ = self.enc(img_source, pose_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)
        
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a = self.direction_exp.get_shared_out(alpha_D, self.direction_lipnonlip.weight)
        e = self.direction_exp.get_exp_latent(a)
        directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + directions_D 
        img_recon = self.dec(latent_poseD, feats, e)
        return img_recon

    
    def test_EDTalk_AV_use_exp_weight(self, img_source, lip_img_drive, pose_img_drive, alpha_D_exp, h_start=None):

        wa, wa_t_p, feats, _ = self.enc(img_source, pose_img_drive, h_start) # torch.Size([1, 512]) alpha3个torch.Size([1, 20])
        # shared_fc = self.fc(wa_t)
        # alpha_D_lip = self.lip_fc(shared_fc)
        alpha_D_lip = lip_img_drive

        shared_fc_p = self.fc(wa_t_p)
        alpha_D_pose = self.pose_fc(shared_fc_p)
        
        # alpha_D_pose = self.pose_fc(shared_fc)
        alpha_D = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)
        a = self.direction_exp.get_shared_out(alpha_D, self.direction_lipnonlip.weight)
        e = self.direction_exp.get_exp_latent(a)
        directions_D = self.direction_exp(alpha_D, self.direction_lipnonlip.weight) # torch.Size([1, 512])
        latent_poseD = wa + directions_D 
        img_recon = self.dec(latent_poseD, feats, e)
        return img_recon
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Generator 초기화 (size는 실제 모델과 맞춰주세요: 128, 256 등)
    gen = Generator(size=128, style_dim=512, lip_dim=20, pose_dim=6, exp_dim=10).to(device)
    gen.eval()

    # -----------------------------
    # 1) Expression encoder 파라미터 수 (원하시면)
    # -----------------------------
    enc_params       = count_parameters(gen.enc)
    dir_exp_params   = count_parameters(gen.direction_exp)
    exp_fc_params    = count_parameters(gen.exp_fc)
    dir_lipnon_params = count_parameters(gen.direction_lipnonlip)

    exp_encoder_params = enc_params + dir_exp_params + exp_fc_params + dir_lipnon_params
    print(f"[Expression encoder params] {exp_encoder_params/1e6:.2f} M")

    # -----------------------------
    # 2) Expression encoder 경로 시간 측정
    # -----------------------------
    # 예시용 더미 입력 (B=1, C=3, H=W=128)
    B, C, H, W = 1, 3, 128, 128
    img_source    = torch.randn(B, C, H, W, device=device)
    exp_img_drive = torch.randn(B, C, H, W, device=device)

    def expression_encoder_forward():
        # 예: img_source, exp_img_drive는 이미 위에서 정의된 dummy 입력이라고 가정
        # B, C, H, W = 1, 3, 128, 128
        # img_source    = torch.randn(B, C, H, W, device=device)
        # exp_img_drive = torch.randn(B, C, H, W, device=device)

        # Encoder forward: exp branch에서 쓰이는 latent만 사용
        # enc 시그니처는 실제 코드에 맞게 사용하시면 됩니다.
        wa, wa_t_exp, _, _ = gen.enc(img_source, exp_img_drive, h_start=None)

        # shared_fc from exp latent
        # shared_fc_exp = gen.fc(wa_t_exp)

        # 여기서 "간단히" lip / pose / exp 모두를 같은 shared_fc_exp에서 만들자
        # alpha_D_lip  = gen.lip_fc(shared_fc_exp)   # (B, lip_dim=20)
        # alpha_D_pose = gen.pose_fc(shared_fc_exp)  # (B, pose_dim=6)
        # alpha_D_exp  = gen.exp_fc(shared_fc_exp)   # (B, exp_dim=10)

        # # Direction_exp가 기대하는 전체 alpha_D (36차원)
        # alpha_D = torch.cat([alpha_D_lip, alpha_D_pose, alpha_D_exp], dim=-1)  # (B, 36)

        # # shared_out & expression latent
        # a = gen.direction_exp.get_shared_out(alpha_D, gen.direction_lipnonlip.weight)
        # e = gen.direction_exp.get_exp_latent(a)

        # return e  # 출력은 중요X, 시간만 측정
        return None


    # avg_ms = measure_module_time(expression_encoder_forward,
    #                              warmup=10,
    #                              iters=100,
    #                              use_cuda=(device=="cuda"))
    # print(f"[Expression encoder] Avg forward time: {avg_ms:.3f} ms")

    # ---- 2) wa_t_exp의 입력을 T=100으로 고정 ----
    B = 1
    T = 100
    style_dim = 512

    # (B*T, 512) 모양의 wa_t_exp 생성
    wa_t_exp = torch.randn(B*T, style_dim, device=device)   # (100, 512)

    # ---- 3) 우리가 측정할 두 줄의 forward만 정의 ----
    def test_exp():
        wa, wa_t_exp, _, _ = gen.enc(img_source, exp_img_drive, h_start=None)
        shared_fc_exp = gen.fc(wa_t_exp)           # (100, 512)
        alpha_D_exp   = gen.exp_fc(shared_fc_exp)  # (100, 10)
        return alpha_D_exp

    # ---- 4) 시간 측정 ----
    avg_ms = measure_module_time(
        test_exp,
        warmup=10,
        iters=200,
        use_cuda=(device == "cuda"),
    )

    print(f"[fc + exp_fc] avg forward time @ T=100: {avg_ms:.4f} ms")