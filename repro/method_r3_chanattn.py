"""R3: 通道加权重建 + 时间步注意力 — 复制R1稳定训练, 改进重建信号质量。
用法: python repro/method_r3_chanattn.py brand1 --fold 0 [--epochs N]"""
import os, sys, argparse, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'DyAD'))
sys.path.insert(0, os.path.dirname(__file__))
from model.dynamic_vae import DynamicVAE
from sklearn.covariance import LedoitWolf
import brands as B

class Norm:
    def __init__(self, dfs):
        res = np.asarray(dfs); self.mean = res.mean(1).mean(0); self.std = res.std(1).mean(0)
        self.max_norm, self.min_norm = res.max(1).max(0), res.min(1).min(0)
    def __call__(self, df):
        d = np.maximum(np.maximum(1e-4, self.std), 0.1*(self.max_norm-self.min_norm)); return (df-self.mean)/d

def run(brand, fold, epochs, outdir):
    torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '32')))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.manual_seed(0); np.random.seed(0)
    hp = B.BRANDS[brand]['hp'].copy(); epochs = epochs or max(30, hp['epochs']*6)
    if brand == 'brand1': hp['noise'] = 0.01
    c = np.load(f'repro/cache/{brand}.npz', allow_pickle=True)
    X, car, lab, mil, cols = c['X'], c['car'], c['lab'], c['mil'], list(c['cols'])
    ENC, DEC, TGT = B.col_indices(brand, cols); L = hp['latent']
    sp = np.load(f'repro/cache/{brand}_split.npy', allow_pickle=True).item(); ind, ood = sp['ind_sorted'], sp['ood_sorted']; n = len(ind)
    test_ind = ind[int(fold*n/5):int((fold+1)*n/5)]
    train_cars = set(ind[:int(fold*n/5)] + ind[int((fold+1)*n/5):]); test_cars = set(test_ind) | set(ood)
    tr = np.isin(car, list(train_cars)); te = np.isin(car, list(test_cars))
    Xtr, miltr = X[tr], mil[tr]; Xte, carte, labte = X[te], car[te], lab[te]
    print(f"[{brand}-r3 f{fold}] train={tr.sum()}({len(train_cars)}车) test={te.sum()}({len(test_cars)}车) ep={epochs} noise={hp['noise']}")
    nrm = Norm(Xtr[:200]); Xtr_n = nrm(Xtr).astype(np.float32); Xte_n = nrm(Xte).astype(np.float32)
    q5, q95 = np.percentile(miltr, 5), np.percentile(miltr, 95)
    miltr_n = np.clip((miltr-q5)/max(1e-6, q95-q5), 0, 1).astype(np.float32)
    # [R3-A] 通道权重: 对目标通道按1/std加权
    tgt_std = Xtr_n[:, :, TGT].reshape(-1, len(TGT)).std(axis=0)  # per-channel std
    chan_w = 1.0/(tgt_std+1e-6); chan_w = chan_w/chan_w.sum()*len(TGT)
    chan_w_t = torch.from_numpy(chan_w.astype(np.float32)).to(dev)
    print(f"  channel std: {tgt_std.round(4)}  weights: {chan_w.round(4)}")
    model = DynamicVAE(rnn_type='gru', hidden_size=hp['hidden'], latent_size=L,
                       encoder_embedding_size=len(ENC), output_embedding_size=len(TGT),
                       decoder_embedding_size=len(DEC), num_layers=hp['num_layers'], bidirectional=hp['bidir']).float().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=hp['cosine_factor']*hp['lr'])
    mse = nn.MSELoss()
    ef = lambda x: x[:, :, ENC]; df = lambda x: x[:, :, DEC]; tf = lambda x: x[:, :, TGT]
    Xtr_t = torch.from_numpy(Xtr_n); mil_t = torch.from_numpy(miltr_n); Ntr = len(Xtr_t)
    spe = max(1, Ntr//hp['batch']); period = spe*max(2, epochs//4); step = 0
    warmup_steps = (epochs//4)*spe
    for ep in range(epochs):
        model.train(); perm = torch.randperm(Ntr); tot = 0.0
        for s in range(0, Ntr, hp['batch']):
            idx = perm[s:s+hp['batch']]; xb = Xtr_t[idx].to(dev); mb = mil_t[idx].to(dev)
            log_p, mean, log_v, z, mp = model(xb, ef, df, None, hp['noise']); target = tf(xb)
            # [R3] 通道加权 SmoothL1
            diff = log_p-target; abs_diff = diff.abs()
            smooth_flag = abs_diff < 1.0
            per_chan = torch.where(smooth_flag, 0.5*diff**2, abs_diff-0.5)
            nll = (per_chan*chan_w_t[None, None, :]).mean()
            kl = -0.5*torch.sum(1+log_v-mean.pow(2)-log_v.exp())
            if brand == 'brand1' and step < warmup_steps:
                klw = hp['anneal0']*min(1.0, step/max(1, warmup_steps//2))
            else:
                klw = hp['anneal0']*min(1.0, (step % period)/(period/2))
            lbl = mse(mp.squeeze(), mb)
            loss = hp['nll_w']*nll + hp['label_w']*lbl + klw*kl/xb.shape[0]
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tot += loss.item(); step += 1
        sch.step()
        if (ep+1) % 5 == 0 or ep == 0: print(f"  ep{ep+1}/{epochs} loss={tot/spe:.4f}")
    model.eval()
    # [R3-B] 时间步z-score推理: 在训练集上计算每时间步的误差统计
    def enc_train(Xn):
        Xt = torch.from_numpy(Xn); E = np.zeros(len(Xt), np.float32); Z = np.zeros((len(Xt), L), np.float32)
        TSE = np.zeros((len(Xt), 128), np.float32)  # per-timestep squared error
        with torch.no_grad():
            for s in range(0, len(Xt), 256):
                xb = Xt[s:s+256].to(dev); log_p, mean, *_ = model(xb, ef, df, None, hp['noise'])
                target = tf(xb); se = (log_p-target)**2
                E[s:s+len(xb)] = se.mean(dim=(1,2)).cpu().numpy()
                TSE[s:s+len(xb)] = se.mean(dim=2).cpu().numpy()  # [B, 128]
                Z[s:s+len(xb)] = mean.cpu().numpy()
        return E, Z, TSE
    def enc_test(Xn, tse_mean, tse_std):
        Xt = torch.from_numpy(Xn); E = np.zeros(len(Xt), np.float32); Z = np.zeros((len(Xt), L), np.float32)
        TSE = np.zeros((len(Xt), 128), np.float32)
        with torch.no_grad():
            for s in range(0, len(Xt), 256):
                xb = Xt[s:s+256].to(dev); log_p, mean, *_ = model(xb, ef, df, None, hp['noise'])
                target = tf(xb); se = (log_p-target)**2
                TSE[s:s+len(xb)] = se.mean(dim=2).cpu().numpy()
                Z[s:s+len(xb)] = mean.cpu().numpy()
        # z-score每时间步
        tse_z = (TSE-tse_mean)/(tse_std+1e-8)
        E = tse_z.mean(axis=1)  # 各时间步z-score均值
        return E.astype(np.float32), Z
    Etr, Ztr, TSE_tr = enc_train(Xtr_n)
    tse_mean = TSE_tr.mean(axis=0); tse_std = TSE_tr.std(axis=0)+1e-8
    Ete, Zte = enc_test(Xte_n, tse_mean, tse_std)
    # 用原版Etr做融合的z-score(较公平)
    ez = (Ete-Etr.mean())/(Etr.std()+1e-8)
    # LedoitWolf maha
    lw = LedoitWolf().fit(Ztr); cov = lw.covariance_; inv = np.linalg.inv(cov)
    mu = Ztr.mean(0); d = Zte-mu; maha = np.einsum('ij,jk,ik->i', d, inv, d).astype(np.float32)
    dtr = Ztr-mu; mtr = np.einsum('ij,jk,ik->i', dtr, inv, dtr)
    mz = (maha-mtr.mean())/(mtr.std()+1e-8)
    combined = (ez+mz).astype(np.float32)
    os.makedirs(outdir, exist_ok=True); import pandas as pd
    pd.DataFrame({'car': carte, 'label': labte, 'rec_error': Ete, 'maha': maha, 'combined': combined}).to_csv(f'{outdir}/seg_fold{fold}.csv', index=False)
    print(f"[{brand}-r3 f{fold}] saved {outdir}/seg_fold{fold}.csv")

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('brand'); ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=0); ap.add_argument('--outdir', default='')
    a = ap.parse_args(); run(a.brand, a.fold, a.epochs, a.outdir or f'repro/out/{a.brand}_r3')
