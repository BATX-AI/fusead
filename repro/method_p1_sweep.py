"""P1: 参数化扫参脚本 — 支持 --noise, --epochs, --hidden, --latent 覆盖。
用法: python repro/method_p1_sweep.py brand1 --noise 1.0 --epochs 3 --fold 0"""
import os, sys, argparse, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code', 'DyAD'))
sys.path.insert(0, os.path.dirname(__file__))
from model.dynamic_vae import DynamicVAE
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture
import brands as B

class Norm:
    def __init__(self, dfs):
        res = np.asarray(dfs); self.mean = res.mean(1).mean(0); self.std = res.std(1).mean(0)
        self.max_norm, self.min_norm = res.max(1).max(0), res.min(1).min(0)
    def __call__(self, df):
        d = np.maximum(np.maximum(1e-4, self.std), 0.1*(self.max_norm-self.min_norm)); return (df-self.mean)/d

def run(brand, fold, epochs, outdir, noise_override, hidden_override, latent_override):
    torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '32')))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.manual_seed(0); np.random.seed(0)
    hp = B.BRANDS[brand]['hp'].copy()
    if noise_override is not None: hp['noise'] = noise_override
    if hidden_override is not None: hp['hidden'] = hidden_override
    if latent_override is not None: hp['latent'] = latent_override
    c = np.load(f'repro/cache/{brand}.npz', allow_pickle=True)
    X, car, lab, mil, cols = c['X'], c['car'], c['lab'], c['mil'], list(c['cols'])
    ENC, DEC, TGT = B.col_indices(brand, cols); L = hp['latent']
    sp = np.load(f'repro/cache/{brand}_split.npy', allow_pickle=True).item(); ind, ood = sp['ind_sorted'], sp['ood_sorted']; n = len(ind)
    test_ind = ind[int(fold*n/5):int((fold+1)*n/5)]
    train_cars = set(ind[:int(fold*n/5)] + ind[int((fold+1)*n/5):]); test_cars = set(test_ind) | set(ood)
    tr = np.isin(car, list(train_cars)); te = np.isin(car, list(test_cars))
    Xtr, miltr = X[tr], mil[tr]; Xte, carte, labte = X[te], car[te], lab[te]
    print(f"[{brand}-sweep f{fold}] train={tr.sum()}({len(train_cars)}车) ep={epochs} noise={hp['noise']} hidden={hp['hidden']} latent={L}", flush=True)
    nrm = Norm(Xtr[:200]); Xtr_n = nrm(Xtr).astype(np.float32); Xte_n = nrm(Xte).astype(np.float32)
    q5, q95 = np.percentile(miltr, 5), np.percentile(miltr, 95)
    miltr_n = np.clip((miltr-q5)/max(1e-6, q95-q5), 0, 1).astype(np.float32)
    model = DynamicVAE(rnn_type='gru', hidden_size=hp['hidden'], latent_size=L,
                       encoder_embedding_size=len(ENC), output_embedding_size=len(TGT),
                       decoder_embedding_size=len(DEC), num_layers=hp['num_layers'], bidirectional=hp['bidir']).float().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=hp['cosine_factor']*hp['lr'])
    smooth, mse = nn.SmoothL1Loss(), nn.MSELoss()
    ef = lambda x: x[:, :, ENC]; df = lambda x: x[:, :, DEC]; tf = lambda x: x[:, :, TGT]
    Xtr_t = torch.from_numpy(Xtr_n); mil_t = torch.from_numpy(miltr_n); Ntr = len(Xtr_t)
    step = 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(Ntr); tot = 0.0
        for s in range(0, Ntr, hp['batch']):
            idx = perm[s:s+hp['batch']]; xb = Xtr_t[idx].to(dev); mb = mil_t[idx].to(dev)
            log_p, mean, log_v, z, mp = model(xb, ef, df, None, hp['noise']); target = tf(xb)
            nll = smooth(log_p, target); kl = -0.5*torch.sum(1+log_v-mean.pow(2)-log_v.exp())
            klw = hp['anneal0']*min(1.0, step/hp['x0'])
            lbl = mse(mp.squeeze(), mb)
            loss = hp['nll_w']*nll + hp['label_w']*lbl + klw*kl/xb.shape[0]
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tot += loss.item(); step += 1
        sch.step()
        if (ep+1) % max(1, epochs//3) == 0 or ep == 0: print(f"  ep{ep+1}/{epochs} loss={tot/max(1,Ntr//hp['batch']):.4f}", flush=True)
    model.eval()
    def enc(Xn):
        Xt = torch.from_numpy(Xn); E = np.zeros(len(Xt), np.float32); Z = np.zeros((len(Xt), L), np.float32)
        with torch.no_grad():
            for s in range(0, len(Xt), 256):
                xb = Xt[s:s+256].to(dev); log_p, mean, *_ = model(xb, ef, df, None, hp['noise'])
                E[s:s+len(xb)] = ((log_p-tf(xb))**2).mean(dim=(1,2)).cpu().numpy(); Z[s:s+len(xb)] = mean.cpu().numpy()
        return E, Z
    Etr, Ztr = enc(Xtr_n); Ete, Zte = enc(Xte_n)
    lw = LedoitWolf().fit(Ztr); cov = lw.covariance_; inv = np.linalg.inv(cov)
    mu = Ztr.mean(0); d = Zte-mu; maha = np.einsum('ij,jk,ik->i', d, inv, d).astype(np.float32)
    dtr = Ztr-mu; mtr = np.einsum('ij,jk,ik->i', dtr, inv, dtr)
    n_cars = len(train_cars); max_k = min(10, max(2, n_cars//3))
    # GMM
    bics = []
    for k in range(1, max_k+1):
        g = GaussianMixture(n_components=k, covariance_type='full', random_state=0, reg_covar=1e-4)
        g.fit(Ztr); bics.append(g.bic(Ztr))
    best_k = int(np.argmin(bics)+1)
    gmm = GaussianMixture(n_components=best_k, covariance_type='full', random_state=0, reg_covar=1e-4).fit(Ztr)
    gmm_nll_tr = -gmm.score_samples(Ztr); gmm_nll_te = -gmm.score_samples(Zte)
    ez = (Ete-Etr.mean())/(Etr.std()+1e-8); gz = (gmm_nll_te-gmm_nll_tr.mean())/(gmm_nll_tr.std()+1e-8)
    combined = (ez+gz).astype(np.float32)
    mz = (maha-mtr.mean())/(mtr.std()+1e-8); combined_maha = (ez+mz).astype(np.float32)
    os.makedirs(outdir, exist_ok=True); import pandas as pd
    pd.DataFrame({'car': carte, 'label': labte, 'rec_error': Ete, 'maha': maha,
                  'gmm_nll': gmm_nll_te.astype(np.float32), 'combined': combined,
                  'combined_maha': combined_maha}).to_csv(f'{outdir}/seg_fold{fold}.csv', index=False)
    print(f"[{brand}-sweep f{fold}] saved (k={best_k})", flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('brand'); ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=3); ap.add_argument('--outdir', default='')
    ap.add_argument('--noise', type=float, default=None)
    ap.add_argument('--hidden', type=int, default=None); ap.add_argument('--latent', type=int, default=None)
    a = ap.parse_args()
    run(a.brand, a.fold, a.epochs, a.outdir or f'repro/out/{a.brand}_sweep', a.noise, a.hidden, a.latent)
