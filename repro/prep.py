"""通用预处理: 为指定 brand 建内存缓存(.npz) + 5折划分(.npy)。用法: python repro/prep.py brand2"""
import sys, os, glob, random, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import brands as B
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import read_torch_pkl as R
_orig = torch.load; torch.load = lambda *a, **k: _orig(*a, **{**k, 'weights_only': False})

def main(brand):
    files = B.list_pkls(brand)
    cols = list(R.load(B.BRANDS[brand]['col']))
    N = len(files); C = len(cols)
    print(f"{brand}: {N} pkl, cols={cols}")
    X = np.zeros((N, 128, C), np.float32); car = np.zeros(N, np.int32)
    lab = np.zeros(N, np.int8); mil = np.zeros(N, np.float32)
    t0 = time.time()
    for i, p in enumerate(files):
        arr, meta = torch.load(p)
        a = np.asarray(arr, np.float32)
        X[i, :a.shape[0], :] = a[:128]
        car[i] = int(meta['car']); lab[i] = 0 if str(meta['label']).strip("'") == '00' else 1
        mil[i] = float(meta.get('mileage', 0))
        if i % 20000 == 0: print(f"  {i}/{N} {time.time()-t0:.0f}s")
    os.makedirs('repro/cache', exist_ok=True)
    np.savez(f'repro/cache/{brand}.npz', X=X, car=car, lab=lab, mil=mil, cols=cols)
    # 5折划分 (seed=0, 与官方逐字节一致)
    ind = sorted(set(car[lab == 0].tolist())); ood = sorted(set(car[lab == 1].tolist()))
    random.seed(0); random.shuffle(ind); random.shuffle(ood)
    np.save(f'repro/cache/{brand}_split.npy', {'ind_sorted': ind, 'ood_sorted': ood}, allow_pickle=True)
    print(f"saved repro/cache/{brand}.npz  X={X.shape} | 正常车={len(ind)} 故障车={len(ood)} "
          f"| 正常段={int((lab==0).sum())} 故障段={int((lab==1).sum())}  ({time.time()-t0:.0f}s)")

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'brand3')
