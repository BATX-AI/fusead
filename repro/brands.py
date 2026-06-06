"""多 brand 配置: 数据布局 / task 列名 / 官方超参。供 prep/repro/improve 复用。"""
import os, glob

# task -> (encoder 列名顺序, decoder_dimension)  ; target = encoder[dec_dim:]
TASKS = {
    'batterybranda': (['soc','current','min_temp','max_single_volt','max_temp','min_single_volt','volt'], 2),
    'batterybrandb': (['soc','current','min_temp','max_single_volt','min_single_volt','volt','max_temp'], 4),
    'ev':            (['soc','current','max_temp','max_single_volt','min_single_volt','volt'], 2),
}

# 各 brand: 数据目录(可多个), 列文件, task, 官方超参(model_params_battery_brandX.json)
BRANDS = {
    'brand1': dict(
        dirs=['data/battery_brand1/train', 'data/battery_brand1/test'],
        col='data/battery_brand1/column.pkl', task='batterybranda',
        hp=dict(epochs=3, batch=128, lr=0.005, cosine_factor=0.1, hidden=128, latent=8,
                num_layers=2, bidir=True, noise=1.0, nll_w=10, label_w=0.001, anneal0=0.01, x0=500)),
    'brand2': dict(
        dirs=['data/battery_brand2/train', 'data/battery_brand2/test'],
        col='data/battery_brand2/column.pkl', task='batterybrandb',
        hp=dict(epochs=3, batch=128, lr=0.0001, cosine_factor=0.1, hidden=1024, latent=24,
                num_layers=1, bidir=True, noise=0.01, nll_w=5, label_w=1.0, anneal0=0.1, x0=500)),
    'brand3': dict(
        dirs=['data/battery_brand3/data'],
        col='data/battery_brand3/column.pkl', task='batterybranda',
        hp=dict(epochs=5, batch=128, lr=0.005, cosine_factor=0.1, hidden=256, latent=16,
                num_layers=1, bidir=True, noise=0.01, nll_w=10, label_w=0.001, anneal0=0.1, x0=500)),
}

def list_pkls(brand):
    fs = []
    for d in BRANDS[brand]['dirs']:
        fs += glob.glob(os.path.join(d, '*.pkl'))
    return sorted(fs)

def col_indices(brand, cols):
    """返回 (ENC, DEC, TGT) 列索引: 按该 brand 的 task 列名解析到实际 column.pkl 顺序。"""
    cols = list(cols)
    names, dec_dim = TASKS[BRANDS[brand]['task']]
    ENC = [cols.index(n) for n in names]
    return ENC, ENC[:dec_dim], ENC[dec_dim:]
