"""
极简 torch .pkl 读取器 (无需安装 torch)。
torch.save 的 zip 格式: archive/data.pkl 是 pickle 流, archive/data/<key> 是各 storage 原始字节。
本脚本自定义 Unpickler, 把 storage 还原成 numpy 数组, 用于离线读取 DyAD 数据集。
"""
import io, zipfile, pickle, struct
import numpy as np

_DTYPE = {
    'FloatStorage': np.float32, 'DoubleStorage': np.float64,
    'HalfStorage': np.float16, 'LongStorage': np.int64,
    'IntStorage': np.int32, 'ShortStorage': np.int16,
    'CharStorage': np.int8, 'ByteStorage': np.uint8, 'BoolStorage': np.bool_,
}

class _Stub:
    def __init__(self, dtype): self.dtype = dtype

def load(path):
    zf = zipfile.ZipFile(path)
    names = zf.namelist()
    root = names[0].split('/')[0]
    raw = {}  # key -> bytes
    for n in names:
        if '/data/' in n and not n.endswith('data.pkl'):
            raw[n.split('/')[-1]] = zf.read(n)

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name in _DTYPE:
                return _Stub(_DTYPE[name])
            if name == '_rebuild_tensor_v2':
                return _rebuild
            if name == 'OrderedDict':
                from collections import OrderedDict; return OrderedDict
            if name == '_rebuild_tensor':
                return _rebuild
            try:
                return super().find_class(mod, name)
            except Exception:
                return lambda *a, **k: None
        def persistent_load(self, pid):
            # pid = ('storage', storage_type_stub, key, location, numel)
            _, stub, key, _loc, numel = pid
            dt = stub.dtype if isinstance(stub, _Stub) else np.float32
            buf = raw.get(str(key), b'')
            return np.frombuffer(buf, dtype=dt)

    def _rebuild(storage, offset, size, stride, *rest):
        a = np.asarray(storage)
        try:
            return np.lib.stride_tricks.as_strided(
                a[offset:], shape=tuple(size),
                strides=tuple(s * a.itemsize for s in stride)).copy()
        except Exception:
            n = int(np.prod(size)) if len(size) else a.size
            return a[offset:offset + n].reshape(size) if len(size) else a

    data_pkl = [n for n in names if n.endswith('data.pkl')][0]
    return U(io.BytesIO(zf.read(data_pkl))).load()


if __name__ == '__main__':
    import sys, glob, collections
    base = sys.argv[1] if len(sys.argv) > 1 else 'data/battery_brand3'
    cols = load(f'{base}/column.pkl')
    print("COLUMNS:", list(cols))
    files = sorted(glob.glob(f'{base}/data/*.pkl'))
    print("n_files:", len(files))
    arr, meta = load(files[0])
    a = np.asarray(arr)
    print("snippet shape:", a.shape, "dtype:", a.dtype)
    print("meta:", {k: (repr(v)[:60]) for k, v in meta.items()})
    print("per-column min/max:")
    for i, c in enumerate(cols):
        print(f"   {str(c):20s} min={a[:,i].min():.3f} max={a[:,i].max():.3f}")
    labs, cars = collections.Counter(), set()
    for p in files[:5000]:
        m = load(p)[1]
        labs[str(m.get('label'))] += 1
        cars.add(int(m.get('car')))
    print("label dist (first 5000):", dict(labs), "| unique cars:", len(cars))
