"""Offline verification for the torchreid / Market-1501 setup. Setup check only, no training."""
import os, sys, glob, socket
import os.path as osp

ROOT = osp.expanduser('~/Documents/RSKDLabs/data')

# 1. force offline: any socket connection attempt raises (set OFFLINE_BLOCK=1)
if os.environ.get('OFFLINE_BLOCK') == '1':
    _real_connect = socket.socket.connect
    def _blocked(self, addr, *a, **k):
        if isinstance(addr, tuple):  # allow only AF_UNIX / loopback (dataloader workers etc.)
            if addr[0] not in ('127.0.0.1', '::1', 'localhost'):
                raise OSError('NETWORK BLOCKED by verify_offline.py: %r' % (addr,))
        return _real_connect(self, addr, *a, **k)
    socket.socket.connect = _blocked
    socket.getaddrinfo = lambda *a, **k: (_ for _ in ()).throw(OSError('DNS BLOCKED'))
try:
    socket.create_connection(('1.1.1.1', 443), timeout=3); print('NETWORK: REACHABLE (not an offline test!)')
except OSError as e:
    print('NETWORK: unreachable ->', e)

import torch, torchreid
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.is_available(), '| torchreid', torchreid.__version__)

# 2. distractors
dist = osp.join(ROOT, 'market1501', 'Market-1501-v15.09.15', 'images')
n_dist = len(glob.glob(osp.join(dist, '*.jpg'))) if osp.isdir(dist) else 0
print(f'distractor images in {dist}: {n_dist}', '(expect ~500000)' if n_dist else '  ** MISSING **')

# 3. data manager
dm = torchreid.data.ImageDataManager(root=ROOT, sources='market1501', height=256, width=128,
                                     batch_size_train=32, workers=2)
print('num_train_pids:', dm.num_train_pids, '(expect 751)')
print('query size:', len(dm.test_loader['market1501']['query'].dataset), '| gallery size:', len(dm.test_loader['market1501']['gallery'].dataset))
print('train size:', len(dm.train_loader.dataset))

# 4. model + one forward pass
model = torchreid.models.build_model('osnet_x1_0', num_classes=dm.num_train_pids, pretrained=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(dev).train()
batch = next(iter(dm.train_loader))
with torch.no_grad():
    out = model(batch['img'].to(dev))
print('batch img', tuple(batch['img'].shape), '-> logits', tuple(out.shape), 'on', dev)

# 5. 500k option (only if distractors present)
if n_dist:
    dm2 = torchreid.data.ImageDataManager(root=ROOT, sources='market1501', targets='market1501',
                                          height=256, width=128, batch_size_train=32, workers=2, market1501_500k=True)
    print('500k gallery size:', len(dm2.test_loader['market1501']['gallery'].dataset))
print('VERIFY_OK' if dm.num_train_pids == 751 else 'VERIFY_FAIL')
