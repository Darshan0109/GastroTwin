"""U-Net training, inference and scoring."""
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

from .config import DEVICE

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
ARCHS = {'unet_r34': ('resnet34', 3e-4), 'unet_mitb2': ('mit_b2', 1e-4)}


def to_tensor(x):
    return torch.from_numpy(((x.astype(np.float32) / 255 - MEAN) / STD).transpose(0, 3, 1, 2))


class AugmentedDataset(torch.utils.data.Dataset):
    """Flips, 90-degree rotations, per-channel gain and a brightness offset."""

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x, y = self.x[i].astype(np.float32) / 255, self.y[i].astype(np.float32)
        if np.random.rand() < 0.5:
            x, y = x[:, ::-1], y[:, ::-1]
        if np.random.rand() < 0.5:
            x, y = x[::-1], y[::-1]
        k = np.random.randint(4)
        x, y = np.rot90(x, k), np.rot90(y, k)
        x = np.clip(x * np.random.uniform(0.8, 1.2, 3) + np.random.uniform(-0.08, 0.08), 0, 1)
        x = ((x - MEAN) / STD).astype(np.float32).transpose(2, 0, 1)
        return torch.from_numpy(x.copy()), torch.from_numpy(y[None].copy())


def build_model(arch, pretrained=True):
    encoder, _ = ARCHS[arch]
    return smp.Unet(encoder, encoder_weights='imagenet' if pretrained else None, classes=1)


def train(x, y, seed, arch, cfg, log_every=500):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(arch, cfg.pretrained).to(DEVICE)
    lr = ARCHS[arch][1]
    loader = torch.utils.data.DataLoader(
        AugmentedDataset(x, y), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        drop_last=True, persistent_workers=cfg.num_workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=cfg.iters)
    dice_loss = smp.losses.DiceLoss('binary')

    model.train()
    it, running = 0, 0.0
    while it < cfg.iters:
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == 'cuda'):
                logits = model(xb)
            logits = logits.float()
            loss = F.binary_cross_entropy_with_logits(logits, yb) + dice_loss(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            it += 1
            running += loss.item()
            if it % log_every == 0:
                print(f'  iter {it}  loss {running / log_every:.3f}')
                running = 0.0
            if it >= cfg.iters:
                break
    return model.eval()


@torch.no_grad()
def predict(model, x, batch_size=64):
    out = []
    for i in range(0, len(x), batch_size):
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == 'cuda'):
            logits = model(to_tensor(x[i:i + batch_size]).to(DEVICE))
        out.append(torch.sigmoid(logits.float())[:, 0].cpu())
    return torch.cat(out)


def image_score(prob):
    """Image-level polyp score: maximum of the 5x5 average-pooled probability map."""
    return F.avg_pool2d(prob[:, None], 5, 1, 2)[:, 0].flatten(1).amax(1).numpy()


def dice_score(prob, y):
    pred = (prob > 0.5).numpy()
    inter = (pred * y).sum((1, 2))
    return np.mean((2 * inter + 1) / (pred.sum((1, 2)) + y.sum((1, 2)) + 1))


def audit_detections(prob, masks):
    """A polyp counts as detected if any pixel inside its true mask exceeds 0.5."""
    prob = prob.numpy()
    return np.array([(prob[i][masks[i] == 1] > 0.5).any() for i in range(len(masks))])
