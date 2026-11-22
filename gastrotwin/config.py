"""Experiment settings. The defaults are the settings used for the BVM 2027 paper."""
from dataclasses import dataclass, replace

import torch

IMG_SIZE = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# GastroVision class folders used as negatives. 'retroflex stomach' matches no folder in the
# public release; it is kept so that the class matching is identical to the published runs.
NEG_CLASSES = ('normal stomach', 'pylorus', 'retroflex stomach')
POS_CLASS = 'gastric polyp'


@dataclass(frozen=True)
class Config:
    n_twin: int = 10_000          # full twin set (configs B and C10k)
    n_syn: int = 3_000            # cut-paste set, C3k and every ablated twin
    iters: int = 4_000            # identical training budget for every model
    batch_size: int = 32
    num_workers: int = 8          # part of the data-loader RNG stream; keep at 8 to reproduce
    seeds: tuple = (0, 1, 2, 3, 4)
    n_ablation_seeds: int = 3
    audit_reps: int = 8
    pretrained: bool = True

    @property
    def seeds_ablation(self):
        return self.seeds[:self.n_ablation_seeds]

    @property
    def n_c1k(self):
        return self.n_syn // 3


PAPER = Config()
QUICK = replace(PAPER, n_twin=60, n_syn=30, iters=3, batch_size=4, num_workers=0,
                seeds=(0,), audit_reps=2, pretrained=False)
