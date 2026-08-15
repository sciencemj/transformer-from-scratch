import urllib.request
from pathlib import Path

import torch

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "tinyshakespeare.txt"
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)


def ensure_dataset(path=DEFAULT_PATH, url=TINY_SHAKESPEARE_URL) -> Path:
    """Download the corpus on first use so a fresh clone just runs."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url} -> {path}")
        urllib.request.urlretrieve(url, path)
    return path


class CharTokenizer:
    """Character-level vocabulary built from the corpus itself.

    Every distinct character in the text gets an id, so there is no unknown
    token and no reserved pad id -- language modelling batches are dense.
    """

    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = dict(enumerate(self.chars))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos[int(i)] for i in ids)


def load_text(path=DEFAULT_PATH, download=False) -> str:
    path = Path(path)
    if download:
        ensure_dataset(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; download tinyshakespeare into data/ first"
        )
    return path.read_text(encoding="utf-8")


def split_data(ids, val_frac=0.1):
    """Contiguous split so validation text is never seen during training."""
    n = int(len(ids) * (1 - val_frac))
    return ids[:n], ids[n:]


def get_batch(data, block_size, batch_size, device=None, generator=None):
    """Random windows into the corpus.

    x: (B, block_size), y: x shifted one character left.
    """
    if len(data) <= block_size:
        raise ValueError(f"data of length {len(data)} too short for {block_size=}")
    idx = torch.randint(
        len(data) - block_size - 1, (batch_size,), generator=generator
    )
    x = torch.stack([data[i : i + block_size] for i in idx])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in idx])
    if device is not None:
        x, y = x.to(device), y.to(device)
    return x, y
