import torch

_CACHED = None


def select_torch_device():
    global _CACHED
    if _CACHED is not None:
        return _CACHED
    if torch.cuda.is_available():
        try:
            t = torch.zeros(1).cuda()
            _ = (t + t).item()
            _CACHED = torch.device("cuda")
            return _CACHED
        except Exception:
            _CACHED = torch.device("cpu")
            return _CACHED
    _CACHED = torch.device("cpu")
    return _CACHED
