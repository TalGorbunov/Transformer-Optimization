import torch

def describe(x, name="x", max_list=8):
    """Print structure + (if tensor) shape/dtype/device."""
    print(f"\n=== {name} ===")
    if x is None:
        print("None")
        return

    # torch tensor
    if isinstance(x, torch.Tensor):
        print("Tensor")
        print(" shape:", tuple(x.shape))
        print(" dtype:", x.dtype)
        print(" device:", x.device)
        return

    # tuple/list
    if isinstance(x, (tuple, list)):
        print(type(x).__name__, "len=", len(x))
        for i, xi in enumerate(x[:max_list]):
            describe(xi, name=f"{name}[{i}]", max_list=max_list)
        if len(x) > max_list:
            print(f"... ({len(x)-max_list} more)")
        return

    # dict
    if isinstance(x, dict):
        print("dict keys:", list(x.keys())[:max_list])
        for k in list(x.keys())[:max_list]:
            describe(x[k], name=f"{name}['{k}']", max_list=max_list)
        if len(x) > max_list:
            print(f"... ({len(x)-max_list} more keys)")
        return

    # fallback
    print("type:", type(x))
    s = str(x)
    print(s[:500] + ("..." if len(s) > 500 else ""))