import torch

_global_torch_compile = None


def compile_ignore(*args, **kwargs):
    global _global_torch_compile
    if _global_torch_compile:
        torch.compile = _global_torch_compile

    def inner_func(func=None):
        return func

    return inner_func


def patch_compile():
    global _global_torch_compile
    _global_torch_compile = torch.compile
    torch.compile = compile_ignore
