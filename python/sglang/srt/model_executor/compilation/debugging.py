import datetime
import inspect
import os
import torch
import torch_npu

from sglang.srt.model_executor.forward_batch_info import ForwardBatch

def getpid_():
    if torch.compiler.is_dynamo_compiling():
        return 0
    return os.getpid()

def print_(text: str, flush=True):
    if not torch.compiler.is_dynamo_compiling():
        print(text, flush=flush)

def write_to_file(graph: torch.fx.GraphModule, postfix: str = ""):
    import os
    import datetime
    now = datetime.datetime.now()
    now_formatted1 = now.strftime("%Y%m%d")
    now_formatted2 = now.strftime("%H:%M:%S")
    path = f"/home/eshogulin/projects/sglang2/test/srt/logs/{now_formatted1}_{now_formatted2}_{os.getpid()}_graph_{postfix}.py"

    # code adapted from https://github.com/thuml/depyf/blob/dab831108a752d1facc00acdd6d4243891845c37/depyf/explain/patched_lazy_format_graph_code.py#L30 # noqa
    # use `print_readable` because it can include submodules
    src = "from __future__ import annotations\nimport torch\n\n" + graph.print_readable(print_output=False)
    src = src.replace("<lambda>", "GraphModule")
    src = src.replace("\", ", "\",\n            ")
    with open(path, "w") as f:
        f.write(src)

    print(f"write: graph was saved to path={path}", flush=True)
    return path


def write_per_file(split_gm: torch.fx.GraphModule, piecewise_graphs: list["SplitItem"]):
    # TODO: rename
    submod_names_compiled_only = [
        item.submod_name for item in piecewise_graphs if item.is_compiled_only
    ]

    named_modules = split_gm.named_modules()
    # print(f"NpuBackend::write_per_file: len(named_modules)={len(named_modules)}", flush=True)

    graph_index = 0
    for name, graph_module in named_modules:
        print(f"NpuBackend::write_per_file: {os.getpid()}: {graph_index}: {name}", flush=True)

        graph_index += 1
        if not name:
            continue

        graph = getattr(split_gm, name)
        if name in submod_names_compiled_only:
            write_to_file(graph, f"splitted{graph_index}")
        else:
            write_to_file(graph, f"splitted{graph_index}_compiled")


def print_forward_batch(forward_batch: ForwardBatch):    
    members = inspect.getmembers(forward_batch)
    out = f"print_forward_batch: {os.getpid()}: forward_batch ({hex(id(forward_batch))}, len(members)=({len(members)}))={type(forward_batch)}:"
    for name, value in members:
        if not name.startswith('__') and not inspect.isfunction(value) and not inspect.ismethod(value):
            if isinstance(value, torch.Tensor):
                out += f"{name} ({hex(id(value))}): {value}\n"
            else:
                out += f"{name}: {value}\n"
            
            if name == "req_to_token_pool":
                value = value.req_to_token
                out += f"{name}.req_to_token: ({hex(id(value))}): {value}\n"
    print(f"\n{out}\n", flush=True)