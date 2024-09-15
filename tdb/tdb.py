import gc
import torch


class tdb:

    options = {
        "disable": False,
        "max_values": 10,
        "assigment_symbol": "=",
    }

    @staticmethod
    def set_options(**kwargs) -> None:
        for key, value in kwargs.items():
            if key not in tdb.options:
                raise KeyError(f"Unknown option: {key}")
            tdb.options[key] = value
            
    @staticmethod
    def print_memory() -> None:
        if tdb.options["disable"]:
            return
        print(
            f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB, "
            f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB, "
            f"Memory cached: {torch.cuda.memory_cached() / 1024**3:.2f} GB, "
            f"Max memory cached: {torch.cuda.max_memory_cached() / 1024**3:.2f} GB"
        )
    
    @staticmethod
    def release_memory(*objects):
        if not isinstance(objects, list):
            objects = list(objects)
        for i in range(len(objects)):
            objects[i] = None
        gc.collect()
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            torch.mps.empty_cache()
        else:
            torch.cuda.empty_cache()
        return objects

    @staticmethod
    def log(*args, **kwargs) -> None:
        if tdb.options["disable"]:
            return
        print(*args, **kwargs)

    @staticmethod
    def tlog(
        tensor: torch.Tensor,
        title: str = None,
        values: bool = True,
        stats: bool = True,
        meta: bool = True,
        sep: bool = False,
        ) -> None:

        if tdb.options["disable"]:
            return

        if not isinstance(tensor, torch.Tensor):
            raise TypeError("The input must be a PyTorch tensor.")

        assignment_symbol = tdb.options["assigment_symbol"]
        tensor_clone = tensor.clone() # Avoid modifying the original tensor

        if sep:
            print("-"*100)
        
        if title is not None:
            print(title, end="")
            if values:
                print(" => ", end="")
            else:
                print()

        if values:
            torch.set_printoptions(threshold=tdb.options["max_values"])
            slice_ = tuple(
                slice(1) if i < len(tensor_clone.shape)-1
                else slice(None)
                for i in range(tensor_clone.dim())
            )
            print(tensor_clone[slice_].data.cpu())
            torch.set_printoptions(profile="default")
        
        if stats:
            print(
                f"min{assignment_symbol}{tensor_clone.min().item():.3f}, "
                f"max{assignment_symbol}{tensor_clone.max().item():.3f}, ",
                end=""
            )
            if (tensor_clone.dtype.is_floating_point or tensor_clone.dtype.is_complex):
                print(
                    f"mean{assignment_symbol}{tensor_clone.mean().item():.3f}, "
                    f"std{assignment_symbol}{tensor_clone.std().item():.3f}",
                )
            else:
                print(
                    f"mean{assignment_symbol}{tensor_clone.float().mean().item():.3f}, "
                    f"std{assignment_symbol}{tensor_clone.float().std().item():.3f}",
                )

        if meta:
            # Warning: this section is using original tensor
            print(
                f"{tensor.shape}, "
                f"dtype{assignment_symbol}{tensor.dtype}, "
                f"device{assignment_symbol}'{tensor.device}', "
                f"grad_fn{assignment_symbol}{tensor.grad_fn}\n"
            )
