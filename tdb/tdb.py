import torch


class tdb:

    options = {
        "print_values_threshold": 10,
        "disable": False,
        "assigment_symbol": "=",
    }

    @staticmethod
    def set_options(**kwargs):
        for key, value in kwargs.items():
            if key not in tdb.options:
                raise KeyError(f"Unknown option: {key}")
            tdb.options[key] = value

    @staticmethod
    def print(
        tensor: torch.Tensor,
        title: str = None,
        values: bool = True,
        metadata: bool = True,
        sep: bool = False,
        ) -> None:

        def _print(*args, **kwargs) -> None:
            if not tdb.options["disable"]:
                print(*args, **kwargs)

        if not isinstance(tensor, torch.Tensor):
            raise TypeError("The input must be a PyTorch tensor.")

        assignment_symbol = tdb.options["assigment_symbol"]
        tensor_clone = tensor.clone() # Avoid modifying the original tensor

        if sep:
            _print("-"*100)
        
        if title is not None:
            _print(title, end="")
            if values:
                _print(" => ", end="")
            else:
                _print()

        if values:
            torch.set_printoptions(threshold=tdb.options["print_values_threshold"])
            slice_ = tuple(
                slice(1) if i < len(tensor_clone.shape)-1
                else slice(None)
                for i in range(tensor_clone.dim())
            )
            _print(tensor_clone[slice_].data.cpu())
            torch.set_printoptions(profile="default")

        if metadata:
            _print(
                f"min{assignment_symbol}{tensor_clone.min().item():.3f}, "
                f"max{assignment_symbol}{tensor_clone.max().item():.3f}, ",
                end=""
            )
            if (tensor_clone.dtype.is_floating_point or tensor_clone.dtype.is_complex):
                _print(
                    f"mean{assignment_symbol}{tensor_clone.mean().item():.3f}, "
                    f"std{assignment_symbol}{tensor_clone.std().item():.3f}",
                )
            else:
                _print(
                    f"mean{assignment_symbol}{tensor_clone.float().mean().item():.3f}, "
                    f"std{assignment_symbol}{tensor_clone.float().std().item():.3f}",
                )
            # Warning: this section is using original tensor
            _print(
                f"{tensor.shape}, "
                f"dtype{assignment_symbol}{tensor.dtype}, "
                f"device{assignment_symbol}'{tensor.device}', "
                f"grad_fn{assignment_symbol}{tensor.grad_fn}\n"
            )
