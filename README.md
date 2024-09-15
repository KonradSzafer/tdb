# tdb - Tensor Debugging Library for PyTorch

**tdb** is a lightweight, intuitive Python library designed for efficient debugging and exploration of PyTorch tensors. It streamlines development and improves troubleshooting by offering a straightforward interface for inspecting tensor values and shapes, facilitating a better understanding of operations and data flow in PyTorch.

## Installation:
```bash
pip install git+https://github.com/KonradSzafer/tdb.git
```

## Usage:

### Debugging:

```python
import tdb

x = torch.rand(1, 20, 5, 30)
tdb.tlog(x, 'x')
```

![](assets/example_output_1.png)

<!-- ### Example Usage:
![](assets/example_output_2.png) -->

### Memory Management:

Use the following function to free the CUDA memory of the given models and tensors. The function returns objects with their memory released:
```python
model, batch, output = tdb.release_memory(model, batch, output)
```

To see the current memory usage, run:
```python
tdb.print_memory()
```

### Configuration Options:
```python
tdb.options['disable'] = False # Set to True to disable all tdb output
tdb.options['max_values'] = 10 # Determines the maximum number of values to display from the last dimension of a tensor
tdb.options['assignment_symbol'] = '=' # Specifies the symbol used to separate tensor parameters from their values
```

## Acknowledgements:

Memory functions inspired by [Zach Mueller blog post](https://muellerzr.github.io/til/free_memory.html). Current `release_memory` function is adapted from the [Accelerate library](https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/memory.py#L29).
