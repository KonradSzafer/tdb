# tdb - Tensor Debugging Library for PyTorch

tdb: A Python library for simple debugging and inspection of PyTorch tensors, facilitating the development and troubleshooting of PyTorch-based machine learning models.

### Installation:
```
pip install git+https://github.com/KonradSzafer/tdb.git
```

### Usage:
```
from tdb import tdb

x = torch.rand(1, 20, 5, 30)
tdb.print(x, 'x')
```

![](assets/example_output_1.png)

### Example output:
![](assets/example_output_2.png)
