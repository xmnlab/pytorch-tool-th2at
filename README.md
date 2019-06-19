# pytorch-dev-tools

PyTorch Dev Tools is a bunch of util scripts to help pytorch development.

At this moment there is just a script that help convert TH files to ATen files.

It has an initial rules for the convertion. If you want to add more rules, please consider to open a PR adding new rules to:

- [nn_th2at.py/replace rules](https://github.com/xmnlab/pytorch-dev-tools/blob/c214a5a5b158113ec640b4bf3519dc94a8c98c01/pytorch_dev_tools/nn_th2at.py#L23) or [nn_th2at.py/regex rules](https://github.com/xmnlab/pytorch-dev-tools/blob/c214a5a5b158113ec640b4bf3519dc94a8c98c01/pytorch_dev_tools/nn_th2at.py#L88) for GPU and CPU
- [nn_th2at4cpu.py/replace rules](https://github.com/xmnlab/pytorch-dev-tools/blob/c214a5a5b158113ec640b4bf3519dc94a8c98c01/pytorch_dev_tools/nn_th2at4cpu.py#L11) for CPU only
- [nn_th2at4gpu.py/replace rules](https://github.com/xmnlab/pytorch-dev-tools/blob/c214a5a5b158113ec640b4bf3519dc94a8c98c01/pytorch_dev_tools/nn_th2at4gpu.py#L11) for GPU only

The `TH to ATen` script is at [th2at.py](https://github.com/xmnlab/pytorch-dev-tools/blob/master/pytorch_dev_tools/th2at.py). Following there is an example how to use this script:

```sh

 ./th2at.py --cpu \
      -f VolumetricFullDilatedConvolution.c \
      -o ~/tmp/pytorch \
      -p $PYTORCH_SRC_PATH \
      -r ~/tmp/th2at.json
```

For each file you want to port use the parameter `-f` or `--th_file`.
 
If you want to pass more extra rules use the parameter `-r` or `--file_extra_rules`.
    
Example of file with extra rules (json format):

```
    {
      "#include <THCUNN/Im2Col.h>": "#include <ATen/cuda/Im2Col.h>",
      "kH": "kernel_height",
      "kW": "kernel_width",
      "padH": "pad_height",
      "padW": "pad_width",
      "sH": "stride_height",
      "sW": "stride_width",
      "nBlocksH": "n_blocks_height",
      "nBlocksW": "n_blocks_width",
      "shapeCheck": "shape_check"
    }
```

If you have any idea to improve these scripts, feel free to open an issue and share your thoughts!
