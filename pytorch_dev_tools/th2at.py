import json
import os

import click

from pytorch_dev_tools.nn_th2at4cpu import NN_TH2AT_CPU
from pytorch_dev_tools.nn_th2at4gpu import NN_TH2AT_CUDA


@click.command()
@click.option(
    '--th_file',
    '-f',
    multiple=True,
    required=True,
    help='TH file that will be ported'
)
@click.option(
    '--output_path',
    '-o',
    required=True,
    help='Output path for the new ATen files'
)
@click.option(
    '--pytorch_path',
    '-p',
    required=True,
    help='PyTorch path'
)
@click.option(
    '--cpu',
    is_flag=True,
    default=False,
    show_default=True,
    help='Run porting for CPU'
)
@click.option(
    '--gpu',
    is_flag=True,
    default=False,
    show_default=True,
    help='Run porting for CUDA'
)
@click.option(
    '--file_extra_rules',
    '-r',
    default=None, 
    help='The path of the file that add extra rules'
)
def run(
    th_file,
    output_path,
    pytorch_path,
    cpu,
    gpu,
    file_extra_rules
):
    """Run TH to ATen porting.
    
    If you want to pass more extra rules use the parameter: --file_extra_rules
    
    Example of file with extra rules (json format):
    
    \b
    {
      "#include <THCUNN/Im2Col.h>": "#include <ATen/cuda/Im2Col.h>",
      "kH": "kernel_height",
      "kW": "kernel_width",
      "dH": "dilation_height",
      "dW": "dilation_width",
      "padH": "pad_height",
      "padW": "pad_width",
      "sH": "stride_height",
      "sW": "stride_width",
      "nBlocksH": "n_blocks_height",
      "nBlocksW": "n_blocks_width",
      "shapeCheck": "shape_check"
    }
    
    """
    if cpu == gpu:
        raise Exception(
            'Specify just CPU or GPU, not both or none of them.'
        )
    
    rules_extra = []
    
    if not os.path.isdir(pytorch_path):
        raise Exception(
            'The path specified for `pytorch_path` doesn\'t exist.'
        )
    
    if file_extra_rules:
        if not os.path.isfile(file_extra_rules):
            raise Exception(
                'The file specified for `file_extra_rules` doesn\'t exist.'
            )
        with open(file_extra_rules, 'r') as f:
            rules_extra = list(json.load(f).items())            
    
    kwargs = {
        'output_path': output_path, 
        'pytorch_path': pytorch_path,
        'th_files': list(th_file),
        'rules_extra': rules_extra,
        'rules_name_extra': []  # not implemented yet
    }

    if cpu:
        print('[II] CPU mode recognized')
        th2at = NN_TH2AT_CPU(**kwargs)
    else:
        print('[II] GPU mode recognized')
        th2at = NN_TH2AT_CUDA(**kwargs)
    th2at.store_files()


if __name__ == '__main__':
    run()
