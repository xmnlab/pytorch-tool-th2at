import os

from pytorch_dev_tools.nn_th2at import NN_TH2AT


class NN_TH2AT_CUDA(NN_TH2AT):
    is_cpu = False
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rules + [
            ('updateOutput', 'out_cuda'),
            ('updateGradInput', 'backward_out_cuda')
        ]
        self.output_path = os.path.join(self.output_path, 'gpu')
