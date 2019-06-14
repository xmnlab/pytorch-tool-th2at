import os

from pytorch_dev_tools.nn_th2at import NN_TH2AT


class NN_TH2AT_CPU(NN_TH2AT):
    def __init__(self, *args, **kwargs):
        # force cpu mode
        kwargs['is_cpu'] = True
        super().__init__(*args, **kwargs)
        self.rules + [
            ('updateOutput', 'out_cpu'),
            ('updateGradInput', 'backward_out_cpu')
        ]
        self.output_path = os.path.join(self.output_path, 'cpu')
        
