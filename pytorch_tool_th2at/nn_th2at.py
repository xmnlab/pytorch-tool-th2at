import os
import re
import shutil


class NN_TH2AT:
    """NN_TH2AT
    
    Attributes
    ----------
    output_path : str
    pytorch_path : str
    th_files : list
    is_cpu : bool
    th_path : str
    at_path : str
    rules_name : list
    rules_name_extra : list
    rules : list
    th_dirname : str
    """
    
    rules = [
        ('#include <THNN/THNN.h>', 
         '/* TODO: remove duplicated includes */\n'
         '#include <ATen/ATen.h>\n'
         '#include <ATen/AccumulateType.h>\n'
         '#include <ATen/NativeFunctions.h>\n'
         '#include <ATen/TensorUtils.h>\n'
         '#include <ATen/Utils.h>\n'
        ),
        ('getSize(', 'size('),
        ('Acctype', 'accscalar_t'),
        ('Dtype', 'scalar_t'),
        ('ScalarConvert<scalar_t, accscalar_t>::to',
         'static_cast<accscalar_t>'),
        ('ScalarConvert<accscalar_t, scalar_t>::to',
         'static_cast<scalar_t>'),
        ('THCNumerics<scalar_t>::min()',
         'at::numeric_lmits<scalar_t>::lowest()'),
        ('THCUNN_argCheck', '/* TODO: AT_CHECK just have 2 args*/ AT_CHECK'),
        ('THAssert', 'AT_ASSERT'),
        ('THCTensor ', 'Tensor '),
        ('THCTensor*', 'Tensor*'),
        ('THTensor ', 'Tensor '),
        ('THTensor*', 'Tensor*'),
        ('putDepth', 'put_depth'),
        ('putHeight', 'put_height'),
        ('putWidth', 'put_width'),
        ('putLength', 'put_length'),
        ('putPlane', 'put_plane'),
        ('gradOut', 'grad_out'),
        ('gradIn', 'grad_in'),
        ('nBatch', 'nbatch'),
        ('nChannel', 'nchannel'),
        ('THCState *state,', ''),
        ('THState *state,', ''),
        ('THNNState *state,', ''),
        ('THCDeviceTensor', 'PackedTensorAccessor'),
        ('state, ', ''),
        ('THCState_getCurrentStream(state)', 'at::cuda::getCurrentCUDAStream()'),
        ('THArgCheck(', '/* TODO: AT_CHECK just have 2 args: condition and message */\n   AT_CHECK('),
        ('THNN_ARGCHECK(', '/* TODO: AT_CHECK just have 2 args: condition and message */\n  AT_CHECK('),
        ('THCudaCheck(cudaGetLastError())',
         'AT_CUDA_CHECK(cudaGetLastError())'),
        ('NULL,', 'Tensor(),'),
        ('THCNumerics<scalar_t>::min()', 'at::numeric_limits<scalar_t>::lowest()'),
        ('->dim()', '.dim()'),
        ('->size(', '.size('),
        ('THCeilDiv', 'cuda::ATenCeilDiv'),
        ('nInput', 'n_input'),
        ('nOutput', 'n_output'),
        ('THCTensor_(new)(state)', 'Tensor()'),
        ('THTensor_(new)(state)', 'Tensor()'),
        ('THTensor_(new)()', 'Tensor()'),
        ('batchSize', 'batch_size'),
        ('THError', 'AT_ERROR'),
        ('c10::raw::intrusive_ptr::decref', '// c10::raw::intrusive_ptr::decref'),
        ('updateOutput', 'out_cpu'),
        ('updateGradInput', 'backward_out_cpu'),
        ('#if', '// #if'),
        ('#def', '// #def'),
        ('#else', '// #else'),
        ('#endif', '// #endif'),
    ]
    
    # regex rules
    rules_regex = (
        # rule, output pattern 
        (r'THNN_\((.*)\)', None),
        (r'TH[C]*Tensor_\(size\)\(\s*([^,]*),\s*(.*)\s*\)', '{}.size({})'),
        (r'TH[C]*Tensor_\(resize([0-9]*)d\)\(\s*([^,]*),\s*(.*)\s*\)', '{1}.resize_({{ {2} }})'),
        (r'TH[C]*Tensor_\(nDimensionLegacyNoScalars\)\(\s*(.*)\s*\)', '{}.ndimension()'),
        (r'TH[C]*Tensor_\(zero\)\(\s*(.*)\s*\)', '{0}.zero_()'),
        (r'TH[C]*Tensor_\(data\)\(\s*(.*)\s*\)', '{0}.data()'),
        (r'[!](.*)->is_empty\(\)', '{}.numel() != 0'),
        (r'(\w)\s*!=\s*NULL', '{}.defined()'),
        (r'THCUNN_assertSameGPU\([0-9]*,\s*(.*)\s*\);', 
         '/* TODO: TensorArg tensorname_arg{{tensorname, "tensorname", 1}}; */\n'
         '/* TODO: checkAllSameGPU should use TensorArg */\n'
         'checkAllSameGPU(\n'
         '  "/* TODO: use the name of the function as description here */",'
         '  {{ {} }});'), 
        (r'(.*)=\s*TH[C]*Tensor_\(newContiguous\)\(\s*(.*)\s*\);', 
         'Tensor {0} = {1}_.contiguous(); /* TODO: add _ to the arg definition above */'),
        (r'accscalar_t\(\s*(.*)\s*\)', 'static_cast<accscalar_t>({})'),
        (r'TH[C]*Numerics\<scalar_t\>::ne\(\s*(.*),\s*(.*)\s*\)\s*', '{} != {}'),
    )
    
    is_cpu = True
        
    def __init__(
        self, 
        output_path, 
        pytorch_path,
        th_files,
        rules_extra=[],
        rules_name_extra=[],
        is_cpu=None
    ):
        th_dirname = 'THNN' if is_cpu else 'THCUNN'
        
        os.makedirs(output_path, exist_ok=True)
        th_path = os.path.join(pytorch_path, 'aten', 'src', th_dirname)
        at_path = os.path.join(pytorch_path, 'aten', 'src', 'ATen', 'native')
        
        if not is_cpu:
            at_path = os.path.join(at_path, 'cuda')
            
        rules_name = [
            lambda v, w='Temporal': (
                _remove_ext(v).replace(w, '') + '1d' + _get_ext(v)
                if v.startswith(w)
                else v
            ),
            lambda v, w='Spatial': (
                _remove_ext(v).replace(w, '') + '2d' + _get_ext(v)
                if v.startswith(w)
                else v
            ),
            lambda v, w='Volumetric': (
                _remove_ext(v).replace(w, '') + '3d' + _get_ext(v)
                if v.startswith(w)
                else v
            ),
        ]

        rules = rules_extra + rules_name_extra
            
        self.output_path = output_path
        self.pytorch_path = output_path
        self.th_files = th_files
        self.th_path = th_path
        self.at_path = at_path
        self.rules_name = rules_name
        self.rules_name_extra = rules_name_extra
        self.rules += rules
        self.th_dirname = th_dirname
        self.at_files = self.convert_filenames(self.th_files)
        
        if is_cpu is not None:
            self.is_cpu = is_cpu
    
    def _remove_ext(self, v):
        if '.' in v:
            return v.split('.')[0]
        return v
    
    def _get_ext(self, v):
        if '.' in v:
            return '.' + v.split('.')[-1]
        return ''

    def apply_rules(self, rules, text):
        _fn = text
        for r in rules:
            if isinstance(r, tuple):
                _fn = _fn.replace(*r)
            else:
                _fn = r(_fn)
        return _fn

    def convert_filenames(self, filenames, extra_rules: list = []):
        rules = self.rules + extra_rules

        result = []
        for fn in filenames:
            result.append(self.apply_rules(rules, fn))
        return result

    def create_initial_aten_files(self):
        """Porting code from TH[CU]NN to ATen"""
        th_at_filenames = zip(self.th_files, self.at_files)
        for th_fn, at_fn in th_at_filenames:
            # get file data from TH[CU]NN
            path_src = os.path.join(self.th_path, th_fn)
            at_file_output_path = os.path.join(self.output_path, at_fn)
            # create an empty ouput file or clean an existent file
            os.makedirs(self.output_path, exist_ok=True)
            empty_file = True
            with open(at_file_output_path, 'w') as f:
                f.write('')
            # copy also properties and metadata
            if os.path.isfile(path_src):
                empty_file = False
                shutil.copy2(path_src, at_file_output_path)
            
            with open(at_file_output_path, 'a') as f_dst:
                # get file data from TH[CU]NN/generic
                f_dst.write('\n')
                path_src = os.path.join(self.th_path, 'generic', th_fn) 
                if not os.path.isfile(path_src):
                    print('[EE] {} not found.'.format(path_src))
                    if empty_file:
                        os.remove(at_file_output_path)
                    continue
                with open(path_src, 'r') as f_src:
                    f_dst.write('\n// ' + self.th_dirname +  '/generic\n')
                    f_dst.write(f_src.read())

                # get file data from ATen/native/cuda
                # expetec a initial gpu porting after a `just cpu porting`
                # if just_gpu_porting:
                #     f_dst.write('\n')
                #     path_src = os.path.join(at_cuda_path, at_fn)
                #     if os.path.isfile(path_src):
                #         with open(path_src, 'r') as f_src:
                #             f_dst.write('\n// ATen/native/cuda\n')
                #             f_dst.write(f_src.read())

    def add_replace_rule(by, to):
        return lambda v: v.replace(by, to)

    def th2at(self, text: str):
        # replace rules
        for by, to in self.rules:
            text = text.replace(by, to)

        for rule, output_format in self.rules_regex:
            result = re.finditer(rule, text, re.MULTILINE)
            for r in result:
                _in = r.group(0)
                if output_format is None:
                    _out = r.group(1)
                else:
                    _out = output_format.format(*r.groups())
                text = text.replace(_in, self.apply_rules(self.rules_name_extra, _out))

        return text

    def store_files(self):
        # refresh output files
        self.create_initial_aten_files()

        at_files_path = [
            os.path.join(self.output_path, fn) 
            for fn in self.at_files
        ]
        
        for f_path in at_files_path:
            if not os.path.isfile(f_path):
                print('[EE] {} was not generated.'.format(f_path))
                continue

            with open(f_path, 'r') as f:
                f_content = self.th2at(f.read())

            with open(f_path, 'w') as f:
                f.write(f_content)
