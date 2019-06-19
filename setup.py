from setuptools import setup, find_packages

setup(
    name='pytorch_tool_th2at',
    version='0.1',
    description='A tool that helps to convert TH code to ATen code',
    url='http://github.com/xmnlab/pytorch-tool-th2at/',
    author='Ivan Ogasawara',
    author_email='ivan.ogasawara@gmail.com',
    license='MIT',
    packages=find_packages(),
    zip_safe=False
)
