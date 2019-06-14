from setuptools import setup, find_packages

setup(
    name='pytorch_dev_tools',
    version='0.1',
    description='Some tools that helps pytorch development',
    url='http://github.com/xmnlab/pytorch_dev_tools/',
    author='Ivan Ogasawara',
    author_email='ivan.ogasawara@gmail.com',
    license='MIT',
    packages=find_packages(),
    zip_safe=False
)