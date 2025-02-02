from distutils.core import setup
from setuptools import find_packages

setup(
    name='mocapact',
    version='0.1',
    author='Nolan Wagener',
    author_email='nolan.wagener@gatech.edu',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'setuptools==66',
        'wheel==0.38.4',
        'stable-baselines3==1.8.0',
        'azure.storage.blob==12.9.0',
        'cloudpickle>=2.1.0',
        'dm_control>=1.0.7',
        'gym==0.21',
        'h5py',
        'imageio',
        'imageio-ffmpeg',
        'ml_collections',
        'mujoco',
        'pytorch-lightning==2.0.7',
        'tensorboard',
    ]
)
