from setuptools import setup, find_packages

setup(
    name='tf-albumentations',
    version='0.0.1',
    url='https://github.com/hoyso48/tf-albumentations.git',
    author='hoyso48',
    author_email='hoyeol0730@gmail.com',
    description='image augmentation library for tensorflow',
    packages=find_packages(),    
    install_requires=['tensorflow-addons >= 0.17.1'],
)
