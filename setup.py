from setuptools import setup, find_packages


setup(
    name='baby',
    version='0.0.1',
    author='Adrien Hadj-Salah',
    author_email='adrien.hadj.salah@gmail.com',
    description='A special env for a generic mathematical problem',
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy==1.18.1',
        'tensorflow==1.15.2',
        'tensorflow-gpu==1.14.0',
        'gym==0.15.4',
        #'baselines', If mpi4py error while installing baselines, install separately 
        'pandas==0.25.3',
        'pytest==5.2.2',
        'matplotlib==3.1.2',
        'jupyter==1.0.0'],
)
