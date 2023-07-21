from setuptools import setup, find_packages

setup(
    name='bayespso',
    version='0.0.1',
    description='Evolutionary algorithms for hyperparameter optimization',
    url='',
    author=['Laurits Tani'],
    author_email='laurits.tani@cern.ch',
    license='GPLv3',
    packages=find_packages(),
    package_data={
        'bayespso': [
            'tests/*',
            'slurm_scripts/*',
            'scripts/*'
        ]
    },
    install_requires=[
        'docopt',
        'scipy',
        'scikit-learn',
        'pandas',
        'numpy',
        'xgboost'
    ],
)
