import os
import sys

from setuptools import setup, find_packages

base_dir = os.path.dirname(__file__)
src_dir = os.path.join(base_dir, 'src')
sys.path.insert(0, src_dir)

import cpcspeech

def get_requirements(requirements_path='requirements.txt'):
    with open(requirements_path) as fp:
        return [x.strip() for x in fp.read().split('\n') if not x.startswith('#')]


setup(
    name='cpcspeech',
    version=cpcspeech.__version__,
    description='speech embedding extraction with contrastive predictive coding',
    author='giovanni maffei',
    author_email='giovanni.maffei@gmail.com',
    packages=find_packages(where='src', exclude=['tests']),
    package_dir={'': 'src'},
    install_requires=get_requirements(),
    setup_requires=['pytest-runner', 'wheel'],
    testsp_require=get_requirements('requirements.test.txt'),
    url='https://github.com/giovannimaffei/cpcspeech.git',
    classifiers=[
        'Programming Language :: Python :: 3.7.7'
    ],
)