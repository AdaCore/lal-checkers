from setuptools import setup, find_packages

setup(
    name='lal-checkers',
    version='0.1-dev',
    author='AdaCore',
    author_email='report@adacore.com',
    url='https://github.com/AdaCore/lal-checkers',
    description='A lightweight static analysis framework for Ada.',
    requires=['funcy', 'libadalang'],
    packages=find_packages(include=['lalcheck*']),
    entry_points={
        'console_scripts': [
            'run-checkers = lalcheck.checker_runner:main'
        ]
    }
)
