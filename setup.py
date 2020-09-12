from setuptools import setup, find_packages

setup(
    name='gym_adserver',
    version='0.0.2',
    packages=find_packages(),
    install_requires=[
        'gym',
        'numpy',
        'matplotlib'
    ],
    tests_require=[
        'pytest'
    ],
    python_requires='>=3.6'
)