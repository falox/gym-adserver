import setuptools
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='gym_adserver',
    version='0.1.0',
    packages=find_packages(),
    license='MIT',
    description="An OpenAI gym environment for ad serving algorithms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Alberto Falossi",
    url = 'https://github.com/falox/gym-adserver',
    keywords = ['openai', 'openai-gym', 'gym', 'environment', 'agent', 'rl', 'ads'],
    install_requires=[
        'gym',
        'numpy',
        'matplotlib',
        'pytest',
        'pytest-cov'
    ],
    python_requires='>=3.6',
    classifiers=[
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'        
    ],
)