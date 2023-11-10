#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = ['pytest>=3', ]

setup(
    author="yes",
    author_email='qiudan.wang@outlook.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="one model llm ",
    entry_points={
        'console_scripts': [
            'one_model=one_model.cli:main',
        ],
    },
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='one_model',
    name='one_model',
    packages=find_packages(include=['one_model', 'one_model.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/audreyr/one_model',
    version='0.1.0',
    zip_safe=False,
)
