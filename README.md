# cxflow

This is an official repository of **cxflow** - a smart manager and personal trainer of various deep learning models.

## Development Status

- [![CircleCI](https://circleci.com/gh/Cognexa/cxflow/tree/master.svg?style=shield)](https://circleci.com/gh/Cognexa/cxflow/tree/master)
- [![Development Status](https://img.shields.io/badge/status-CX%20Regular-brightgreen.svg?style=flat)]()
- [![Master Developer](https://img.shields.io/badge/master-Petr%20Bělohlávek-lightgrey.svg?style=flat)]()

## Example
For a quick example usage of **cxflow** please refer to a dedicated repository [Cognexa/mnist-example](https://gitlab.com/Cognexa/cxMNIST).

## Requirements
The officially supported operating system is [Arch Linux](https://www.archlinux.org).
In addition, **cxflow** is tested on [Ubuntu 16.10](http://releases.ubuntu.com/16.10).
Please note that all operating systems are expected to be fully up-to-date.

The following environments are supported and tested:
- Python 3.6 with TensorFlow 1.2 (Arch Linux)
- Python 3.5 with TensorFlow 1.2 (Ubuntu 16.10)

List of Python package requirements is listed in `requirements.txt`.

## Installation
Installation to a [virtualenv](https://docs.python.org/3/library/venv.html) is suggested, however, completely optional. 

### Standard Installation
1. Install **cxflow** `$ pip install git+https://gitlab.com/Cognexa/cxflow.git`

### Development Installation
1. Clone the **cxflow** repository `$ git clone git@gitlab.com:Cognexa/cxflow.git`
2. Enter the directory `$ cd cxflow`
3. **Optional**: *Install some of the required of packages (e.g. TensorFlow).
4. Install **cxflow**: `$ pip install -e .`

## Usage
The installation process installs `cxflow` command which might be used simply from the command line.
Please refer to repository [Cognexa/mnist-example](https://gitlab.com/Cognexa/mnist-example) for more information.

## Tutorials
The following tutorials serve as a gentle introduction to the cxflow framework:
- [introduction](tutorial)
- [cxflow hooks](cxflow/hooks/README.md)

## Extensions
**cxflow** is meant to be extremely lightweight.
For that reason the whole functionality is divided into various extensions with separate dependencies.

## Officially Supported Extensions

- [cxflow-tensorflow](https://gitlab.com/Cognexa/cxflow-tensorflow) - TensorFlow support
- [cxflow-scikit](https://gitlab.com/Cognexa/cxflow-scikit) - scientific computations and statistics

## Testing
Unit tests might be run by `$ python setup.py test`.

## License
MIT License
