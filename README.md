# cxflow
[![CircleCI](https://circleci.com/gh/Cognexa/cxflow/tree/master.svg?style=shield)](https://circleci.com/gh/Cognexa/cxflow/tree/master)
[![Development Status](https://img.shields.io/badge/status-CX%20Regular-brightgreen.svg?style=flat)]()
[![MIT license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)
[![Master Developer](https://img.shields.io/badge/master-Petr%20Bělohlávek-lightgrey.svg?style=flat)]()

This is an official repository of **cxflow** - a smart manager and personal trainer of various deep learning models.

## Example
For a quick example usage of **cxflow** please refer to a dedicated repository [Cognexa/cxMNIST](https://github.com/Cognexa/cxMNIST).

## Requirements
Supported operating systems:
- [Arch Linux](https://www.archlinux.org)
- The latest [Ubuntu LTS release](http://releases.ubuntu.com)
- [Ubuntu rolling release](http://releases.ubuntu.com).

Supported Python interpreters:
- Python 3.5+
- Python 3.6+

The list of Python package requirements is listed in `requirements.txt`.

## Installation
**cxflow** is available through the official PyPI repository; hence, the recommended installation is with `pip`:
```
pip install cxflow
```

## Usage
The installation process installs `cxflow` command which might be used simply from the command line.
Please refer to repository [Cognexa/cxMNIST](https://github.com/Cognexa/cxMNIST) for more information.

## Docs & Tutorials
The documentation and tutorials are yet to be done.

So far, the following documents are available:
- [introduction](tutorial)
- [cxflow hooks](cxflow/hooks/README.md)

## Extensions
**cxflow** is meant to be extremely lightweight.
For that reason the whole functionality is divided into various extensions with separate dependencies.

At the moment we support the following extensions:

- [cxflow-tensorflow](https://github.com/Cognexa/cxflow-tensorflow) - TensorFlow support
- [cxflow-scikit](https://github.com/Cognexa/cxflow-scikit) - scientific computations and statistics
- [cxflow-rethinkdb](https://github.com/Cognexa/cxflow-rethinkdb) - RethinkDB hook for training management with NoSQL (experimental)

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for more information.
