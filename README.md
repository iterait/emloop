# emloop
[![CircleCI](https://circleci.com/gh/iterait/emloop/tree/master.svg?style=shield)](https://circleci.com/gh/iterait/emloop/tree/master)
[![PyPI version](https://badge.fury.io/py/emloop.svg)](https://badge.fury.io/py/emloop)
[![Coverage 
Status](https://coveralls.io/repos/github/iterait/emloop/badge.svg?branch=master)](https://coveralls.io/github/iterait/emloop?branch=master)
[![Development Status](https://img.shields.io/badge/status-Regular-brightgreen.svg?style=flat)]()
[![MIT license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)
[![Master Developer](https://img.shields.io/badge/master-Petr%20Bělohlávek-lightgrey.svg?style=flat)]()

This is an official repository of **emloop** - a lightweight framework for machine learning with focus on modularization, re-usability and rapid experimenting.

## Installation
```
pip install emloop
```

## Quick start

- [10 minutes tutorial](https://emloop.org/tutorial) ([code](https://github.com/iterait/emloop-examples/tree/master/majority))
- [Documentation & API Reference](https://emloop.org/)
- [Additional examples](https://github.com/iterait/emloop-examples)

## Requirements
 - **emloop** is supported (and tested) on [Arch Linux](https://www.archlinux.org) and [Ubuntu](http://releases.ubuntu.com) (latest LTS and rolling) with Python 3.7 and 3.6, respectively.
 - **emloop** will most likely work on [Windows with Anaconda](https://www.anaconda.com/download/) and Python 3.6 or 3.7 yet **it is not tested regularly**

## Extensions
**emloop** is meant to be extremely lightweight.
For that reason the whole functionality is divided into various extensions with separate dependencies.

At the moment we support the following extensions:

- [emloop-tensorflow](https://github.com/iterait/emloop-tensorflow) - TensorFlow support
- [emloop-rethinkdb](https://github.com/iterait/emloop-rethinkdb) - RethinkDB hook for training management with NoSQL (experimental)

## Contributions

All contributions are welcomed. Please read our [contributor's guide](CONTRIBUTING.md).
