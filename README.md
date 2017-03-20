# cxflow

This is an official repository of **cxflow** - a smart manager and personal trainer of TensorFlow models.

## Development Status

- **Status**: Regular Project
- **Master Developer**: Petr Bělohlávek

## Example
For example usage of **cxflow** please refer to a dedicated repository [Cognexa/mnist-example](https://gitlab.com/Cognexa/mnist-example).

## Requirements
The officially supported operating system is [Arch Linux](https://www.archlinux.org).
In addition, **cxflow** is tested on [Ubuntu 16.10](http://releases.ubuntu.com/16.10).
All operating systems are expected to be fully up-to-date.

The following environments are supported and tested:
- Python 3.6 with TensorFlow 1.0.1
- Python 3.5 with TensorFlow 0.12.1
- Python 3.5 with TensorFlow 0.11
- Python 3.5 with TensorFlow 0.10

List of Python package requirements is listed in `requirements.txt`.

## Installation
Installation to a [virtualenv](https://docs.python.org/3/library/venv.html) is suggested, however, completely optional. 

1. Clone the **cxflow** repository `$ git clone git@gitlab.com:Cognexa/cxflow.git`
2. Enter the directory `$ cd cxflow`
3. **Optional**: *install any specific version of packages (e.g. TensorFlow). If this step is skipped, the up-to-date version will be installed by default in the next step.*
4. Install **cxflow** dependencies `$ pip install -r requirements.txt`. If forced update of all previously installed packages is required, `-U` flag should be used.
5. Install **cxflow** `$ pip install .`. For development purposes, an additional `-e` flag should be used.

## Usage
The installation process installs `cxflow` command which might be used simply from the command line.
Please refer to repository [Cognexa/mnist-example](https://gitlab.com/Cognexa/mnist-example) for more information.

In addition, `cxgridsearch` command is installed.

## Testing
Unit tests might be run by `$ python setup.py test`.

## License
MIT License
