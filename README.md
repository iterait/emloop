# cxflow

This is a repository of **cxflow** - a smart manager and personal trainer of TensorFlow models.

## Development Status
This software is in heavy pre-alpha stage.
**Usage of this software is not recommended** since the back-compatibility will not be maintained.

## Example
For example usage of **cxflow** please refer to a dedicated repository [Cognexa/cxflow-demo](https://gitlab.com/Cognexa/cxflow-demo).

## Installation
So far, **Python 3.5** is required. Support of Python 3.6 and newer will be provided soon.
Usage of virtual environment is recommended.
Only development installation is supported.
Tested on ArchLinux.

1. Create virtualenv directory `$ mkdir ~/virtualenv`
2. Create new virtualenv `$ python3.5 -m venv ~/virtualenv/myvenv`
3. Activate the new virtualenv `$ source ~/virtualenv/myvenv/bin/activate`
4. Clone the **cxflow** repository `(myvenv) $ git clone git@gitlab.com:Cognexa/cxflow.git`
5. Enter the directory `(myvenv) $ cd cxflow`
6. **Optional**: *if you want to run some special version of TensorFlow or just to install it by yourself (e.g. when pip installation in the next step fails), do it now. The following TensorFlow versions were tested: 0.10.0, 0.11.0, 0.12.1, 1.0.0*
7. Install **cxflow** dependencies `(myvenv) $ pip install -r requirements.txt`
8. Install **cxflow**  `(myvenv) $ pip install -e .`

## Usage
The installation process installs `cxflow` command which might be used simply from the command line.
Please refer to repository [Cognexa/mnist-example](https://gitlab.com/Cognexa/mnist-example) for more information.

## Testing
Unit tests might be run by `(myvenv) $ python setup.py test`.
Note that so far the unit tests are missing.

## License
MIT License
