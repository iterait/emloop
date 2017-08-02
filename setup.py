from pip.req import parse_requirements
from setuptools import setup

setup(name='cxflow',
      version='0.7.0',
      description='Smart neural network trainer',
      long_description='Trainer of neural network models that automatically manages the whole process of training,'
                       'saving and restoring models and much more',
      classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: Unix',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
      ],
      keywords='neural networks training framework',
      url='https://github.com/cognexa/cxflow',
      author='Cognexa Solutions s.r.o.',
      author_email='info@cognexa.com',
      license='MIT',
      packages=['cxflow',
                'cxflow.datasets',
                'cxflow.hooks',
                'cxflow.nets',
                'cxflow.utils',
                'cxflow.tests'
                ],
      include_package_data=True,
      zip_safe=False,
      test_suite='cxflow.tests',
      install_requires=[str(ir.req) for ir in parse_requirements('requirements.txt', session='hack')],
      entry_points={
          'console_scripts': [
              'cxflow=cxflow.entry_point:entry_point'
          ]
      })
