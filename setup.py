from setuptools import setup

tests_require = [
    'pytest',
]

setup(name='emloop',
      version='0.2.1',
      description='Smart machine learning trainer',
      long_description='Trainer of machine learning models that automatically manages the whole process of training,'
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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
      ],
      keywords='machine learning training framework',
      url='https://github.com/iterait/emloop',
      author='Iterait a.s.',
      author_email='hello@iterait.com',
      license='MIT',
      packages=['emloop',
                'emloop.cli',
                'emloop.datasets',
                'emloop.hooks',
                'emloop.models',
                'emloop.utils',
                'emloop.tests'
                ],
      include_package_data=True,
      zip_safe=False,
      setup_requires=['pytest-runner'],
      tests_require=tests_require,
      install_requires=[
        'click>=7.0',
        'numpy>=1.16',
        'requests>=2.21',
        'ruamel.yaml>=0.15',
        'tabulate>=0.8.3',
        'babel>=2.6',
        'matplotlib>=3.0',
        'pandas>=0.24',
      ],
      extras_require={
          'docs': [
              'sphinx>=2.0',
              'autoapi>=1.4',
              'sphinx-argparse',
              'sphinx-autodoc-typehints',
              'sphinx-bootstrap-theme',
          ],
          'tests': tests_require,
      },
      entry_points={
          'console_scripts': [
              'emloop=emloop.entry_point:entry_point'
          ]
      })
