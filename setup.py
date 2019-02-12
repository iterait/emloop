from setuptools import setup

setup(name='emloop',
      version='0.1.2',
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
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
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
      tests_require=['pytest'],
      install_requires=[line for line in open('requirements.txt', 'r').readlines() if not line.startswith('#')],
      entry_points={
          'console_scripts': [
              'emloop=emloop.entry_point:entry_point'
          ]
      })
