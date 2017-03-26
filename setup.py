from setuptools import setup

setup(name='cxflow',
      version='0.3',
      description='Smart TensorFlow trainer',
      long_description='Trainer of TensorFlow models that automatically manages the whole process of training,'
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
      keywords='tensorflow wrapper',
      url='https://github.com/Cognexa/cxflow',
      author='Petr Belohlavek',
      author_email='me@petrbel.cz',
      license='MIT',
      packages=['cxflow'],
      include_package_data=True,
      zip_safe=False,
      test_suite='cxflow.tests',
      entry_points={
          'console_scripts': [
              'cxflow=cxflow.entry_point:entry_point',
              'cxgridsearch=cxflow.grid_search:init_grid_search',
          ]
      })
