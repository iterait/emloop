"""**emloop** module containing various constants."""

EL_LOG_FORMAT = '%(asctime)s.%(msecs)03d: %(levelname)-8s@%(module)-12s: %(message)s'
"""General logging format."""

EL_LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
"""Date format used in logging."""

EL_FULL_DATE_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
"""Full date format."""

EL_HOOKS_MODULE = 'emloop.hooks'
"""Module with standard emloop hooks (as would be used in import)."""

EL_CONFIG_FILE = 'config.yaml'
"""Configuration file name (dumped in the output directory)."""

EL_DEFAULT_LOG_DIR = './log'
"""Default log directory."""

EL_LOG_FILE = 'train.log'
"""Log file (dumped in the output directory)."""

EL_TRACE_FILE = 'trace.yaml'
"""Training trace filename."""

EL_PREDICT_STREAM = 'predict'
"""Predict stream name."""

EL_NA_STR = 'N/A'
"""N/A string for pretty printing."""

EL_BUFFER_SLEEP = 0.02
"""The duration for which the buffer sleeps before it starts to process the next batch."""

EL_DEFAULT_TRAIN_STREAM = 'train'
"""The stream to be used for training."""

__all__ = ['EL_LOG_FORMAT', 'EL_LOG_DATE_FORMAT', 'EL_FULL_DATE_FORMAT', 'EL_HOOKS_MODULE', 'EL_CONFIG_FILE',
           'EL_LOG_FILE', 'EL_TRACE_FILE', 'EL_DEFAULT_TRAIN_STREAM', 'EL_PREDICT_STREAM', 'EL_DEFAULT_LOG_DIR',
           'EL_NA_STR', 'EL_BUFFER_SLEEP']
