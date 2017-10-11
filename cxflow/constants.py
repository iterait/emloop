"""**cxflow** module containing various constants."""

CXF_LOG_FORMAT = '%(asctime)s.%(msecs)06d: %(levelname)-8s@%(module)-15s: %(message)s'
"""General logging format."""

CXF_LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
"""Date format used in logging."""

CXF_FULL_DATE_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
"""Full date format."""

CXF_HOOKS_MODULE = 'cxflow.hooks'
"""Module with standard cxflow hooks (as would be used in import)."""

CXF_CONFIG_FILE = 'config.yaml'
"""Configuration file name (dumped in the output directory)."""

CXF_LOG_FILE = 'train.log'
"""Log file (dumped in the output directory)."""

CXF_TRACE_FILE = 'trace.yaml'
"""Training trace filename."""

CXF_TRAIN_STREAM = 'train'
"""Train stream name."""

CXF_PREDICT_STREAM = 'predict'
"""Predict stream name."""

__all__ = ['CXF_LOG_FORMAT', 'CXF_LOG_DATE_FORMAT', 'CXF_FULL_DATE_FORMAT', 'CXF_HOOKS_MODULE', 'CXF_CONFIG_FILE',
           'CXF_LOG_FILE', 'CXF_TRACE_FILE', 'CXF_TRAIN_STREAM', 'CXF_PREDICT_STREAM']
