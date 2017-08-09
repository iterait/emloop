"""cxflow module containing various constants."""

# cxflow logging formats
CXF_LOG_FORMAT = '%(asctime)s.%(msecs)06d: %(levelname)-8s@%(module)-15s: %(message)s'
CXF_LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
CXF_FULL_DATE_FORMAT = '%Y-%m-%d %H:%M:%S.%f'

# Module with standard cxflow hooks (as would be used in import).
CXF_HOOKS_MODULE = 'cxflow.hooks'

# configuration file name
CXF_CONFIG_FILE = 'config.yaml'
CXF_LOG_FILE = 'train.log'
