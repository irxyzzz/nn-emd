import logging

from crypto.utils import generate_config_files

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

sec_param_config_file = 'config/sec_param.json'
dlog_table_config_file = 'config/dlog_b8.json'

def test_generate_config_files():
    logger.info('testing generating config files')
    func_value_bound = 100000000
    sec_param = 256
    generate_config_files(sec_param, sec_param_config_file, dlog_table_config_file, func_value_bound)