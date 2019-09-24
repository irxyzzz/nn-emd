import random
import zlib
import base64
import json
import logging

import gmpy2 as gp

logger = logging.getLogger(__name__)

def _random(maximum, bits):
    rand_function = random.SystemRandom()
    r = gp.mpz(rand_function.getrandbits(bits))
    while r > maximum:
        r = gp.mpz(rand_function.getrandbits(bits))
    return r

def _random_generator(bits, p, r):
    while True:
        h = _random(p, bits)
        g = gp.powmod(h, r, p)
        if not g == 1:
            break
    return g

def _random_prime(bits):
    rand_function = random.SystemRandom()
    r = gp.mpz(rand_function.getrandbits(bits))
    r = gp.bit_set(r, bits - 1)
    return gp.next_prime(r)

def _param_generator(bits, r=2):
    while True:
        q = _random_prime(bits - 1)
        p = q * 2 + 1
        if gp.is_prime(p) and gp.is_prime(q):
            break;
    return p, q, r

def _json_zip(store_dict):
    return base64.b64encode(zlib.compress(
                json.dumps(store_dict).encode('utf-8')
    )).decode('ascii')

def _json_unzip(content):
    try:
        dec_compress = zlib.decompress(base64.b64decode(content.encode('ascii'))).decode('utf-8')
    except Exception:
        raise RuntimeError("Could not decode/unzip the contents")
    try:
        return json.loads(dec_compress)
    except Exception:
        raise RuntimeError("Could interpret the unzipped contents")

def generate_config_files(sec_param, sec_param_config, dlog_table_config, func_bound):
    p, q, r = _param_generator(sec_param)
    g = _random_generator(sec_param, p, r)
    group_info = {
        'p': gp.digits(p),
        'q': gp.digits(q),
        'r': gp.digits(r)
    }
    sec_param_dict = {'g': gp.digits(g), 'sec_param': sec_param, 'group': group_info}

    with open(sec_param_config, 'w') as outfile:
        json.dump(sec_param_dict, outfile)
    logger.info('Generate secure parameters config file successfully, see file %s' % sec_param_config)

    dlog_table = dict()
    bound = func_bound + 1
    for i in range(bound):
        dlog_table[gp.digits(gp.powmod(g, i, p))] = i
    for i in range(-1, -bound, -1):
        dlog_table[gp.digits(gp.powmod(g, i, p))] = i

    dlog_table_dict = {
        'g': gp.digits(g),
        'func_bound': func_bound,
        'dlog_table': dlog_table
    }

    with open(dlog_table_config, 'w') as outfile:
        # outfile.write(_json_zip(dlog_table_dict))
        json.dump(dlog_table_dict, outfile)
    logger.info('Generate dlog table config file successfully, see file %s' % dlog_table_config)


def load_sec_param_config(sec_param_config_file):
    with open(sec_param_config_file, 'r') as infile:
        sec_param_dict = json.load(infile)

        p = gp.mpz(sec_param_dict['group']['p'])
        q = gp.mpz(sec_param_dict['group']['q'])
        r = gp.mpz(sec_param_dict['group']['r'])
        g = gp.mpz(sec_param_dict['g'])
        sec_param = sec_param_dict['sec_param']

    return p, q, r, g, sec_param

def load_dlog_table_config(dlog_table_config_file):
    with open(dlog_table_config_file, 'r') as infile:
        # config_content = infile.read()
        # store_dict = _json_unzip(config_content)
        store_dict = json.load(infile)

        dlog_table = store_dict['dlog_table']
        func_bound = store_dict['func_bound']
        g = gp.mpz(store_dict['g'])

    return {
        'dlog_table': dlog_table,
        'func_bound': func_bound,
        'g': g
    }
