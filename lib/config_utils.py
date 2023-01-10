import io
import configparser
from omegaconf import DictConfig, OmegaConf


def read_ini(fpath: str) -> configparser.ConfigParser:
    """ Append a [root] as base header. """
    config = configparser.ConfigParser()
    with open(fpath, 'r') as fp:
        lines = fp.readlines()
    content = '[root]\n' + ''.join(lines)
    config.read_string(content)
    return config


def write_ini(config: configparser.ConfigParser, fpath: str):
    """ 
    Remove the header to match colmap format. 
    """
    with io.StringIO() as fp, open(fpath, 'w') as ofp:
        config.write(fp)
        fp.seek(0)
        content = fp.readlines()[1:]  # Get rid of header
        # content = [v.replace(' ', '') for v in content if v != '\n']
        ofp.writelines(content)


def ini_to_yaml(config: configparser.ConfigParser) -> DictConfig:
    return OmegaConf.create(
        {section: dict(config[section]) for section in config.sections()}
    )
    

def print_cfg(config: configparser.ConfigParser):
    print(
        {section: dict(config[section]) for section in config.sections()}
        )