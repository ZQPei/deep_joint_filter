import os
import yaml
from easydict import EasyDict as edict

class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """
    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert(os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                cfg_dict.update(yaml.load(fo.read()))
                cfg_dict['config_file'] = os.path.abspath(config_file)
                cfg_dict['config_name'] = os.path.splitext(os.path.basename(config_file))[0]
                cfg_dict['config_path'] = os.path.abspath(os.path.dirname(config_file))
                cfg_dict['current_path'] = os.path.curdir

        super(YamlParser, self).__init__(cfg_dict)



if __name__ == "__main__":
    cfg = YamlParser(config_file="../config.yml")
    import ipdb; ipdb.set_trace()