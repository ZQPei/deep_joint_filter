import os
import yaml
from easydict import EasyDict as edict

class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """
    def __init__(self, cfg_dict=None, config_path=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_path is not None:
            assert(os.path.isfile(config_path))
            with open(config_path, 'r') as fo:
                cfg_dict.update(yaml.load(fo.read()))
                cfg_dict['config_path'] = os.path.abspath(os.path.dirname(config_path))
        
        super(YamlParser, self).__init__(cfg_dict)



if __name__ == "__main__":
    cfg = YamlParser(config_path="../config.yml")
    import ipdb; ipdb.set_trace()