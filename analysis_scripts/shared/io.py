import re
from pathlib import Path

import yaml


def load_config(**kwargs) -> dict:
    conf = yaml.safe_load(open("config.yaml"))
    conf = replace_templates(conf, flatten_dict(conf))

    for k, v in kwargs.items():
        conf[k] = v

    if "data_root" in conf.keys():
        conf["data_root"] = Path(conf["data_root"])

    if "report_root" in conf.keys():
        conf["report_root"] = Path(conf["report_root"])

    return conf


def replace_templates(conf, conf_flat):
    """
    Find <[^>]*> in the strings and replace with appropriate key val

    The template <some.thing[0]> might include an index [0] --> assuming
    that the value for some.thing is iterable, choose according to index
    """
    for k, v in conf.items():
        if isinstance(v, str):
            templates = re.findall("<([^>]*)>", v)

            for tmp in templates:
                try:
                    idx = re.findall(r"\[(\d*)\]", tmp)
                    tmp_stump = re.sub(r"\[(\d*)\]", "", tmp)
                    rval = conf_flat[tmp_stump]
                    rval = rval[int(idx[0])] if idx != [] else rval
                    v = v.replace(f"<{tmp}>", str(rval))
                except KeyError:
                    raise KeyError(
                        f"Template str <{tmp}> not pointing to "
                        " a valid key in config. Cannot replace!"
                    )
            conf[k] = v
        elif isinstance(v, dict):
            conf[k] = replace_templates(v, conf_flat)

    return conf


def flatten_dict(d_in):
    """flatten a dict if any of its values is a dict -> use key as prefix"""
    d = d_in.copy()
    # list as we do not want a generator
    kvals = [(k, v) for k, v in d.items()]
    for k, v in kvals:
        if isinstance(v, dict):
            new_d = {".".join([k, kv]): vv for kv, vv in flatten_dict(v).items()}
            d.pop(k)
            d.update(new_d)

    return d
