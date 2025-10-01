from typing import Any
import yaml


class Config(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = Config(value)
            elif isinstance(value, list):
                self[key] = [
                    Config(item) if isinstance(item, dict) else item for item in value
                ]

    def _substitute_values(self, full_config=None):
        if full_config is None:
            full_config = self

        for key, value in self.items():
            if isinstance(value, str):
                try:
                    while "{" in self[key] and "}" in self[key]:
                        formatted_value = self[key].format(**full_config)
                        if formatted_value == self[key]:  # Break if no change
                            break
                        self[key] = formatted_value
                except (KeyError, IndexError):
                    pass
            elif isinstance(value, Config):
                value._substitute_values(full_config)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Config):
                        item._substitute_values(full_config)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"No such attribute: {key}")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"No such attribute: {key}")

    def __repr__(self):
        return f"Config({super().__repr__()})"

    def __display_tree(self, indent=0):
        output = []
        for key, value in self.items():
            output.append("  " * indent + f"{key}:")
            if isinstance(value, Config):
                output.append(value.__display_tree(indent + 1))
            elif isinstance(value, list):
                output.append("  " * (indent + 1) + "[")
                for item in value:
                    if isinstance(item, Config):
                        output.append(item.__display_tree(indent + 2))
                    else:
                        output.append("  " * (indent + 2) + str(item))
                output.append("  " * (indent + 1) + "]")
            else:
                output[-1] += f" {value}"
        return "\n".join(output)

    def __str__(self):
        return self.__display_tree()


def get_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    config_ = Config(raw_config)
    config_._substitute_values()
    return config_
