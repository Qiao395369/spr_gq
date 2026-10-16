"""Complete historical teacher configurations before pretraining reconstruction.

New code may read fields that did not exist when an older ``config.json`` was
written.  Only missing fields are filled from today's defaults.  Recorded
values, including explicit ``None`` values, are never replaced. This helper is
only used for the teacher in pretraining, never for reload or student configs.
"""

import copy
import logging
from typing import Any, Dict, List, Tuple

from ml_collections import ConfigDict

from vmcnet.train.default_config import get_default_config


def _fill_missing(saved: Dict[str, Any], defaults: Dict[str, Any],
                  prefix: str, added: List[str]) -> Dict[str, Any]:
    result = copy.deepcopy(saved)
    for key, default_value in defaults.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in result:
            result[key] = copy.deepcopy(default_value)
            added.append(path)
        elif isinstance(result[key], dict) and isinstance(default_value, dict):
            result[key] = _fill_missing(result[key], default_value, path, added)
    return result


def _filled_leaf_values(path: str, value: Any):
    """Expand a newly added config subtree into individual parameter values."""
    if isinstance(value, dict) and value:
        for key, child in value.items():
            yield from _filled_leaf_values(f"{path}.{key}", child)
    else:
        yield path, value


def complete_config(
    saved_config: ConfigDict, source: str = "<in-memory>"
) -> Tuple[ConfigDict, Tuple[str, ...]]:
    """Return a completed copy and the names of fields that were filled.

    Saved ``model`` sections are narrowed to the selected model type, whereas
    the default config contains *all* model types for command-line flags.  Use
    only the selected type's defaults, so completion cannot insert an entire
    unrelated model tree into an old checkpoint config.
    """
    saved = saved_config.to_dict()
    defaults = get_default_config().to_dict()
    # Before these options existed, the ordinary gaoqiao network used the
    # original particle-type encoding and reused activation_type for Jastrow.
    # Today's defaults (element1 / tanh) are not necessarily equivalent for
    # an old checkpoint, even if all explicitly saved fields are preserved.
    saved_gq = saved.get("gq")
    if isinstance(saved_gq, dict) and saved_gq.get("type") != "so3":
        if "dp_type" not in saved_gq:
            defaults["gq"]["dp_type"] = "original"
        if "jastrow_activation_type" not in saved_gq:
            defaults["gq"]["jastrow_activation_type"] = saved_gq.get(
                "activation_type", defaults["gq"]["activation_type"])
    model = saved.get("model")
    if isinstance(model, dict):
        model_type = model.get("type", defaults["model"].get("type"))
        # A normal saved run contains only the selected model's fields.  A
        # freshly exported default config may still contain all model types.
        if model_type in defaults["model"] and model_type not in model:
            selected = copy.deepcopy(defaults["model"][model_type])
            selected["type"] = model_type
            defaults["model"] = selected
        else:
            # Unknown historical model: preserve it and do not invent fields.
            defaults["model"] = {}

    added: List[str] = []
    completed = _fill_missing(saved, defaults, "", added)
    if added:
        entries = []
        for path in added:
            value = completed
            for key in path.split("."):
                value = value[key]
            entries.extend(f"{name} = {item!r}"
                           for name, item in _filled_leaf_values(path, value))
        logging.warning("配置补全 [%s]:\n%s", source, "\n".join(entries))
    return ConfigDict(completed), tuple(added)
