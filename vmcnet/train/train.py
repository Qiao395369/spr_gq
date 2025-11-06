from absl import app
from absl import flags
import logging
from .parse_config_flags import parse_flags
from .runners import run_molecule
from ml_collections.config_flags import config_flags

# internal imports

FLAGS = flags.FLAGS

def main(_):
  reload_config, config = parse_flags(FLAGS)
  root_logger = logging.getLogger()
  root_logger.setLevel(config.logging_level)
  run_molecule(reload_config, config)


if __name__ == '__main__':
  app.run(main)
