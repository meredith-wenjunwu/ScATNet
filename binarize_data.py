from config.opts import get_config
from bin_data.binarize import main_binarize


if __name__ == "__main__":
    opts, parser = get_config()
    main_binarize(opts=vars(opts))
