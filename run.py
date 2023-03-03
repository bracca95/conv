from src.utils import Logger, Utils
from src.config_parser import Config, Layer

if __name__=="__main__":
	config_path = Utils.validate_path("config/config.json")
	config = Config.deserialize(config_path)

	layers = config.layers
	Layer.chain_conv(config.input_img_size, layers)

	Logger.instance().debug("program ended")