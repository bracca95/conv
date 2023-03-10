import os
import sys
import json
import numpy as np

from dataclasses import dataclass
from typing import Optional, Union, List, Any, Callable, Iterable, Type, cast

from src.utils import Utils, Logger
from config.consts import *

def from_bool(x: Any) -> bool:
    Utils.check_instance(x, bool)
    return x

def from_int(x: Any) -> int:
    Utils.check_instance(x, int)
    return x

def from_str(x: Any) -> str:
    Utils.check_instance(x, str)
    return x

def from_none(x: Any) -> Any:
    Utils.check_instance(x, None)
    return x

def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    Utils.check_instance(x, list)
    return [f(y) for y in x]

def from_union(fs: Iterable[Any], x: Any):
    for f in fs:
        try:
            return f(x)
        except:
            pass
    raise TypeError(f"{x} should be one out of {[type(f) for f in fs]}")


def to_class(c: Type[T], x: Any) -> dict:
    Utils.check_instance(x, c)
    return cast(Any, x).to_dict()


@dataclass
class Layer:
    ksize: int
    padding: int
    stride: int

    output_size: Optional[int] = None

    def conv(self, input_size: int, layer_number: int = 0) -> int:
        self.output_size = np.floor((input_size - self.ksize + 2 * self.padding) / self.stride).astype(int) + 1
        Logger.instance().debug(f"\n## LAYER {layer_number} ##\noutput size: {self.output_size}")

        return self.output_size

    @classmethod
    def chain_conv(cls, og_img_size: int, layers: List['Layer']):
        i = 0

        while i < len(layers):
            if i == 0:
                layers[i].conv(og_img_size, i)
            else:
                layers[i].conv(layers[i-1].output_size, i)
            i += 1
        

    @classmethod
    def deserialize(cls, obj: Any) -> 'Layer':
        try:
            Utils.check_instance(obj, dict)
            ksize = from_int(obj.get(CONFIG_KSIZE))
            padding = from_int(obj.get(CONFIG_PADDING))
            stride = from_int(obj.get(CONFIG_STRIDE))

            if stride == 0:
                raise ZeroDivisionError("stride value cannot be 0.")
        except TypeError as te:
            Logger.instance().critical(te.args)
            sys.exit(-1)

        Logger.instance().info("Layer deserialized correctly")
        return Layer(ksize, padding, stride)

    def serialize(self) -> dict:
        result: dict = {}

        result[CONFIG_KSIZE] = from_int(self.ksize)
        result[CONFIG_PADDING] = from_int(self.padding)
        result[CONFIG_STRIDE] = from_int(self.stride)
        
        Logger.instance().info("Serializing Layers...")
        return result


@dataclass
class Config:
    input_img_size: int
    layers: List[Layer]

    @classmethod
    def deserialize(cls, str_path: str) -> 'Config':
        obj = Utils.read_json(str_path)
        
        try:
            input_img_size = from_int(obj.get(CONFIG_INPUT_IMG_SIZE))
            layers = from_list(Layer.deserialize, obj.get(CONFIG_LAYERS))
        except TypeError as te:
            Logger.instance().critical(te.args)
            sys.exit(-1)
        
        Logger.instance().info("Config deserialized correctly")
        return Config(input_img_size, layers)

    def serialize(self, directory: str, filename: str):
        result: dict = {}
        dire = None

        try:
            dire = Utils.validate_path(directory)
        except FileNotFoundError as fnf:
            Logger.instance().critical(f"{fnf.args}")
            sys.exit(-1)
        
        result[CONFIG_INPUT_IMG_SIZE] = from_int(self.input_img_size)
        result[CONFIG_LAYERS] = from_list(lambda x: to_class(Layer, x), self.layers)
        
        Logger.instance().info("Serializing Config...")
        with open(os.path.join(dire, filename), "w") as f:
            json_dict = json.dumps(result, indent=4)
            f.write(json_dict)


def config_to_json(x: Config) -> Any:
    return to_class(Config, x)
