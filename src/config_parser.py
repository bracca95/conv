from __future__ import annotations

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
    pool: Optional[int]

    output_size: Optional[int] = None

    @staticmethod
    def conv(layer: Layer, input_size: int, layer_number: int = 0) -> int:
        layer.output_size = np.floor((input_size - layer.ksize + 2 * layer.padding) / layer.stride).astype(int) + 1

        if layer.output_size is None:
            Logger.instance().critical("something went wrong while computing the output size! (None)")
            raise ValueError("something went wrong while computing the output size! (None)")

        if layer.pool is not None:
            pool = layer.pool if layer.pool % 2 == 0 else layer.pool + 1
            layer.output_size //= 2
        
        Logger.instance().debug(f"\n## LAYER {layer_number} ##\noutput size: {layer.output_size}")
        return layer.output_size

    @staticmethod
    def transposed(layer: Layer, input_size: int, layer_number: int=0) -> int:
        layer.output_size = (input_size - 1) * layer.stride - 2 * (layer.padding) + layer.ksize
        Logger.instance().debug(f"\n## LAYER {layer_number} ##\noutput size: {layer.output_size}")

        return layer.output_size

    @staticmethod
    def chain_operation(op: Callable[[Layer, int, int], int], og_img_size: int, layers: List['Layer']):
        i = 0

        while i < len(layers):
            if i == 0:
                op(layers[i], og_img_size, i)
            else:
                op(layers[i], layers[i-1].output_size, i)
            i += 1      

    @classmethod
    def deserialize(cls, obj: Any) -> 'Layer':
        try:
            Utils.check_instance(obj, dict)
            ksize = from_int(obj.get(CONFIG_KSIZE))
            padding = from_int(obj.get(CONFIG_PADDING))
            stride = from_int(obj.get(CONFIG_STRIDE))
            pool = from_union([from_none, from_int], obj.get(CONFIG_POOL))

            if pool == 0:
                pool = None
                Logger.instance().warning("pool is interpreted as None")
            if stride == 0:
                raise ZeroDivisionError("stride value cannot be 0.")
        except TypeError as te:
            Logger.instance().critical(te.args)
            sys.exit(-1)

        Logger.instance().info("Layer deserialized correctly")
        return Layer(ksize, padding, stride, pool)

    def serialize(self) -> dict:
        result: dict = {}

        result[CONFIG_KSIZE] = from_int(self.ksize)
        result[CONFIG_PADDING] = from_int(self.padding)
        result[CONFIG_STRIDE] = from_int(self.stride)
        result[CONFIG_POOL] = from_union([from_none, from_int], self.pool)
        
        Logger.instance().info("Serializing Layers...")
        return result


@dataclass
class Config:
    mode: str
    input_img_size: int
    layers: List[Layer]

    @classmethod
    def deserialize(cls, str_path: str) -> 'Config':
        obj = Utils.read_json(str_path)
        
        try:
            mode = from_str(obj.get(CONFIG_MODE))
            input_img_size = from_int(obj.get(CONFIG_INPUT_IMG_SIZE))
            layers = from_list(Layer.deserialize, obj.get(CONFIG_LAYERS))
        except TypeError as te:
            Logger.instance().critical(te.args)
            sys.exit(-1)
        
        Logger.instance().info("Config deserialized correctly")
        return Config(mode, input_img_size, layers)

    def serialize(self, directory: str, filename: str):
        result: dict = {}
        dire = None

        try:
            dire = Utils.validate_path(directory)
        except FileNotFoundError as fnf:
            Logger.instance().critical(f"{fnf.args}")
            sys.exit(-1)
        
        result[CONFIG_MODE] = from_str(self.mode)
        result[CONFIG_INPUT_IMG_SIZE] = from_int(self.input_img_size)
        result[CONFIG_LAYERS] = from_list(lambda x: to_class(Layer, x), self.layers)
        
        Logger.instance().info("Serializing Config...")
        with open(os.path.join(dire, filename), "w") as f:
            json_dict = json.dumps(result, indent=4)
            f.write(json_dict)


def config_to_json(x: Config) -> Any:
    return to_class(Config, x)
