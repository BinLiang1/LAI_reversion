#!/usr/bin/python
# -*- coding : utf-8 -*-

import warnings


class Register(dict):

    def __init__(self, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self._dict = {}

    def register(self, target):
        def add_register_item(key, value):
            if not callable(value):
                raise Exception(
                    f"register object must be callable! But receice:{value} is not callable!")
            if key in self._dict:
                warnings.warn(
                    f"\033[33m{value.__name__} has been registered before, so we will overriden it\033[0m")
            self._dict[key] = value
            return value

        if callable(target):
            return add_register_item(target.__name__, target)
        else:
            return lambda x: add_register_item(target, x)

    def __call__(self, target):
        return self.register(target)

    def __setitem__(self, key, value):
        if not callable(value):
            raise ValueError(f"expected {str(value)} a callable")
        if key is None:
            key = value.__name__
        if key in self._dict:
            warnings.warn(f"{key} already exists, and will be replaced")
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return str(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()


CLASSIFIERS = Register()
