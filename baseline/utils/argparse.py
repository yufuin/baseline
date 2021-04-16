import argparse as _A
import dataclasses as _D
import typing as _T

DataclassType = _T.TypeVar("DataclassType")

class DataclassArgumentParser(_A.ArgumentParser, _T.Generic[DataclassType]):
    """
    variable name: treated as "dest"
    metadata:
    - args: treated as aliases.
    - default: default value. priority=> field.default > field.default_factory > field.metadata["default"]

    - choices: choices
    - required: required
    - help: help

    - type, dest, action: prohibited.
    """
    def __init__(self, dataclass:DataclassType, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dataclass = dataclass
        self._add_argument_from_dataclass(self._dataclass)

    def _add_argument_from_dataclass(self, dataclass:DataclassType) -> None:
        for field in _D.fields(dataclass):
            argparams = list()
            argkwparams = dict()

            assert "dest" not in field.metadata, field
            assert "type" not in field.metadata, field
            assert "action" not in field.metadata, field
            if "args" in field.metadata:
                assert type(field.metadata["args"]) in {list, tuple}, field
                assert all(a[0] == "-" for a in field.metadata["args"]), f"must be a non-positional argument: {field}"

            doesnt_have_default = (
                (field.default == _D.MISSING)
                and (field.default_factory == _D.MISSING)
                and ("default" not in field.metadata)
            )
            if not doesnt_have_default:
                default_value = (
                    field.default if field.default != _D.MISSING
                    else field.default_factory() if field.default_factory != _D.MISSING
                    else field.metadata["default"]
                )

            if "help" in field.metadata:
                argkwparams["help"] = field.metadata["help"]

            if field.type is bool:
                raise NotImplementedError()
            else:
                argkwparams["type"] = field.type

                is_positional = ("args" not in field.metadata) and doesnt_have_default

                if is_positional:
                    raise NotImplementedError()

                else:
                    argkwparams["dest"] = field.name
                    if "args" in field.metadata:
                        argparams = field.metadata["args"]
                    else:
                        argparams = ["--"+field.name]
                    for key in ["choices", "required"]:
                        if key in field.metadata:
                            argkwparams[key] = field.metadata[key]
                    if not doesnt_have_default:
                        argkwparams["default"] = default_value

                print(argparams, argkwparams)
                self.add_argument(*argparams, **argkwparams)

    def parse_args(self, *args, **kwargs) -> DataclassType:
        namespace_args = super(DataclassArgumentParser, self).parse_args(*args, **kwargs)
        return self._dataclass(**vars(namespace_args))

