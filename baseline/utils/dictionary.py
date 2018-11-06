import json

class Dictionary:
    _UNK_REPR = "$$UNK$$"

    def __init__(self, dump=None):
        if dump is None:
            self._repr_to_id = dict()
            self._id_to_repr = list()

            self._repr_to_id[self.unk_repr] = 0
            self._id_to_repr.append(self.unk_repr)
        else:
            self.load(dump=dump)

    @property
    def unk_repr(self):
        return self._UNK_REPR
    @property
    def unk_id(self):
        if not hasattr(self, "_unk_id"):
            self._unk_id = self.to_id(self.unk_repr, allow_new=False)
        return self._unk_id

    def dump(self):
        return json.dumps(self._id_to_repr)
    def load(self, dump):
        del self._unk_id
        self._id_to_repr = json.loads(dump)
        self._repr_to_id = {repr_:id_ for id_,repr_ in enumerate(self._id_to_repr)}

    def to_id(self, target_repr, allow_new):
        if target_repr in self._repr_to_id:
            return self._repr_to_id[target_repr]
        else:
            if allow_new:
                new_id = len(self)
                self._repr_to_id[target_repr] = new_id
                self._id_to_repr.append(target_repr)
                return new_id
            else:
                return self._repr_to_id[self.unk_repr]

    def to_id_recursive(self, target_reprs, allow_new):
        if isinstance(target_reprs, list):
            return [self.to_id_recursive(target_reprs=target_repr, allow_new=allow_new) for target_repr in target_reprs]
        elif isinstance(target_reprs, dict):
            return {key:self.to_id_recursive(target_reprs=value, allow_new=allow_new) for key, value in target_reprs.items()}
        else:
            return self.to_id(target_repr=target_reprs, allow_new=allow_new)

    def to_repr(self, target_id):
        assert target_id < len(self), "out of range index"
        return self._id_to_repr[target_id]

    def to_repr_recursive(self, target_ids):
        if isinstance(target_ids, list):
            return [self.to_repr_recursive(target_ids=target_id) for target_id in target_ids]
        elif isinstance(target_ids, dict):
            return {key:self.to_repr_recursive(target_ids=value) for key, value in target_ids.items()}
        else:
            return self.to_repr(target_id=target_ids)

    def __len__(self):
        len_repr_to_id = len(self._repr_to_id)
        len_id_to_repr = len(self._id_to_repr)
        assert len_repr_to_id == len_id_to_repr
        return len_repr_to_id

"""
class MultiDictionary:
    def __init__(self):
        pass

    def to_id(self, target, category, allow_new):
        target_dict = self.dicts[category]
        if target in target_dict:
            return target_dict[target]
        else:
            if allow_new:
                new_id = len(target_dict)
                target_dict[target] = new_id
                return new_id
            else:
                return target_dict[self.unk_repr]

    def to_id_recursive(self, targets, category, allow_new):
        if isinstance(targets, list):
            return [to_id_recursive(targets=target, category=category, allow_new=allow_new) for target in targets]
        elif isinstance(targets, dict):
            return {key:to_id_recursive(targets=value, category=category, allow_new=allow_new) for key, value in targets.items()}
        else:
            return to_id(target=targets, category=category, allow_new=allow_new)
"""
