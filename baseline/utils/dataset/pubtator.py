from typing import List as _List, Any as _Any, Union as _Union, Optional as _Optional
import re as _re

class TableRow:
    def __init__(self, row_values, table):
        self.row_values = row_values
        self._table = table
    def __getitem__(self, key):
        return self.row_values[self._table._key_to_index[key]]
    def __iter__(self):
        return ([key,value] for key,value in zip(self._table.keys(), self.row_values))
    def __contains__(self, item):
        return item in self.keys()

    def __str__(self):
        return str(self.row_values)

    def keys(self):
        return self._table.keys()
    def values(self):
        return self.row_values
    def items(self):
        return iter(self)
    def get(self, key, default_value=None):
        if key in self:
            return self[key]
        else:
            return default_value

class Table:
    def __init__(self, data:_Union[_List[_Any], _List[dict]], keys:_Optional[_List[_Any]]=None):
        """
        Each row of data should be a (non-dict-like) iterable or a dict-like object.
        Precisely, a row is expected to be a dict-like object if it has the "keys" attribute. Otherwise it will be treated as a non-dict-like iterable.

        "keys" can be None if a row of data is of a dict-like object.
        """
        need_to_infer_keys = keys is None
        is_row_dict_like = None # boolean, will be set at first iteration of rows.

        if keys is not None:
            keys = tuple(keys)

        values = list()
        for row in data:
            if is_row_dict_like is None:
                is_row_dict_like = hasattr(row, "keys")
            else:
                assert is_row_dict_like == hasattr(row, "keys"), "rows of data must be all non-dict-like iterables or all dict-like."

            if need_to_infer_keys:
                assert is_row_dict_like, "to infer keys, each row must be a dict-like."
                keys = tuple(row.keys())
                need_to_infer_keys = False

            if is_row_dict_like:
                row_values = [row[key] for key in keys]
            else:
                row_values = list(row)
            values.append(row_values)

        assert not need_to_infer_keys, "if keys are not provided, data should have at least one row."

        self._keys = keys
        self._key_to_index = {key:idx for idx,key in enumerate(keys)}
        self.data = values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if type(idx) is slice:
            raise NotImplementedError("slice idx")

        return TableRow(self.data[idx], self)
    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def keys(self):
        return self._keys

    def to_dataframe(self):
        import pandas as pd
        df = pd.DataFrame(data=self.data, columns=self.keys())
        return df


_ptn_text = _re.compile("^([^\t\|]+)\|([^\t\|]+)\|(.*)$")
_ptn_entity = _re.compile("^([^\t\|]+)\t([0-9]+)\t([0-9]+)(.*)$")

def _format_doc(doc:dict):
    out = dict(doc)
    out["ents"] = Table(doc["ents"])
    return out

def load_pubtator_file(file, explode_entity:bool=True, entity_sep:str=";"):
    docs = list()
    passed_ids = set()

    current_id = None
    current_doc = {"ents":list()}
    for line in file:
        line = line.rstrip()
        if len(line) == 0:
            if current_id is not None:
                docs.append(_format_doc(current_doc))
                current_id = None
                current_doc = {"ents":list()}
            continue

        match_text = _ptn_text.match(line)
        if match_text:
            id_, tag, text = match_text.groups()
            if current_id is None:
                assert current_id not in passed_ids
                current_id = id_
                passed_ids.add(id_)
                current_doc["id"] = id_
            else:
                assert id_ == current_id
            assert tag not in current_doc
            current_doc[tag] = text

        else:
            match_entity = _ptn_entity.match(line)
            assert match_entity
            id_, start, end, infos = match_entity.groups()
            assert id_ == current_id

            ent = dict()
            start, end = map(int, [start, end])
            ent["start"] = start
            ent["end"] = end

            infos = infos[1:].rstrip() # skip first \t then rstrip
            if len(infos) > 0:
                infos = infos.split("\t")
                assert len(infos) <= 3
                for value, key in zip(infos, ["surface", "type", "entity"]):
                    ent[key] = value

            if explode_entity and ("entity" in ent):
                for exploded_ent_id in ent["entity"].split(entity_sep):
                    exploded_ent = dict(ent)
                    exploded_ent["entity"] = exploded_ent_id
                    current_doc["ents"].append(exploded_ent)
            else:
                current_doc["ents"].append(ent)

    if current_id is not None:
        docs.append(_format_doc(current_doc))

    return docs

