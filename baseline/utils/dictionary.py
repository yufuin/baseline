import json

class BasicDictionary:
    def __init__(self, unk_symbol):
        self._unk_symbol = unk_symbol
        self._finalized = False

        self.dic = {self.unk_symbol:self.unk_symbol_id}
        self.reverse_dic = {self.unk_symbol_id:self.unk_symbol}

    @property
    def size(self):
        return len(self.dic)
    def __len__(self):
        return self.size

    @property
    def unk_symbol_id(self):
        return 0
    @property
    def unk_symbol(self):
        return self._unk_symbol

    def add(self, key):
        assert not self.finalized
        if type(key) in [list, tuple]:
            return [self.add(k) for k in key]
        else:
            if key in self.dic:
                return self.dic[key]
            else:
                value = len(self.dic)
                self.dic[key] = value
                self.reverse_dic[value] = key
                return value
    def get(self, key):
        if type(key) in [list, tuple]:
            return [self.get(k) for k in key]
        else:
            return self.dic.get(key, self.unk_symbol_id)

    def encode(self, key, allow_new):
        if allow_new:
            return self.add(key)
        else:
            return self.get(key)
    def decode(self, value):
        if type(value) in [list, tuple]:
            return [self.decode(v) for v in value]
        else:
            return self.reverse_dic[value]

    def dump(self):
        return dict(self.dic)
    def load(self, dic):
        self.dic = dict(dic)
        self.reverse_dic = {value:key for key,value in self.dic.items()}

        self._unk_symbol = self.reverse_dic[self.unk_symbol_id]
        self._finalized = False

    def finalize(self):
        # this is only for self.add
        self._finalized = True
    @property
    def finalized(self):
        return self._finalized


class BILOUDictionary:
    # for BILOU tagged labels
    # this dictionary should hold transition information
    pass

class MultiDictionary:
    # d = MultiDictionary(words="basic", POSs="basic", "entities"="bilou")
    # train = d.encode(json.load(open("train.json")), allow_new=True)
    pass
