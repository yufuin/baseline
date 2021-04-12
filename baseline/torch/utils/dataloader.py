import torch
import torch.utils.data

from baseline.utils import pad

class Selector:
    def __init__(self, name, origin=None, mapping=None, dtype=None, device=None, padding=False, padding_value=0, padding_mask=False):
        assert not ((origin is not None) and (mapping is not None)), "cannot set both origin and mapping"
        self.name = name
        self._origin = origin
        self.mapping = mapping
        self.dtype = dtype
        self.device = device
        self.padding = padding
        self.padding_value = padding_value
        self.padding_mask = padding_mask
    @property
    def origin(self):
        if self._origin is not None:
            return self._origin
        elif self.mapping is not None:
            return None
        else:
            return self.name

    def select(self, instance):
        if self.mapping is not None:
            return self.mapping(instance)
        else:
            return instance[self.origin]

class SelectiveDataset(torch.utils.data.Dataset):
    def __init__(self, instances, selectors, sort_key=None):
        assert all(type(selector) in [Selector, dict] for selector in selectors)
        selectors = [selector if type(selector) is Selector else Selector(**selector) for selector in selectors]
        assert len(selectors) == len(set(s.name for s in selectors)), "cannot use a same name multiple times."

        self.instances = list(instances)
        self.selectors = list(selectors)
        self.sort_key = sort_key

    def __getitem__(self, idx):
        instance = self.instances[idx]
        return {selector.name:selector.select(instance) for selector in self.selectors}

    def __len__(self):
        return len(self.instances)

    def collate_fn(self, instances):
        if self.sort_key is not None:
            instances = sorted(instances, key=self.sort_key, reverse=True)

        outputs = dict()
        for selector in self.selectors:
            key = selector.name
            values = [instance[key] for instance in instances]

            if selector.padding:
                values, masks = pad(values, selector.padding_value)

                if selector.padding_mask:
                    masks = torch.FloatTensor(masks)
                    if selector.device is not None:
                        masks = masks.to(selector.device)
                    outputs[key + "_mask"] = masks

            if selector.dtype is not None:
                values = torch.tensor(values, dtype=selector.dtype)
                if selector.device is not None:
                    values = values.to(selector.device)

            outputs[key] = values

        return outputs

    def dataloader(self, batch_size, shuffle, *args, **kwargs):
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn, *args, **kwargs)

"""
i1 = {"id":"instance1", "foo":32, "bar":[[1,2]]}
i2 = {"id":"instance2", "foo":50, "bar":[[10],[32],[5]]}
i3 = {"id":"instance3", "foo":43, "bar":[], "baz":-1}
instances = [i1,i2,i3,i1,i1,i1,i1]

device = torch.device("cpu")
#device = torch.device("cuda:0")
selectors = [
    Selector("id"),
    Selector("foo", dtype=torch.long),
    Selector("bar", dtype=torch.float, device=device, padding=True, padding_value=-7, padding_mask=True),
    Selector("hoge", origin="bar"),
    {"name":"fuga", "origin":"foo", "dtype":torch.float},
    {"name":"piyo", "mapping":lambda x:x["foo"]**2, "dtype":torch.long},
]
dataset = SelectiveDataset(instances, selectors, sort_key=lambda x:len(x["hoge"]))
"""
