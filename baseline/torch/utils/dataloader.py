import torch
import torch.utils.data

from baseline.utils import pad

class Selecter:
    def __init__(self, name, origin=None, dtype=None, device=None, padding=False, padding_value=0, padding_mask=False):
        self.name = name
        self._origin = origin
        self.dtype = dtype
        self.device = device
        self.padding = padding
        self.padding_value = padding_value
        self.padding_mask = padding_mask
    @property
    def origin(self):
        return self._origin if self._origin is not None else self.name

class SelectiveDataset(torch.utils.data.Dataset):
    def __init__(self, instances, selecters, sort_key=None):
        assert all(type(selecter) in [Selecter, dict] for selecter in selecters)
        selecters = [selecter if type(selecter) is Selecter else Selecter(**selecter) for selecter in selecters]
        assert len(selecters) == len(set(s.name for s in selecters)), "the same name occurs multiple times."

        self.instances = list(instances)
        self.selecters = list(selecters)
        self.sort_key = sort_key

    def __getitem__(self, idx):
        instance = self.instances[idx]
        return {selecter.name:instance[selecter.origin] for selecter in self.selecters}

    def __len__(self):
        return len(self.instances)

    def collate_fn(self, instances):
        if self.sort_key is not None:
            instances = sorted(instances, key=self.sort_key, reverse=True)

        outputs = dict()
        for selecter in self.selecters:
            key = selecter.name
            values = [instance[key] for instance in instances]

            if selecter.padding:
                values, masks = pad(values, selecter.padding_value)

                if selecter.padding_mask:
                    masks = torch.FloatTensor(masks)
                    if selecter.device is not None:
                        masks = masks.to(selecter.device)
                    outputs[key + "_mask"] = masks

            if selecter.dtype is not None:
                values = torch.tensor(values, dtype=selecter.dtype)
                if selecter.device is not None:
                    values = values.to(selecter.device)

            outputs[key] = values

        return outputs

    def dataloader(self, batch_size, shuffle):
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)

"""
i1 = {"id":"instance1", "foo":32, "bar":[[1,2]]}
i2 = {"id":"instance2", "foo":50, "bar":[[10],[32],[5]]}
i3 = {"id":"instance3", "foo":43, "bar":[], "baz":-1}
instances = [i1,i2,i3,i1,i1,i1,i1]

device = torch.device("cpu")
#device = torch.device("cuda:0")
selecters = [
    Selecter("id"),
    Selecter("foo", dtype=torch.long),
    Selecter("bar", dtype=torch.float, device=device, padding=True, padding_value=-7, padding_mask=True),
    Selecter("hoge", origin="bar"),
    {"name":"piyo", "origin":"foo", "dtype":torch.float},
]
dataset = SelectiveDataset(instances, selecters, sort_key=lambda x:len(x["hoge"]))
"""
