import torch
import torch.utils.data

from baseline.utils import pad

class Exporter:
    def __init__(self, origin, name=None, dtype=None, device=None, padding=False, padding_value=0, padding_mask=False):
        self.origin = origin
        self._name = name
        self.dtype = dtype
        self.device = device
        self.padding = padding
        self.padding_value = padding_value
        self.padding_mask = padding_mask
    @property
    def name(self):
        return self._name if self._name is not None else self.origin

class ExportableDataset(torch.utils.data.Dataset):
    def __init__(self, instances, exporters, sort_key=None):
        assert len(exporters) == len(set(e.name for e in exporters)), "the same name occurs multiple times."

        self.instances = list(instances)
        self.exporters = list(exporters)
        self.sort_key = sort_key

    def __getitem__(self, idx):
        instance = self.instances[idx]
        return {exporter.name:instance[exporter.origin] for exporter in self.exporters}

    def __len__(self):
        return len(self.instances)

    def collate_fn(self, instances):
        if self.sort_key is not None:
            instances = sorted(instances, key=self.sort_key, reverse=True)

        outputs = dict()
        for exporter in self.exporters:
            key = exporter.name
            values = [instance[key] for instance in instances]

            if exporter.padding:
                values, masks = pad(values, exporter.padding_value)

                if exporter.padding_mask:
                    masks = torch.FloatTensor(masks)
                    if exporter.device is not None:
                        masks = masks.to(exporter.device)
                    outputs[key + "_mask"] = masks

            if exporter.dtype is not None:
                values = torch.tensor(values, dtype=exporter.dtype)
                if exporter.device is not None:
                    values = values.to(exporter.device)

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
exporters = [
    Exporter("id"),
    Exporter("foo", dtype=torch.long),
    Exporter("bar", dtype=torch.float, device=device, padding=True, padding_value=-7, padding_mask=True),
    Exporter("bar", name="hoge"),
]
dataset = ExportableDataset(instances, exporters, sort_key=lambda x:len(x["hoge"]))
"""
