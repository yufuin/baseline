import numpy as np
import torch

# https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/networks/large_hourglass.py
def create_ctdet_coco_hourglass_net(get_large_hourglass_net_func, pretrained_model_path=None):
    num_layers = 0
    heads = {"hm":80, "wh":2, "reg": 2}
    head_conv = 64
    module = get_large_hourglass_net_func(num_layers=num_layers, heads=heads, head_conv=head_conv)
    model = CtdetCocoHourglassNet(module=module)

    if pretrained_model_path is not None:
        full_state_dict = torch.load(pretrained_model_path, map_location=torch.device("cpu")) # keys=["epoch", "state_dict"]
        state_dict = full_state_dict["state_dict"]
        model.load_state_dict(state_dict)

    return model

class CtdetCocoHourglassNet(torch.nn.Module):
    def __init__(self, module):
        super(CtdetCocoHourglassNet, self).__init__()
        self.module = module

        self.resolution = (512,512)
        self.mean = torch.FloatTensor([0.408, 0.447, 0.470])
        self.std = torch.FloatTensor([0.289, 0.274, 0.278])
        self.num_classes = 80

    def forward(self, images: "[B,3(bgr),512,512] float32(\\in [0,1])"):
        return self.module(images)

    class_name = (
        'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    )

    def preprocess(self, images:"[B,3,512,512] float32(\\in [0,1])", color:"'rgb' or 'bgr'"="rgb"):
        assert color in ["rgb", "bgr"]
        assert images.shape[-3:] == torch.Size([3,512,512])
        images = (images - self.mean[:,None,None]) / self.std[:,None,None]
        if color == "rgb":
            images = images.flip(-3)
        return images

    def reset_input_convolution_layer(self, ksize=7, input_dim=3, stride=2):
        # default self.module.pre is nn.Sequential(convolution(7, 3, 128, stride=2), residual(3, 128, 256, stride=2))
        default_pre0_output_dim = self.module.pre[0].conv.out_channels
        pre_residual = self.module.pre[1]
        self.module.pre = torch.nn.Sequential(
            _centernet_convolution(k=ksize, inp_dim=input_dim, out_dim=default_pre0_output_dim, stride=stride),
            pre_residual
        )
        return self

    def reset_output_heatmap_layer(self, num_classes=80, std=0.002):
        # default heat-map heads (self.module.hm) are nn.ModuleList([nn.Sequential(convolution(3, cnv_dim, curr_dim, with_bn=False), nn.Conv2d(curr_dim, num_classes, (1, 1))), ...]) here len(hm)==#stacking-hourglass
        heat_map_heads = self.module.hm
        curr_dims = [head[0].conv.out_channels for head in heat_map_heads]
        assert len(set(curr_dims)) == 1
        first_convolutions = [head[0] for head in heat_map_heads]
        new_second_convolutions = [torch.nn.Conv2d(curr_dim, num_classes, (1,1)) for curr_dim in curr_dims]
        for conv in new_second_convolutions:
            torch.nn.init.normal_(conv.weight, mean=0.0, std=std)
            conv.bias.data.fill_(-2.19)
        if not hasattr(self, "olds"):
            self.olds = dict()
        self.olds["hm"] = [head[1] for head in heat_map_heads]

        self.module.hm = torch.nn.ModuleList([torch.nn.Sequential(first, second) for first, second in zip(first_convolutions, new_second_convolutions)])
        return self

    def reset_output_layer(self, key, output_dim, std=0.002):
        assert key != "hm"
        # default non-hm heads (self.module.hm) are nn.ModuleList([nn.Sequential(convolution(3, cnv_dim, curr_dim, with_bn=False), nn.Conv2d(curr_dim, out_dim, (1, 1))), ...]) here len(heads)==#stacking-hourglass
        heads = getattr(self.module, key)
        curr_dims = [head[0].conv.out_channels for head in heads]
        assert len(set(curr_dims)) == 1
        first_convolutions = [head[0] for head in heads]
        new_second_convolutions = [torch.nn.Conv2d(curr_dim, output_dim, (1,1)) for curr_dim in curr_dims]
        for conv in new_second_convolutions:
            torch.nn.init.normal_(conv.weight, mean=0.0, std=std)
        if not hasattr(self, "olds"):
            self.olds = dict()
        self.olds[key] = [head[1] for head in heads]

        new_heads = torch.nn.ModuleList([torch.nn.Sequential(first, second) for first, second in zip(first_convolutions, new_second_convolutions)])
        setattr(self.module, key, new_heads)
        return self

    def remove_head(self, key):
        del self.module.heads[key]
        delattr(self.module, key)


# https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/networks/large_hourglass.py
class _centernet_convolution(torch.nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(_centernet_convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = torch.nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = torch.nn.BatchNorm2d(out_dim) if with_bn else torch.nn.Sequential()
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu


# Detecting objects aspaired keypoints. (ECCV-2018)
# https://discuss.pytorch.org/t/discussion-about-the-paper-of-name-centernet/80658/2
# this function assumes xs ~= ys.
def size_adaptive_standard_deviation(xs, ys, iou=0.7, coeff=1/3):
    """
    x,y : object size (i.e. size of the boundary box).
    return : stddev for the supervision of CenterNet's heat map.
    """
    return _inside_radius(xs, ys, iou=iou) * coeff

def _inside_radius(xs, ys, iou=0.7):
    r_dsq = (xs+ys) / 4 - (( (((xs+ys)**2)/4) - xs*ys*(1-iou) )**0.5) / 2
    return r_dsq * (2**0.5)
def _outside_radius(xs, ys, iou=0.7):
    r_dsq = (-(xs+ys) / 4) + (( (((xs+ys)**2)/4) + xs*ys*((1-iou)/iou) )**0.5) / 2
    return r_dsq * (2**0.5)

def _test_inside_radius(x, y):
    r_dsq = _inside_radius(x, y) / (2**0.5)
    area = x * y
    inner_area = (x - r_dsq - r_dsq) * (y - r_dsq - r_dsq)
    real_iou = inner_area / area
    return (real_iou, area, inner_area, r_dsq)
def _test_outside_radius(x, y):
    r_dsq = _outside_radius(x, y) / (2**0.5)
    area = (x+r_dsq+r_dsq) * (y+r_dsq+r_dsq)
    inner_area = x * y
    real_iou = inner_area / area
    return (real_iou, area, inner_area, r_dsq)



"""
example1:
import centernet as C
import large_hourglass as L # https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/networks/large_hourglass.py

input_dim = 7
num_classes = 7
path = "ctdet_coco_hg.pth"
model = C.create_ctdet_coco_hourglass_net(L.get_large_hourglass_net, path)
model.reset_input_convolution_layer(input_dim=input_dim)
model.reset_output_heatmap_layer(num_classes=num_classes):

batch_size = 4
image_resolution = (512,512)
input_images = torzh.ones([batch_size, input_dim, *image_resolution])

heads = model(input_images) # 2x{"hm":heatmap.[B,num_classes,128,128], "wh":window size.[B,2,128,128], "reg":offset.[B,2,128,128]}

object_sizes = [[30, 20] for _ in range(batch_size)]
xs, ys = [v.squeeze() for v in np.split(np.array(object_sizes), 2,1)]
heatmap_stddevs = size_adaptive_standard_deviation(xs, ys)
[gold_0_x, gold_0_y] = [77,42]
gold_0_var = heatmap_stddevs[0] ** 2
gold_0 = [[np.exp(-((x-gold_0_x)**2+(y-gold_0_y)**2)/(2*gold_0_var)) for y in range(128)] for x in range(128)]

head_1, head_2 = heads
hm_loss = focal_loss([gold_0, gold_1,...], head_1["hm"]) + focal_loss([gold_0, gold_1,...], head_2["hm"])
...



example2:
input_dim = 3
original_color_order = "rgb"
num_classes = 7
path = "ctdet_coco_hg.pth"
model = C.create_ctdet_coco_hourglass_net(L.get_large_hourglass_net, path)
# no need to reset input conv.
model.reset_output_heatmap_layer(num_classes=num_classes):

input_images = get_input_image_by_rgb() # some loading function. float32. value \\in [0,1].
input_images = model.preprocess(images=input_images, color=original_color_order) # even if "bgr", still need to preprocess to normalize.

heads = model(input_images) # 2x{"hm":heatmap.[B,num_classes,128,128], "wh":window size.[B,2,128,128], "reg":offset.[B,2,128,128]}

"""





