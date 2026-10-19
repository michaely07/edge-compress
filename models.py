"""CIFAR-adapted torchvision models.

Torchvision models assume 224px ImageNet images. For 32px (CIFAR) and 64px (drone crops)
inputs we shrink the stem so the feature maps don't collapse.
ResNet-50 is the teacher; ResNet-18 and MobileNetV2 are the students.
"""
import torch.nn as nn
import torchvision.models as tvm
from torchvision.models.resnet import BasicBlock as _TVBasic, Bottleneck as _TVBottleneck


# torchvision's residual blocks call one `self.relu` module at every activation site. PT2E's
# X86InductorQuantizer finds conv+add+relu patterns by source module, so the shared ReLU becomes one
# partition with several outputs and annotation fails ("Input partition has more than one output
# node"). These blocks give each call site its own ReLU and use an out-of-place residual add. The
# parameters and state_dict keys are identical to torchvision's.

class BasicBlock(_TVBasic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relu = nn.ReLU()
        self.relu_out = nn.ReLU()

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu_out(out + identity)


class Bottleneck(_TVBottleneck):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relu = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu_out = nn.ReLU()

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu_out(out + identity)


def cifar_resnet(depth, num_classes=100):
    block, layers = {18: (BasicBlock, [2, 2, 2, 2]), 50: (Bottleneck, [3, 4, 6, 3])}[depth]
    m = tvm.ResNet(block, layers, num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)  # 3x3, stride 1
    m.relu = nn.ReLU()
    m.maxpool = nn.Identity()
    return m


def cifar_mobilenetv2(num_classes=100):
    m = tvm.mobilenet_v2(num_classes=num_classes)
    m.features[0][0].stride = (1, 1)
    return m


def build_model(arch, num_classes=100):
    if arch == "resnet18":
        return cifar_resnet(18, num_classes)
    if arch == "resnet50":
        return cifar_resnet(50, num_classes)
    if arch == "mobilenetv2":
        return cifar_mobilenetv2(num_classes)
    raise ValueError(f"unknown arch {arch!r}")


def classifier_layer(model):
    """The final linear layer (kept unpruned)."""
    if hasattr(model, "fc"):
        return model.fc
    return model.classifier[-1]
