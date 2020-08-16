import torch.nn as nn
import torch
from  model.retina_config import DefaultConfig
import math

class RetinaHead(nn.Module):
    def __init__(self, config = None):
        super(RetinaHead, self).__init__()
        if config is None:
            self.config = DefaultConfig
        else:
            self.config = config
        self.anchor_nums = self.config.anchor_nums
        cls_branch = []
        reg_branch = []
        for i in range(4):
            cls_branch.append(nn.Conv2d(self.config.fpn_out_channels, self.config.fpn_out_channels,
                                        kernel_size=3, stride=1,padding=1,bias=True))
            if self.config.use_GN_head:
                cls_branch.append(nn.GroupNorm(32,self.config.fpn_out_channels))
            cls_branch.append(nn.ReLU(inplace=True))

            reg_branch.append(nn.Conv2d(self.config.fpn_out_channels, self.config.fpn_out_channels,
                                        kernel_size=3, stride=1, padding=1, bias=True))
            if self.config.use_GN_head:
                reg_branch.append(nn.GroupNorm(32, self.config.fpn_out_channels))
            reg_branch.append(nn.ReLU(inplace=True))

        self.cls_conv = nn.Sequential(*cls_branch)
        self.reg_conv = nn.Sequential(*reg_branch)
        self.cls_out = nn.Conv2d(self.config.fpn_out_channels, self.config.class_num * self.anchor_nums, kernel_size=3, stride=1,
                                 padding=1, bias=True)
        self.reg_out = nn.Conv2d(self.config.fpn_out_channels, self.anchor_nums * 4, kernel_size= 3, stride=1,padding=1,bias=True)
        self.prior = self.config.prior
        self.apply(self.init_conv_RandomNormal)
        nn.init.constant_(self.cls_out.bias, -math.log((1 - self.prior) / self.prior))

    def init_conv_RandomNormal(self,module,std=0.01):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=std)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, inputs):
        """
        inputs:fpn output[P3,P4,P5,P6,P7]
        """

        cls_out = []
        reg_out = []
        for pred in inputs:
            batch_size, channel, H, W = pred.shape
            cls_convput = self.cls_conv(pred)
            cls_output = self.cls_out(cls_convput)
            cls_output = cls_output.permute(0,2,3,1).contiguous().view(batch_size, H * W * self.anchor_nums, -1)
            cls_out.append(cls_output)

            reg_output = self.reg_conv(pred)
            reg_output = self.reg_out(reg_output)
            reg_output = reg_output.permute(0, 2, 3, 1).contiguous().view(batch_size, H * W * self.anchor_nums, -1)
            reg_out.append(reg_output)

        cls_logits = torch.cat(cls_out, dim = 1)
        reg_preds = torch.cat(reg_out, dim = 1)
        return cls_logits, reg_preds

