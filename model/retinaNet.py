import torch
import torch.nn as nn
import numpy as np

from model.retina_config import DefaultConfig
from model.backbone.resnet import resnet50
from model.retina_head import RetinaHead
from model.fpn_neck import FPN
from model.retina_loss import LOSS,GenAnchors

class RetinaNet(nn.Module):
    def __init__(self, config = None):
        super(RetinaNet, self).__init__()
        if config is None:
            self.config = DefaultConfig
        else:
            self.config  = config
        self.backbone = resnet50(pretrained=self.config.pretrained)
        self.fpn = FPN(features=self.config.fpn_out_channels,
                       use_p5=self.config.use_p5)
        self.head = RetinaHead(config=self.config)

    def train(self, mode=True):
        super(RetinaNet, self).train(mode=True)

        def freeze_bn(module):
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            class_name = module.__class__.__name__
            if class_name.find('BatchNorm') != -1:
                for p in module.parameters():
                    p.requires_grad = False
        if self.config.freeze_bn:
            self.apply(freeze_bn)
            print("info====>success freeze bn")
        if self.config.freeze_stage_1:
            self.backbone.freeze_stages(1)
            print("info=====> success freeze stage 1")


    def forward(self, images):
        C3,C4,C5 = self.backbone(images)
        all_p_level = self.fpn([C3,C4,C5])
        cls_logits, reg_preds = self.head(all_p_level)

        return cls_logits, reg_preds

class DetectionHead(nn.Module):
    def __init__(self, score_thres, nms_iou_thres, max_detections, config = None):
        super(DetectionHead, self).__init__()
        self.score_thres = score_thres
        self.nms_iou_thres = nms_iou_thres
        self.max_detections = max_detections
        if config is None:
            self.config = DefaultConfig
        else:
            self.config = config

    def forward(self, inputs):
        """
        inputs: cls_logits, reg_preds, anchors
        (N, sum(H*W)*A, class_num)
        (N, sum(H*W)*A, class_num)
        (sum(H*W)*A, 4)
        """
        cls_logits, reg_preds, anchors = inputs
        batch_size = cls_logits.shape[0]
        cls_logits = cls_logits.sigmoid_()

        boxes = self.ped2ori_box(reg_preds, anchors)   #(N, sum(H*W)*A, 4)

        cls_scores, cls_ind = torch.max(cls_logits, dim=2) #(N, sum(H*W)*A)
        cls_ind = cls_ind + 1
        #select topK
        max_det = min(self.max_detections, cls_logits.shape[1]) #topK
        topk_ind = torch.topk(cls_scores, max_det, dim=-1, largest=True,sorted=True)[1] #(N, topK)
        cls_topk =[]
        idx_topk = []
        box_topk = []
        for i in range(batch_size):
            cls_topk.append(cls_scores[i][topk_ind[i]]) #(topK,)
            idx_topk.append(cls_ind[i][topk_ind[i]]) #(topK,)
            box_topk.append(boxes[i][topk_ind[i]]) #(topK,4)
        cls_topk = torch.stack(cls_topk, dim=0) #(N,topK)
        idx_topk = torch.stack(idx_topk, dim=0) #(N,topK)
        box_topk = torch.stack(box_topk, dim=0) #(N,topK,4)

        return self._post_process(cls_topk, idx_topk, box_topk)



    def _post_process(self, topk_scores, topk_inds, topk_boxes):
        """
        topk_scores:(N,topk)
        """
        batch_size = topk_scores.shape[0]
        #mask = topk_scores >= self.score_thres #(N,topK)
        _cls_scores =[]
        _cls_idxs = []
        _reg_preds = []
        for i in range(batch_size):
            mask = topk_scores[i] >= self.score_thres
            per_cls_scores = topk_scores[i][mask] #(?,)
            per_cls_idxs = topk_inds[i][mask] #(?,)
            per_boxes = topk_boxes[i][mask] #(?,4)
            nms_ind = self._batch_nms(per_cls_scores, per_cls_idxs, per_boxes)
            _cls_scores.append(per_cls_scores[nms_ind])
            _cls_idxs.append(per_cls_idxs[nms_ind])
            _reg_preds.append(per_boxes[nms_ind])

        return torch.stack(_cls_scores, dim=0), torch.stack(_cls_idxs,dim=0), torch.stack(_reg_preds,dim=0)

    def _batch_nms(self, scores, idxs, boxes):
        """
        scores:(?,)
        """
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        max_coords = boxes.max()

        offsets = idxs.to(boxes) * (max_coords + 1) #(?,)
        post_boxes = boxes + offsets[:,None] #(?,4)

        keep = self.box_nms(scores, post_boxes, self.nms_iou_thres)
        return keep

    def box_nms(self, scores, boxes,iou_thres):
        if boxes.shape[0] == 0:
            return torch.zeros(0,device=boxes.device).long()
        x1,y1,x2,y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:,3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = torch.sort(scores, descending=True)[1] #(?,)
        keep =[]
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
            else:
                i = order[0].item()
                keep.append(i)

                xmin = torch.clamp(x1[order[1:]], min = float(x1[i]))
                ymin = torch.clamp(y1[order[1:]], min = float(y1[i]))
                xmax = torch.clamp(x2[order[1:]], max = float(x2[i]))
                ymax = torch.clamp(y2[order[1:]], max = float(y2[i]))

                inter_area = torch.clamp(xmax - xmin, min=0.0) * torch.clamp(ymax - ymin, min=0.0)

                iou = inter_area / (areas[i] + areas[order[1:]] - inter_area + 1e-16)

                mask_ind = (iou <= iou_thres).nonzero().squeeze()

                if mask_ind.numel() == 0:
                    break
                order = order[mask_ind + 1]
        return torch.LongTensor(keep)

    def ped2ori_box(self, reg_preds, anchors):
        """
        reg_preds: (N, sum(H*W)*A, 4)
        anchors:(sum(H*W)*A, 4)
        return (N, sum(H*W)*A, 4) 4:(x1,y1,x2,y2)
        """
        if torch.cuda.is_available():
            mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
            std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
        else:
            mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))
            std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        dx,dy,dw,dh = reg_preds[..., 0], reg_preds[..., 1],reg_preds[...,2],reg_preds[...,3] #(N,sum(H*W)*A)
        dx = dx * std[0] + mean[0]
        dy = dy * std[1] + mean[1]
        dw = dw * std[2] + mean[2]
        dh = dh * std[3] + mean[3]

        anchor_w = (anchors[:, 2] - anchors[:, 0]).unsqueeze(0) #(1,sum(H*W)*A)
        anchor_h = (anchors[:, 3] - anchors[:, 1]).unsqueeze(0) #(1,sum(H*W)*A)
        anchor_ctr_x = anchors[:, 0].unsqueeze(0) + anchor_w * 0.5 #(1,sum(H*W)*A)
        anchor_ctr_y = anchors[:, 1].unsqueeze(0) + anchor_h * 0.5 #(1,sum(H*W)*A)

        pred_ctr_x = dx * anchor_w + anchor_ctr_x  #(N,sum(H*W)*A)
        pred_ctr_y = dy * anchor_h + anchor_ctr_y #(N,sum(H*W)*A)
        pred_w =  torch.exp(dw) * anchor_w #(N,sum(H*W)*A)
        pred_h = torch.exp(dh) * anchor_h #(N,sum(H*W)*A)

        x1 = pred_ctr_x - pred_w * 0.5 #(N,sum(H*W)*A)
        y1 = pred_ctr_y - pred_h * 0.5 #(N,sum(H*W)*A)
        x2 = pred_ctr_x + pred_w * 0.5 #(N,sum(H*W)*A)
        y2 = pred_ctr_y + pred_h * 0.5 #(N,sum(H*W)*A)
        return torch.stack([x1,y1,x2,y2], dim=2) #(N,sum(H*W)*A,4)

class ClipBoxes(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,batch_imgs,batch_boxes):
        batch_boxes=batch_boxes.clamp_(min=0)
        h,w=batch_imgs.shape[2:]
        batch_boxes[...,[0,2]]=batch_boxes[...,[0,2]].clamp_(max=w-1)
        batch_boxes[...,[1,3]]=batch_boxes[...,[1,3]].clamp_(max=h-1)
        return batch_boxes

class RetinaNetDetector(nn.Module):

    def __init__(self, config = None, mode='training'):
        super(RetinaNetDetector, self).__init__()
        if config is None:
            self.config = DefaultConfig
        else:
            self.config = config
        self.mode = mode
        self.body = RetinaNet(config=config)
        self.get_anchors = GenAnchors()
        if mode == 'training':
            self.loss_func = LOSS()
        else:
            self.detector = DetectionHead(self.config.score_threshold, self.config.nms_iou_threshold,
                                          self.config.max_detection_boxes_num)
            self.clip_boxes = ClipBoxes()

    def forward(self, inputs):
        if self.mode == 'training':
            batch_images, boxes, classes = inputs
            anchors = self.get_anchors(batch_images)
            cls_logits, reg_preds = self.body(batch_images)
            loss = self.loss_func([cls_logits, reg_preds, anchors, boxes, classes])
            return loss
        elif self.mode =='inference':
            batch_images = inputs
            anchors = self.get_anchors(batch_images)
            cls_logits, reg_preds = self.body(batch_images)
            cls_scores, cls_idxs, boxes = self.detector([cls_logits, reg_preds, anchors])
            boxes = self.clip_boxes(batch_images,boxes)
            return cls_scores, cls_idxs, boxes

if __name__ == "__main__":
    model = RetinaNetDetector(config=DefaultConfig)
    #classification, regression, anchors, annotations
    image = torch.ones((1, 3, 512, 384))
    boxes = [[69, 172, 270, 330], [150, 141, 229, 284], [258, 198, 297, 329]]
    classes = [12, 1, 1]
    boxes = torch.FloatTensor(boxes)  # (3,4)
    boxes = torch.nn.functional.pad(boxes, [0, 0, 0, 47], value=-1).unsqueeze(dim=0)
    classes = torch.FloatTensor(classes)  # (3,)
    classes = torch.nn.functional.pad(classes, [0, 47], value=-1).unsqueeze(dim=0)
    annotation = torch.cat([boxes, classes.unsqueeze(dim=2)], dim=2)
    model.train()
    print(model([image,boxes,classes]))