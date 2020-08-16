import torch.nn as nn
import torch
from model.retina_config import DefaultConfig
import numpy as np

def coords_fmap2orig(image_shape,stride):
    '''
    transfor one fmap coords to orig coords
    Args
    featurn [batch_size,h,w,c]
    stride int
    Returns
    coords [n,2]
    '''
    h,w= image_shape
    shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
    shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = torch.reshape(shift_x, [-1])
    shift_y = torch.reshape(shift_y, [-1])
    coords = torch.stack([shift_x, shift_y, shift_x, shift_y], -1) + stride // 2
    return coords

class GenAnchors(nn.Module):
    def __init__(self, config = None):
        super().__init__()
        if config is None:
            self.config = DefaultConfig
        else:
            self.config = config

        self.pyramid_levels = self.config.pyramid_levels
        self.ratios = np.array(self.config.ratios)
        self.scales = np.array(self.config.scales)
        self.size = self.config.sizes
        self.strides = self.config.strides

    def forward(self, image):
        H, W = image.size(2), image.size(3) #(ori_H, ori_W)
        feature_size = [(H / stride, W / stride) for stride in self.strides]
        all_anchors = []
        for i in range(len(feature_size)):
            anchors = self.generate_anchors(self.size[i], self.ratios, self.scales)
            shift_anchors = self.shift(anchors, feature_size[i], self.strides[i]) #(H*W, A, 4)
            all_anchors.append(shift_anchors)
        all_anchors = torch.cat(all_anchors, dim = 0)
        return all_anchors

    def generate_anchors(self, base_size=16, ratios=None, scales=None):
        if ratios is None:
            ratios = np.array([0.5, 1, 2])
        if scales is None:
            scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

        num_anchors = len(ratios) * len(scales)  # 9
        anchors = np.zeros((num_anchors, 4))
        anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
        # compute areas of anchors
        areas = anchors[:, 2] * anchors[:, 3]  # (9,)
        # fix the ratios of w, h
        anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))  # (9,)
        anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))  # (9,)

        # transfrom from(0 ,0, w, h ) to ( x1, y1, x2, y2)
        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
        anchors = torch.from_numpy(anchors).float().cuda() if torch.cuda.is_available() else torch.from_numpy(anchors).float()
        return anchors

    def shift(self, anchors, image_shape, stride):
        """
        anchors : Tensor(num, 4)
        image_shape : (H, W)
        return shift_anchor: (H*W*num,4)
        """

        ori_coords = coords_fmap2orig(image_shape, stride)  # (H*W, 4) 4:(x,y,x,y)
        ori_coords = ori_coords.to(device=anchors.device)
        shift_anchor = ori_coords[:, None, :] + anchors[None, :, :]
        return shift_anchor.reshape(-1, 4)


def calc_iou(box1, box2):
    """
    box1:(M,4)
    box2:(N,4)
    """
    lt = torch.max(box1[:,None,:2], box2[:, :2]) #(M,N,2)
    rb = torch.min(box1[:,None,2:], box2[:, 2:]) #(M,N,2)
    wh = torch.clamp(rb - lt , min=0.0) #(M, N, 2)
    inter_area = wh[..., 0] * wh[..., 1] #(M, N)
    area_box1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]) #(M,)
    area_box2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1]) #(N,)

    iou = inter_area / (area_box1[:,None] + area_box2 - inter_area + 1e-16) #(M,N)

    return iou
def focal_loss(preds, targets, alpha=0.25, gamma = 2.0):
    preds = preds.sigmoid()
    preds = torch.clamp(preds, min=1e-4,max = 1. - 1e-4)
    if torch.cuda.is_available():
        alpha_factor = torch.ones(targets.shape).cuda() * alpha
    else:
        alpha_factor = torch.ones(targets.shape) * alpha

    alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, (1.  - alpha_factor))
    focal_weights = torch.where(torch.eq(targets, 1.), 1 - preds, preds)
    focal_weights = alpha_factor * torch.pow(focal_weights, gamma)

    bce = - (targets * torch.log(preds) + (1. - targets) * torch.log(1. - preds))
    cls_loss = focal_weights * bce

    if torch.cuda.is_available():
        cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros_like(cls_loss).cuda())
    else:
        cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros_like(cls_loss))

    return cls_loss.sum()


def smooth_l1(pos_inds,anchor_infos, boxes,reg_pred):
    """
    pos_inds : (num_pos,)
    boxes:(sum(H*W)*A, 4)
    reg_pred: (sum(H*W)*A, 4)
    """
    anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y = anchor_infos #(sum(H*W)*A,)
    if pos_inds.sum() > 0:

        pos_reg_pred = reg_pred[pos_inds,:] #(num_pos, 4)

        gt_widths = boxes[pos_inds][:, 2] - boxes[pos_inds][:, 0]
        gt_heights = boxes[pos_inds][:, 3] - boxes[pos_inds][:, 1]
        gt_ctr_x = boxes[pos_inds][:, 0] + gt_widths * 0.5
        gt_ctr_y = boxes[pos_inds][:, 1] + gt_heights * 0.5

        pos_anchor_widths = anchor_widths[pos_inds]
        pos_anchor_heights = anchor_heights[pos_inds]
        pos_anchor_ctr_x = anchor_ctr_x[pos_inds]
        pos_anchor_ctr_y = anchor_ctr_y[pos_inds]

        gt_widths = torch.clamp(gt_widths, min=1.0)
        gt_heights = torch.clamp(gt_heights, min=1.0)

        target_dx = (gt_ctr_x - pos_anchor_ctr_x) / pos_anchor_widths
        target_dy = (gt_ctr_y - pos_anchor_ctr_y) / pos_anchor_heights
        target_dw = torch.log(gt_widths / pos_anchor_widths)
        target_dh = torch.log(gt_heights / pos_anchor_heights)

        targets = torch.stack([target_dx,target_dy,target_dw,target_dh], dim=0).t() #(num_pos,4)
        if torch.cuda.is_available():
            targets = targets / torch.FloatTensor([0.1,0.1,0.2,0.2]).cuda()
        else:
            targets = targets / torch.FloatTensor([0.1,0.1,0.2,0.2])


        reg_diff = torch.abs(targets - pos_reg_pred) #(num_pos,4)
        reg_loss = torch.where(
            torch.le(reg_diff, 1.0/9.0),
            0.5 * 9.0 * torch.pow(reg_diff, 2),
            reg_diff - 0.5 /9.0
        )
        return reg_loss.mean()
    else:
        if torch.cuda.is_available():
            reg_loss = torch.tensor(0).float().cuda()
        else:
            reg_loss = torch.tensor(0).float()

        return reg_loss

def giou(pos_inds,anchor_infos, boxes,reg_pred):
    """
    pos_inds : (num_pos,)
    boxes:(sum(H*W)*A, 4)
    reg_pred: (sum(H*W)*A, 4)
    """
    anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y = anchor_infos #(sum(H*W)*A,)
    if pos_inds.sum() > 0:

        pos_reg_pred = reg_pred[pos_inds,:] #(num_pos, 4)

        gt_boxes = boxes[pos_inds,:] #(num_pos, 4)

        pos_anchor_widths = anchor_widths[pos_inds] #(num_pos,)
        pos_anchor_heights = anchor_heights[pos_inds] #(num_pos,)
        pos_anchor_ctr_x = anchor_ctr_x[pos_inds] #(num_pos,)
        pos_anchor_ctr_y = anchor_ctr_y[pos_inds] #(num_pos,)


        dx = pos_reg_pred[:, 0] * 0.1 #(num_pos,)
        dy = pos_reg_pred[:, 1] * 0.1 #(num_pos,)
        dw = pos_reg_pred[:, 2] * 0.2 #(num_pos,)
        dh = pos_reg_pred[:, 3] * 0.2 #(num_pos,)

        pred_ctr_x = dx * pos_anchor_widths + pos_anchor_ctr_x #(num_pos,)
        pred_ctr_y = dy * pos_anchor_heights + pos_anchor_ctr_y #(num_pos,)
        pred_w = torch.exp(dw) * pos_anchor_widths #(num_pos,)
        pred_h = torch.exp(dh) * pos_anchor_heights #(num_pos,)

        pred_x1 = pred_ctr_x - pred_w * 0.5 #(num_pos,)
        pred_y1 = pred_ctr_y - pred_h * 0.5 #(num_pos,)
        pred_x2 =  pred_ctr_x + pred_w * 0.5 #(num_pos,)
        pred_y2 = pred_ctr_y + pred_h * 0.5 #(num_pos,)

        preds_boxes = torch.stack([pred_x1,pred_y1,pred_x2,pred_y2], dim=0).t() #(num_pos,4)
        reg_loss = compute_giou_loss(gt_boxes, preds_boxes)
    else:
        if torch.cuda.is_available():
            reg_loss = torch.tensor(0).float().cuda()
        else:
            reg_loss = torch.tensor(0).float()

    return reg_loss

def compute_giou_loss(boxes1, boxes2):
    """
    boxes1 :(N,4)  (x1,y1,x2,y2)
    boxes2: (N,4)  (x1,y1,x2,y2)
    """
    x1y1 = torch.max(boxes1[:, :2], boxes2[:, :2])
    x2y2 = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = torch.clamp(x2y2 - x1y1, min=0.)
    area_inter = wh[:, 0] * wh[:, 1]
    area_b1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area_b2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union =  area_b1 + area_b2 - area_inter
    iou = area_inter / (union + 1e-16)

    x1y1_max = torch.min(boxes1[:, :2], boxes2[:, :2])
    x2y2_max = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    g_wh = torch.clamp(x2y2_max - x1y1_max, min=0.)
    g_area = g_wh[:, 0] * g_wh[:, 1]

    giou = iou - (g_area - union) / g_area.clamp(1e-10)
    loss = 1. - giou
    return loss.mean()

class LOSS(nn.Module):
    def __init__(self,reg_mode = 'giou'):
        super(LOSS, self).__init__()
        self.reg_mode = reg_mode

    def forward(self, inputs):
        """
        cls_logits :(n, sum(H*W)*A, class_num+1)
        reg_preds:(n, sum(H*W)*A, 4)
        anchors:(sum(H*W)*A, 4)
        boxes:(n, max_num, 4)
        classes:(n, max_num)
        """
        cls_logits, reg_preds, anchors, boxes, classes = inputs
        anchor_widths = anchors[:, 2] - anchors[:, 0]
        anchor_heights = anchors[:, 3] - anchors[:, 1]
        anchor_ctr_x = anchors[:, 0] + anchor_widths * 0.5
        anchor_ctr_y = anchors[:, 1] + anchor_heights * 0.5

        bacth_size = cls_logits.shape[0]
        class_loss = []
        reg_loss = []
        for i in range(bacth_size):
            per_cls_logit = cls_logits[i,:,:] #(sum(H*W)*A, class_num)
            per_reg_pred = reg_preds[i,:,:]
            per_boxes = boxes[i,:,:]
            per_classes = classes[i,:]
            mask = per_boxes[:, 0] != -1
            per_boxes = per_boxes[mask] #(?, 4)
            per_classes = per_classes[mask] #(?,)
            if per_classes.shape[0] == 0:
                alpha_factor = torch.ones(per_cls_logit.shape).cuda() * 0.25 if torch.cuda.is_available()  else torch.ones(per_cls_logit.shape) * 0.25
                alpha_factor = 1. - alpha_factor
                focal_weights = per_cls_logit
                focal_weights = alpha_factor * torch.pow(focal_weights, 2.0)
                bce = -(torch.log(1.0 - per_cls_logit))
                cls_loss = focal_weights * bce
                class_loss.append(cls_loss.sum())
                reg_loss.append(torch.tensor(0).float())
                continue
            IoU =  calc_iou(anchors, per_boxes) #(sum(H*W)*A, ?)

            iou_max, max_ind = torch.max(IoU, dim=1) #(sum(H*W)*A,)
            
            
            targets = torch.ones_like(per_cls_logit) * -1 #(sum(H*W)*A, class_num)
            
            
            targets[iou_max < 0.4, :] = 0 #bg

            pos_anchors_ind = iou_max >= 0.5 #(?,)
            num_pos =  torch.clamp(pos_anchors_ind.sum().float(), min=1.0)

            assigned_classes = per_classes[max_ind] #(sum(H*W)*A, )
            assigned_boxes = per_boxes[max_ind,:] #(sum(H*W)*A, 4)

            targets[pos_anchors_ind,:] = 0
            targets[pos_anchors_ind, (assigned_classes[pos_anchors_ind]).long() - 1] = 1

            class_loss.append(focal_loss(per_cls_logit, targets).view(1) / num_pos)
            if self.reg_mode == 'smoothl1':
                reg_loss.append(smooth_l1(pos_anchors_ind, [anchor_widths,anchor_heights,anchor_ctr_x,anchor_ctr_y],
                                 assigned_boxes,per_reg_pred))
            elif self.reg_mode =='giou':
                reg_loss.append(giou(pos_anchors_ind, [anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y],
                                          assigned_boxes, per_reg_pred))

        cls_loss = torch.stack(class_loss).mean()
        reg_loss = torch.stack(reg_loss).mean()
        total_loss = cls_loss + reg_loss
        return cls_loss, reg_loss, total_loss


if __name__ =="__main__":
    """
       cls_logits :(n, sum(H*W)*A, class_num+1)
       reg_preds:(n, sum(H*W)*A, 4)
       anchors:(sum(H*W)*A, 4)
       boxes:(n, max_num, 4)
       classes:(n, max_num)
    """
    image = torch.rand((1,3,512,384))
    anchor_model = GenAnchors()
    anchors = anchor_model(image)
    boxes = [[69,172,270,330],[150,141,229,284],[258,198,297,329]]
    classes = [12,1,1]
    boxes = torch.FloatTensor(boxes) #(3,4)
    boxes = torch.nn.functional.pad(boxes,[0, 0, 0, 47],value=-1).unsqueeze(dim=0)
    classes = torch.FloatTensor(classes) #(3,)
    classes = torch.nn.functional.pad(classes,[0,47],value=-1).unsqueeze(dim=0)
    annotation = torch.cat([boxes,classes.unsqueeze(dim=2)], dim=2)
    #print(annotation)
    # print(anchors.dtype)
    # print(boxes.dtype)
    cls_logits = torch.ones((1,36828,20)) * 0.5
    reg_preds = torch.ones((1,36828,4))
    loss = LOSS()
    print(loss([cls_logits,reg_preds,anchors,boxes,classes]))















