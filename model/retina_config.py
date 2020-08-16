class DefaultConfig():
    # backbone
    pretrained = True
    freeze_stage_1 = True
    freeze_bn = True

    # fpn
    fpn_out_channels = 256
    use_p5 = True

    # head
    class_num = 80
    use_GN_head = True
    prior = 0.01
    anchor_nums = 9
    # training
    strides = [8, 16, 32, 64, 128]
    pyramid_levels = [3, 4, 5, 6, 7]
    sizes = [32, 64, 128, 256, 512]
    ratios =[0.5, 1, 2]
    scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    # inference
    score_threshold = 0.05
    nms_iou_threshold = 0.6
    max_detection_boxes_num = 2000