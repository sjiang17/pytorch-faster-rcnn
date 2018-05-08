import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()
        
        
        ################
        in_channels = 256
        use_bias = True
        use_dropout = False
        e1_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=use_bias)
        e1_norm = nn.BatchNorm2d(in_channels)
        e1_relu = nn.LeakyReLU(0.2, True)
        self.e1 = nn.Sequential(e1_conv, e1_norm, e1_relu)
        
        e2_conv = nn.Conv2d(in_channels, 512, kernel_size=4, stride=2, padding=1, bias=use_bias)
        e2_norm = nn.BatchNorm2d(512) # inner_nc
        e2_relu = nn.LeakyReLU(0.2, True)
        self.e2 = nn.Sequential(e2_conv, e2_norm, e2_relu)
        
        e3_conv = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias)
        e3_norm = nn.BatchNorm2d(1024)
        e3_relu = nn.LeakyReLU(0.2, True)
        self.e3 = nn.Sequential(e3_conv, e3_norm, e3_relu)
        
        e4_conv = nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1, bias=use_bias)
        e4_norm = nn.BatchNorm2d(2048)
        e4_relu = nn.LeakyReLU(0.2, True)
        self.e4 = nn.Sequential(e4_conv, e4_norm, e4_relu)

        e5_conv = nn.Conv2d(2048, 4096, kernel_size=4, stride=2, padding=1, bias=use_bias)
        e5_norm = nn.BatchNorm2d(4096)
        e5_relu = nn.LeakyReLU(0.2, True)
        self.e5 = nn.Sequential(e5_conv, e5_norm, e5_relu)

        self.d1_deconv = nn.ConvTranspose2d(4096, 2048, kernel_size=4, stride=2, padding=1, bias=use_bias)
        d1_norm = nn.BatchNorm2d(2048)
        d1_relu = nn.LeakyReLU(True)
        if use_dropout:
            self.d1 = nn.Sequential(d1_norm, nn.Dropout(0.5), d1_relu)
        else:
            self.d1 = nn.Sequential(d1_norm, d1_relu)
        
        self.d2_deconv = nn.ConvTranspose2d(4096, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias)
        d2_norm = nn.BatchNorm2d(1024)
        d2_relu = nn.LeakyReLU(True)
        if use_dropout:
            self.d2 = nn.Sequential(d2_norm, nn.Dropout(0.5), d2_relu)
        else:
            self.d2 = nn.Sequential(d2_norm, d2_relu)
        
        self.d3_deconv = nn.ConvTranspose2d(2048, 512, kernel_size=4, stride=2, padding=1, bias=use_bias)
        d3_norm = nn.BatchNorm2d(512)
        d3_relu = nn.LeakyReLU(True)
        if use_dropout:
            self.d3 = nn.Sequential(d3_norm, nn.Dropout(0.5), d3_relu)
        else:
            self.d3 = nn.Sequential(d3_norm, d3_relu)
        
        self.d4_deconv = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=use_bias)
        d4_norm = nn.BatchNorm2d(256)
        d4_relu = nn.LeakyReLU(True)
        if use_dropout:
            self.d4 = nn.Sequential(d4_norm, nn.Dropout(0.5), d4_relu)
        else:
            self.d4 = nn.Sequential(d4_norm, d4_relu)
        
        d5_conv = nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=use_bias)
        d5_relu = nn.ReLU(True)
        self.d5 = nn.Sequential(d5_conv, d5_relu)
        
        d6_conv = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=use_bias)
        d6_relu = nn.ReLU(True)
        self.d6 = nn.Sequential(d6_conv, d6_relu)
        #######################

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        # base_feat = self.RCNN_base(im_data)
        base_feat_bottom = self.RCNN_base(im_data)
        
        ############
        x_input = base_feat_bottom
        x_e1 = self.e1(x_input)
        x_e2 = self.e2(x_e1)
        x_e3 = self.e3(x_e2)
        x_e4 = self.e4(x_e3)
        x = self.e5(x_e4)
        x = self.d1_deconv(x, output_size=x_e4.size())
        x = self.d1(x)
        x = self.d2_deconv(torch.cat([x_e4, x], 1), output_size=x_e3.size())
        x = self.d2(x)
        x = self.d3_deconv(torch.cat([x_e3, x], 1), output_size=x_e2.size())
        x = self.d3(x)
        x = self.d4_deconv(torch.cat([x_e2, x], 1), output_size=x_input.size())
        x = self.d4(x)
        x = self.d5(torch.cat([x_input, x], 1))
        x = self.d6(x)
        #############

        base_feat = self.RCNN_base2(x)
        

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()