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
        use_bias = True
        use_dropout = False
        lrelu = nn.LeakyReLU(0.1, True)

        conv3_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=use_bias)
        conv3_2 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.conv3 = nn.Sequential(conv3_1, nn.BatchNorm2d(512), lrelu,
                                    conv3_2, nn.BatchNorm2d(1024, affine=False), lrelu)

        conv4_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=use_bias)
        conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=use_bias)
        self.conv4 = nn.Sequential(conv4_1, nn.BatchNorm2d(512), lrelu,
                                    conv4_2, nn.BatchNorm2d(512, affine=False), lrelu)

        conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=use_bias)
        conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=use_bias)
        self.conv5 = nn.Sequential(conv5_1, nn.BatchNorm2d(512), lrelu,
                                    conv5_2, nn.BatchNorm2d(512, affine=False), lrelu)
        # 16x
        e1_conv = nn.Conv2d(521, 512, kernel_size=3, padding=1, bias=use_bias)
        self.e1 = nn.Sequential(e1_conv, nn.BatchNorm2d(512), lrelu)
        
        # 32x
        e2_conv = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.e2 = nn.Sequential(e2_conv, nn.BatchNorm2d(1024), lrelu)
        
        # 64x
        e3_conv = nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1, bias=use_bias)
        self.e3 = nn.Sequential(e3_conv, nn.BatchNorm2d(2048), lrelu)

        self.d1_deconv = nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1, bias=use_bias)
        if use_dropout:
            self.d1 = nn.Sequential(nn.BatchNorm2d(1024), nn.Dropout(0.5), lrelu)
        else:
            self.d1 = nn.Sequential(nn.BatchNorm2d(1024), lrelu)
        
        self.d2_deconv = nn.ConvTranspose2d(2048, 512, kernel_size=4, stride=2, padding=1, bias=use_bias)
        if use_dropout:
            self.d2 = nn.Sequential(nn.BatchNorm2d(512), nn.Dropout(0.5), lrelu)
        else:
            self.d2 = nn.Sequential(nn.BatchNorm2d(512), lrelu)
        
        d3_conv = nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=use_bias)
        if use_dropout:
            self.d3 = nn.Sequential(d3_conv, nn.BatchNorm2d(512), nn.Dropout(0.5), lrelu)
        else:
            self.d3 = nn.Sequential(d3_conv, nn.BatchNorm2d(512), lrelu)

        d4_conv = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=use_bias)
        self.d4 = nn.Sequential(d4_conv, nn.ReLU(True))
        #######################

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        # base_feat = self.RCNN_base(im_data)
        base_feat_conv3 = self.RCNN_base(im_data)
        base_feat_conv4 = self.RCNN_conv4(im_data)
        base_feat_conv5 = self.RCNN_conv5(im_data)
        
        ############
        x_o3 = self.conv3(base_feat_conv3)
        x_o4 = self.conv4(base_feat_conv4)
        x_o5 = self.conv5(base_feat_conv5)
        x_o = x_o3 + x_o5 + x_o5

        x_e1 = self.e1(x_o)
        x_e2 = self.e2(x_e1)
        x = self.e3(x_e2)
        x = self.d1_deconv(x, output_size=x_e2.size())
        x = self.d1(x)
        x = self.d2_deconv(torch.cat([x_e2, x], 1), output_size=base_feat_conv5.size())
        x = self.d3(torch.cat([x_e1, x], 1))
        base_feat = self.d4(x)
        #############

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
