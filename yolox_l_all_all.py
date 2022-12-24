#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp
import torch.nn as nn

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1
        self.width = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.multiscale_range = 0

        # Define yourself dataset path
        self.data_dir = "datasets/all"
        self.train_ann = "instances_train.json"
        self.val_ann = "instances_val.json"
        self.test_ann = "instances_test.json"

        self.num_classes = 56

        self.max_epoch = 200
        self.data_num_workers = 8
        self.eval_interval = 10
        self.no_aug_epochs = 25

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead, YOLOXHead_4_all, YOLOPAFPN_4_all
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [128, 256, 512, 1024]
            backbone = YOLOPAFPN_4_all(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = YOLOXHead_4_all(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model
