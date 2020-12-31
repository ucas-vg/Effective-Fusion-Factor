from maskrcnn_benchmark.modeling.rpn.anchor_generator import make_anchor_generator_retinanet
from maskrcnn_benchmark.modeling.rpn.anchor_generator import make_anchor_generator
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.modeling.statistics_alpha.prepare_match_result import make_Match_result_prepare_module
import logging
import torch

class StaAlphaModule(object):
    def __init__(self,cfg):
        self.data_loader = make_data_loader(
            cfg,
            is_train=True,
        )
        self.prepare_fusionfactor = make_Match_result_prepare_module(cfg)
        if cfg.MODEL.RETINANET_ON == True:
            self.anchor_generator = make_anchor_generator_retinanet(cfg)
        else:
            self.anchor_generator = make_anchor_generator(cfg)
        self.backbone = cfg.MODEL.BACKBONE.CONV_BODY
        self.stop_num_iters = cfg.MODEL.FPN.STATISTICS_ALPHA_ITER
        self.default_fusion_factor = cfg.MODEL.FPN.FUSION_FACTORS
        self.device = torch.device(cfg.MODEL.DEVICE)

    def process(self):
        if self.stop_num_iters <= 0:
            return self.default_fusion_factor
        logger = logging.getLogger("maskrcnn_benchmark.alpha")
        logger.info("Start S-alpha ")
        for iteration, (images, targets, _) in enumerate(self.data_loader, 1):
            features_shape = []
            images = images.to(self.device)
            targets = [target.to(self.device) for target in targets]
            if self.backbone == "R-50-FPN" or "R-101-FPN":
                h, w = images.image_sizes[0][0], images.image_sizes[0][1]
                for i in range(2,7):
                    f_shape = torch.rand((len(images.image_sizes),256,int(h/(2**i)),int(w/(2**i)))).to(self.device)
                    features_shape.append(f_shape)
            anchors = self.anchor_generator(images, features_shape)
            anchors = [[ac.to(self.device)  for ac in anchor ]for anchor in anchors]
            sta_fusion_factors = [round(ff,3) for ff in  self.prepare_fusionfactor(anchors, targets)]

            if iteration % 50 == 0:
                logger.info("S-alpha iter:{}/{} ".format(iteration,self.stop_num_iters))
            if iteration > self.stop_num_iters:
                logger.info("Finish S-alpha , fusion_factor:{}".format(sta_fusion_factors))
                return sta_fusion_factors



