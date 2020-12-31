from maskrcnn_benchmark.modeling.rpn.loss import RPNLossComputation

from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.layers import SigmoidFocalLoss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
import  torch

class PrepareFusionFactorModule(RPNLossComputation):
    def __init__(self,cfg,proposal_matcher, box_coder,
                 generate_labels_func,
                 ):
        self.proposal_matcher = proposal_matcher
        self.box_coder = box_coder
        #self.box_cls_loss_func = sigmoid_focal_loss
        #self.bbox_reg_beta = bbox_reg_beta
        self.copied_fields = ['labels']
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['between_thresholds']
        #self.regress_norm = regress_norm
        self.fuse_factors_generator = Sta_fusion_factors(beta=2)

    def match_targets_to_anchors_forFF(self, anchor, target, copied_fields=[]):
        match_quality_matrix = boxlist_iou(target, anchor)
        # ################# changed by G ###################################################
        if len(target.bbox) == 0:
            matches_gt_forFF = torch.LongTensor([0] * match_quality_matrix.shape[1])
        else:
            _ ,matches_gt_forFF = self.proposal_matcher(match_quality_matrix)
        #####################################################################################
        return matches_gt_forFF

    def prepare_match_result_forFF(self, anchors, targets):
        match_per_batch_forFF = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matches_gt_forFF = self.match_targets_to_anchors_forFF(
                anchors_per_image, targets_per_image, self.copied_fields
            )
            match_per_batch_forFF.append(matches_gt_forFF)

        return match_per_batch_forFF

    def __call__(self, anchors, targets):
        anchors_origin = anchors
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        match_per_batch = self.prepare_match_result_forFF(anchors, targets)

        sta_fusion_factors = self.fuse_factors_generator.process_match_result(match_per_batch, anchors_origin)
        return sta_fusion_factors

class Sta_fusion_factors(object):
    def __init__(self,beta):
        self.beta = beta
        self.total_num_targets = torch.Tensor([0]*5).cuda()

    def cal_ratio(self,tensor):
        tmp = []
        for i in range(len(tensor)-2):
            if i==2:
                tmp.insert(0, (tensor[i + 1] + tensor[i + 2] )/ tensor[i])
            else:
                tmp.insert(0,tensor[i+1]/tensor[i])
        return torch.Tensor(tmp)

    def process_match_result(self,match_per_batch,anchors_origin):
        targets_num_per_batch = []
        for label,anchor in zip(match_per_batch, anchors_origin):
            sta = 0
            targets_num_per_img = []
            for anchor_per_level in anchor:
                num = len(anchor_per_level.bbox)
                N = label[sta:sta + num][label[sta:sta + num] > 0].numel()
                targets_num_per_img.append(N)
                sta += num
            targets_num_per_batch.append(targets_num_per_img)
        targets_num_per_batch = torch.Tensor(targets_num_per_batch).cuda()
        self.total_num_targets += targets_num_per_batch.sum(dim=0)

        targets =  torch.clamp((self.cal_ratio(self.total_num_targets)), self.beta / 10, self.beta)
        return targets.tolist()#[fuse45,fuse34,fuse23]

def generate_retinanet_labels(matched_targets):
    labels_per_image = matched_targets.get_field("labels")
    return labels_per_image

def generate_rpn_labels(matched_targets):
    matched_idxs = matched_targets.get_field("matched_idxs")
    labels_per_image = matched_idxs >= 0
    return labels_per_image

def make_Match_result_prepare_module(cfg):
    if cfg.MODEL.RETINANET_ON == True:
        matcher = Matcher(
            cfg.MODEL.RETINANET.FG_IOU_THRESHOLD,
            cfg.MODEL.RETINANET.BG_IOU_THRESHOLD,
            allow_low_quality_matches=True,
        )
        box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        generate_labels = generate_retinanet_labels
    else:
        matcher = Matcher(
            cfg.MODEL.RPN.FG_IOU_THRESHOLD,
            cfg.MODEL.RPN.BG_IOU_THRESHOLD,
            allow_low_quality_matches=True,
        )
        box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        generate_labels = generate_rpn_labels
    sta_module = PrepareFusionFactorModule(
        cfg,
        matcher,
        box_coder,
        generate_labels,
    )
    return sta_module