import torch
import torch.nn as nn
from network.layers.model_block import FPN
from cfglib.config import config as cfg
import numpy as np
from network.layers.snake import Snake
from network.layers.gcn import GCN
from network.layers.rnn import Rnn
from network.layers.gcn_rnn import GCN_RNN
#from network.layers.transformer import Transformer
#from network.layers.transformer_rnn import Transformer_RNN
import cv2
from util.misc import get_sample_point, fill_hole
from network.layers.gcn_utils import get_node_feature, \
    get_adj_mat, get_adj_ind, coord_embedding, normalize_adj


class Evolution(nn.Module):
    def __init__(self, node_num, adj_num, is_training=True, device=None, model="snake"):
        super(Evolution, self).__init__()
        self.node_num = node_num
        self.adj_num = adj_num
        self.device = device
        self.is_training = is_training
        self.clip_dis = 16

        self.iter = 3
        if model == "gcn":
            self.adj = get_adj_mat(self.adj_num, self.node_num)
            self.adj = normalize_adj(self.adj, type="DAD").float().to(self.device)
            for i in range(self.iter):
                evolve_gcn = GCN(36, 128)
                self.__setattr__('evolve_gcn' + str(i), evolve_gcn)
        elif model == "rnn":
            self.adj = None
            for i in range(self.iter):
                evolve_gcn = Rnn(36, 128)
                self.__setattr__('evolve_gcn' + str(i), evolve_gcn)
        elif model == "gcn_rnn":
            self.adj = get_adj_mat(self.adj_num, self.node_num)
            self.adj = normalize_adj(self.adj, type="DAD").float().to(self.device)
            for i in range(self.iter):
                evolve_gcn = GCN_RNN(36, 128)
                self.__setattr__('evolve_gcn' + str(i), evolve_gcn)
        elif model == "transformer":
            self.adj = None
            for i in range(self.iter):
                evolve_gcn = Transformer(36, 512, num_heads=8,
                                         dim_feedforward=2048, drop_rate=0.0, if_resi=True, block_nums=4)
                self.__setattr__('evolve_gcn' + str(i), evolve_gcn)
        elif model == "transformer_rnn":
            self.adj = None
            for i in range(self.iter):
                evolve_gcn = Transformer_RNN(36, 512, num_heads=8,
                                             dim_feedforward=2048, drop_rate=0.1, if_resi=True, block_nums=4)
                self.__setattr__('evolve_gcn' + str(i), evolve_gcn)
        else:
            self.adj = get_adj_ind(self.adj_num, self.node_num, self.device)
            for i in range(self.iter):
                evolve_gcn = Snake(state_dim=128, feature_dim=36, conv_type='dgrid')
                self.__setattr__('evolve_gcn' + str(i), evolve_gcn)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @staticmethod
    def get_boundary_proposal(input=None, seg_preds=None, switch="gt"):

        if switch == "gt":
            inds = torch.where(input['ignore_tags'] > 0)
            init_polys = input['proposal_points'][inds]
        else:
            tr_masks = input['tr_mask'].cpu().numpy()
            tcl_masks = seg_preds[:, 0, :, :].detach().cpu().numpy() > cfg.threshold
            inds = []
            init_polys = []
            for bid, tcl_mask in enumerate(tcl_masks):
                ret, labels = cv2.connectedComponents(tcl_mask.astype(np.uint8), connectivity=8)
                for idx in range(1, ret):
                    text_mask = labels == idx
                    ist_id = int(np.sum(text_mask*tr_masks[bid])/np.sum(text_mask))-1
                    inds.append([bid, ist_id])
                    poly = get_sample_point(text_mask, cfg.num_points, cfg.approx_factor)
                    init_polys.append(poly)
            inds = torch.from_numpy(np.array(inds)).permute(1, 0).to(input["img"].device)
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device)

        return init_polys, inds

    @staticmethod
    def get_boundary_proposal_eval(input=None, seg_preds=None):

        cls_preds = seg_preds[:, 0, :, :].detach().cpu().numpy()
        dis_preds = seg_preds[:, 1, :, :].detach().cpu().numpy()

        inds = []
        init_polys = []
        for bid, dis_pred in enumerate(dis_preds):
            dis_mask = (dis_pred / np.max(dis_pred)) > cfg.dis_threshold
            dis_mask = fill_hole(dis_mask)
            ret, labels = cv2.connectedComponents(dis_mask.astype(np.uint8), connectivity=8)
            for idx in range(1, ret):
                text_mask = labels == idx
                if np.sum(text_mask) < 150 \
                        or cls_preds[bid][text_mask].mean() < cfg.cls_threshold:
                        # or dis_preds[bid][text_mask].mean() < cfg.dis_th:
                    continue
                inds.append([bid, 0])
                poly = get_sample_point(text_mask, cfg.num_points, cfg.approx_factor)
                init_polys.append(poly)
        if len(inds) > 0:
            inds = torch.from_numpy(np.array(inds)).permute(1, 0).to(input["img"].device)
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device).float()
        else:
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device).float()
            inds = torch.from_numpy(np.array(inds)).to(input["img"].device).float()

        return init_polys, inds

    def evolve_poly(self, snake, cnn_feature, i_it_poly, ind):
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly)
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        node_feats = get_node_feature(cnn_feature, i_it_poly, ind, h, w)
        i_poly = i_it_poly + torch.clamp(snake(node_feats, self.adj).permute(0, 2, 1), -self.clip_dis, self.clip_dis)
        if self.is_training:
            i_poly = torch.clamp(i_poly, 1, w-2)
        else:
            i_poly[:, :, 0] = torch.clamp(i_poly[:, :, 0], 1, w - 2)
            i_poly[:, :, 1] = torch.clamp(i_poly[:, :, 1], 1, h - 2)
        return i_poly

    def forward(self, cnn_feature, input=None, seg_preds=None, switch="gt"):
        # b, h, w = cnn_feature.size(0), cnn_feature.size(2), cnn_feature.size(3)
        # embed_xy = coord_embedding(b, w, h, self.device)
        # embed_feature = torch.cat([cnn_feature, embed_xy], dim=1)
        embed_feature = cnn_feature
        if self.is_training:
            init_polys, inds = self.get_boundary_proposal(input=input, seg_preds=seg_preds, switch=switch)
            # TODO sample fix number
        else:
            init_polys, inds = self.get_boundary_proposal_eval(input=input, seg_preds=seg_preds)
            if init_polys.shape[0] == 0:
                return [init_polys for i in range(self.iter)], init_polys, inds

        py_preds = []
        py_pred = init_polys
        for i in range(self.iter):
            evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
            py_pred = self.evolve_poly(evolve_gcn, embed_feature, py_pred, inds[0])
            py_preds.append(py_pred)

        return py_preds, init_polys, inds


class TextNet(nn.Module):

    def __init__(self, backbone='vgg', is_training=True):
        super().__init__()
        self.is_training = is_training
        self.backbone_name = backbone
        self.fpn = FPN(self.backbone_name, is_training = self.is_training)

        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=2, dilation=2),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=4, dilation=4),
            nn.PReLU(),
            nn.Conv2d(16, 4, kernel_size=1, stride=1, padding=0),
        )
        self.BPN = Evolution(cfg.num_points, adj_num=4,
                             is_training=is_training, device=cfg.device, model="gcn_rnn")

    def load_model(self, model_path):
        print('Loading from {}'.format(model_path))
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict['model'])

    def forward(self, input_dict):
        output = {}
        b, c, h, w = input_dict["img"].shape
        if not self.is_training:
            image = torch.zeros((b, c, cfg.test_size[1], cfg.test_size[1]), dtype=torch.float32).to(cfg.device)
            image[:, :, :h, :w] = input_dict["img"][:, :, :, :]
        else:
            image = input_dict["img"]

        up1, up2, up3, up4, up5 = self.fpn(image)
        up1 = up1[:, :, :h, :w]

        preds = self.seg_head(up1)
        fy_preds = torch.cat([torch.sigmoid(preds[:, 0:2, :, :]), preds[:, 2:4, :, :]], dim=1)
        # fy_preds = torch.sigmoid(preds[:, 0:2, :, :])
        cnn_feats = torch.cat([up1, fy_preds], dim=1)

        py_preds, init_polys, inds = self.BPN(cnn_feats, input=input_dict, seg_preds=fy_preds, switch="gt")
        output["fy_preds"] = fy_preds
        output["py_preds"] = py_preds
        output["init_polys"] = init_polys
        output["inds"] = inds

        return output
