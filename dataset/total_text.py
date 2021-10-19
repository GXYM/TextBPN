import warnings
warnings.filterwarnings("ignore")
import os
import re
import numpy as np
import scipy.io as io
from util import strs
from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance
import cv2
from util import io as libio


class TotalText(TextDataset):

    def __init__(self, data_root, ignore_list=None, is_training=True, transform=None):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.is_training = is_training

        if ignore_list:
            with open(ignore_list) as f:
                ignore_list = f.readlines()
                ignore_list = [line.strip() for line in ignore_list]
        else:
            ignore_list = []

        self.image_root = os.path.join(data_root, 'Images', 'Train' if is_training else 'Test')
        self.annotation_root = os.path.join(data_root, 'gt', 'Train' if is_training else 'Test')
        self.image_list = os.listdir(self.image_root)
        self.image_list = list(filter(lambda img: img.replace('.jpg', '') not in ignore_list, self.image_list))
        self.annotation_list = ['poly_gt_{}'.format(img_name.replace('.jpg', '')) for img_name in self.image_list]

    @staticmethod
    def parse_mat(mat_path):
        """
        .mat file parser
        :param mat_path: (str), mat file path
        :return: (list), TextInstance
        """
        annot = io.loadmat(mat_path + ".mat")
        polygons = []
        for cell in annot['polygt']:
            x = cell[1][0]
            y = cell[3][0]
            text = cell[4][0] if len(cell[4]) > 0 else '#'
            ori = cell[5][0] if len(cell[5]) > 0 else 'c'

            if len(x) < 4:  # too few points
                continue
            pts = np.stack([x, y]).T.astype(np.int32)
            polygons.append(TextInstance(pts, ori, text))

        return polygons

    @staticmethod
    def parse_carve_txt(gt_path):
        """
        .mat file parser
        :param gt_path: (str), mat file path
        :return: (list), TextInstance
        """
        lines = libio.read_lines(gt_path + ".txt")
        polygons = []
        for line in lines:
            line = strs.remove_all(line, '\xef\xbb\xbf')
            gt = line.split(',')
            xx = gt[0].replace("x: ", "").replace("[[", "").replace("]]", "").lstrip().rstrip()
            yy = gt[1].replace("y: ", "").replace("[[", "").replace("]]", "").lstrip().rstrip()
            try:
                xx = [int(x) for x in re.split(r" *", xx)]
                yy = [int(y) for y in re.split(r" *", yy)]
            except:
                xx = [int(x) for x in re.split(r" +", xx)]
                yy = [int(y) for y in re.split(r" +", yy)]
            if len(xx) < 4 or len(yy) < 4:  # too few points
                continue
            text = gt[-1].split('\'')[1]
            try:
                ori = gt[-2].split('\'')[1]
            except:
                ori = 'c'
            pts = np.stack([xx, yy]).T.astype(np.int32)
            polygons.append(TextInstance(pts, ori, text))
        # print(polygon)
        return polygons

    def __getitem__(self, item):

        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)

        # Read image data
        image = pil_load_img(image_path)

        # Read annotation
        annotation_id = self.annotation_list[item]
        annotation_path = os.path.join(self.annotation_root, annotation_id)
        polygons = self.parse_mat(annotation_path)

        if self.is_training:
            return self.get_training_data(image, polygons, image_id=image_id, image_path=image_path)
        else:
            return self.get_test_data(image, polygons, image_id=image_id, image_path=image_path)

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    import time
    from util.augmentation import Augmentation
    from util import canvas as cav

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(
        size=640, mean=means, std=stds
    )

    # transform = BaseTransformNresize(
    #     mean=means, std=stds
    # )

    trainset = TotalText(
        data_root='../data/total-text-mat',
        ignore_list=None,
        is_training=True,
        transform=transform
    )
    for idx in range(0, len(trainset)):
        t0 = time.time()
        img, train_mask, tr_mask, distance_field, \
        direction_field, weight_matrix, ctrl_points, proposal_points, ignore_tags = trainset[idx]
        img, train_mask, tr_mask, distance_field, \
        direction_field, weight_matrix, ctrl_points, proposal_points, ignore_tags\
            = map(lambda x: x.cpu().numpy(),
                  (img, train_mask, tr_mask, distance_field,
                   direction_field, weight_matrix, ctrl_points, proposal_points, ignore_tags))

        img = img.transpose(1, 2, 0)
        img = ((img * stds + means) * 255).astype(np.uint8)

        distance_map = cav.heatmap(np.array(distance_field * 255 / np.max(distance_field), dtype=np.uint8))
        cv2.imshow("distance_map", distance_map)
        cv2.waitKey(0)

        direction_map = cav.heatmap(np.array(direction_field[0] * 255 / np.max(direction_field[0]), dtype=np.uint8))
        cv2.imshow("direction_field", direction_map)
        cv2.waitKey(0)

        from util.vis_flux import vis_direction_field
        vis_direction_field(direction_field)

        weight_map = cav.heatmap(np.array(weight_matrix * 255 / np.max(weight_matrix), dtype=np.uint8))
        cv2.imshow("weight_matrix", weight_map)
        cv2.waitKey(0)


        # boundary_point = ctrl_points[np.where(ignore_tags!=0)[0]]
        # for i, bpts in enumerate(boundary_point):
        #     cv2.drawContours(img, [bpts.astype(np.int32)], -1, (0, 255, 0), 1)
        #     for j,  pp in enumerate(bpts):
        #         if j==0:
        #             cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (255, 0, 255), -1)
        #         elif j==1:
        #             cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (0, 255, 255), -1)
        #         else:
        #             cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (0, 0, 255), -1)
        #
        #     ppts = proposal_points[i]
        #     cv2.drawContours(img, [ppts.astype(np.int32)], -1, (0, 0, 255), 1)
        #     for j,  pp in enumerate(ppts):
        #         if j==0:
        #             cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (255, 0, 255), -1)
        #         elif j==1:
        #             cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (0, 255, 255), -1)
        #         else:
        #             cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (0, 0, 255), -1)
        #     cv2.imshow('imgs', img)
        #     cv2.waitKey(0)

        from util.misc import split_edge_seqence
        from cfglib.config import config as cfg

        ret, labels = cv2.connectedComponents(np.array(distance_field >0.35, dtype=np.uint8), connectivity=4)
        for idx in range(1, ret):
            text_mask = labels == idx
            ist_id = int(np.sum(text_mask*tr_mask)/np.sum(text_mask))-1
            contours, _ = cv2.findContours(text_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            epsilon = 0.007 * cv2.arcLength(contours[0], True)
            approx = cv2.approxPolyDP(contours[0], epsilon, True).reshape((-1, 2))

            pts_num = approx.shape[0]
            e_index = [(i, (i + 1) % pts_num) for i in range(pts_num)]
            control_points = split_edge_seqence(approx, e_index, cfg.num_points)
            control_points = np.array(control_points[:cfg.num_points, :]).astype(np.int32)

            cv2.drawContours(img, [ctrl_points[ist_id].astype(np.int32)], -1, (0, 255, 0), 1)
            cv2.drawContours(img, [control_points.astype(np.int32)], -1, (0, 0, 255), 1)
            for j,  pp in enumerate(control_points):
                if j == 0:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (255, 0, 255), -1)
                elif j == 1:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (0, 255, 255), -1)
                else:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (0, 255, 0), -1)

            cv2.imshow('imgs', img)
            cv2.waitKey(0)

