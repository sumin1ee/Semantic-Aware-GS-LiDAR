#
# Copyright (C) 2025, Fudan Zhang Vision Group
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE and LICENSE_gaussian_splatting.md files.
#
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import lpips
from utils.system_utils import save_ply
from skimage.metrics import structural_similarity
from chamfer.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from chamfer.fscore import fscore


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))

        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f"PSNR = {self.measure():.6f}"


class RMSEMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        rmse = (truths - preds) ** 2
        rmse = np.sqrt(rmse.mean())

        self.V += rmse
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "RMSE"), self.measure(), global_step)

    def report(self):
        return f"RMSE = {self.measure():.6f}"


class MAEMeter:
    def __init__(self, intensity_inv_scale=1.0):
        self.V = 0
        self.N = 0
        self.intensity_inv_scale = intensity_inv_scale

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        # Mean Absolute Error
        mae = np.abs(
            truths * self.intensity_inv_scale - preds * self.intensity_inv_scale
        ).mean()

        self.V += mae
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "MAE"), self.measure(), global_step)

    def report(self):
        return f"MAE = {self.measure():.6f}"


class DepthMeter:
    def __init__(self, scale, lpips_fn=None):
        self.V = []
        self.N = 0
        self.scale = scale
        self.lpips_fn = lpips.LPIPS(net='alex').eval()

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds = preds / self.scale
        truths = truths / self.scale
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        # simplified since max_pixel_value is 1 here.
        depth_error = self.compute_depth_errors(truths, preds)

        depth_error = list(depth_error)
        self.V.append(depth_error)
        self.N += 1

    def compute_depth_errors(
            self, gt, pred, min_depth=1e-6, max_depth=80,
    ):
        pred[pred < min_depth] = min_depth
        pred[pred > max_depth] = max_depth
        gt[gt < min_depth] = min_depth
        gt[gt > max_depth] = max_depth

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        medae = np.median(np.abs(gt - pred))

        if gt.shape[-2] >= 32:
            lpips_loss = self.lpips_fn(torch.from_numpy(pred).squeeze(0),
                                       torch.from_numpy(gt).squeeze(0), normalize=True).item()
        else:
            lpips_loss = 1

        ssim = structural_similarity(
            pred.squeeze(0), gt.squeeze(0), data_range=np.max(gt) - np.min(gt)
        )

        psnr = 10 * np.log10(max_depth ** 2 / np.mean((pred - gt) ** 2))

        return rmse, medae, lpips_loss, ssim, psnr

    def measure(self):
        assert self.N == len(self.V)
        return np.array(self.V).mean(0)

    def write(self, writer, global_step, prefix="", suffix=""):
        writer.add_scalar(
            os.path.join(prefix, f"depth error{suffix}"), self.measure()[0], global_step
        )

    def report(self):
        return f"Depth_error = {self.measure()}"


class SSIMMeter:
    def __init__(self, device=None):
        self.V = 0
        self.N = 0

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def clear(self):
        self.V = 0
        self.N = 0

    # def prepare_inputs(self, *inputs):
    #     outputs = []
    #     for i, inp in enumerate(inputs):
    #         inp = inp.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
    #         inp = inp.to(self.device)
    #         outputs.append(inp)
    #     return outputs

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)
        ssim = structural_similarity(
            preds.squeeze(0).squeeze(-1), truths.squeeze(0).squeeze(-1)
        )

        # preds, truths = self.prepare_inputs(
        #     preds, truths)  # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]

        # ssim = structural_similarity_index_measure(preds, truths)

        self.V += ssim
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "SSIM"), self.measure(), global_step)

    def report(self):
        return f"SSIM = {self.measure():.6f}"


class PointsMeter:
    def __init__(self, scale, vfov):
        self.V = []
        self.N = 0
        self.scale = scale
        self.vfov = vfov
        self.hfov = (-180, 180)
        self.near = 0.2
        self.far = 80

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def pano_to_lidar(self, range_image):
        # TODO: 确认下这行到底要不要
        range_image[range_image > self.far] = 0  # self.far
        panorama_height, panorama_width = range_image.shape[-2:]
        theta, phi = torch.meshgrid(torch.arange(panorama_height, device=range_image.device),
                                    torch.arange(panorama_width, device=range_image.device), indexing="ij")

        vertical_degree_range = self.vfov[1] - self.vfov[0]
        theta = (90 - self.vfov[1] + theta / panorama_height * vertical_degree_range) * torch.pi / 180

        horizontal_degree_range = self.hfov[1] - self.hfov[0]
        phi = (self.hfov[0] + phi / panorama_width * horizontal_degree_range) * torch.pi / 180

        dx = torch.sin(theta) * torch.sin(phi)
        dz = torch.sin(theta) * torch.cos(phi)
        dy = -torch.cos(theta)

        directions = torch.stack([dx, dy, dz], dim=0)
        directions = F.normalize(directions, dim=0)

        points_xyz = directions * range_image
        points_xyz = points_xyz.reshape(3, -1).permute(1, 0)
        points_xyz = points_xyz[points_xyz.norm(dim=1) > self.near]

        return points_xyz

    def update(self, preds, truths):
        preds = preds / self.scale
        truths = truths / self.scale
        # preds, truths = self.prepare_inputs(
        #     preds, truths
        # )  # [B, N, 3] or [B, H, W, 3], range[0, 1]
        chamLoss = chamfer_3DDist()
        pred_lidar = self.pano_to_lidar(preds[0])
        gt_lidar = self.pano_to_lidar(truths[0])
        # pred_lidar = self.discard_outliers(pred_lidar, 1.0)

        dist1, dist2, idx1, idx2 = chamLoss(pred_lidar[None, ...], gt_lidar[None, ...])
        chamfer_dis = dist1.mean() + dist2.mean()
        threshold = 0.05  # monoSDF
        f_score, precision, recall = fscore(dist1, dist2, threshold)
        f_score = f_score.cpu()[0]

        # save_ply(pred_lidar, "origin.ply")
        # drop_lidar = self.visualize_outliers(pred_lidar, dist1, 0.98)
        # save_ply(drop_lidar, "drop.ply")

        self.V.append([chamfer_dis.cpu(), f_score])

        self.N += 1

    @staticmethod
    def visualize_outliers(points, dist, threshold):
        dist_threshold = torch.quantile(dist, threshold)
        filtered_points = points[(dist <= dist_threshold)[0]]
        return filtered_points

    def measure(self):
        # return self.V / self.N
        assert self.N == len(self.V)
        return np.array(self.V).mean(0)

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "CD"), self.measure()[0], global_step)

    def report(self):
        return f"CD f-score = {self.measure()}"


class RaydropMeter:
    def __init__(self, ratio=0.5):
        self.V = []
        self.N = 0
        self.ratio = ratio

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]
        results = []

        rmse = (truths - preds) ** 2
        rmse = np.sqrt(rmse.mean())
        results.append(rmse)

        preds_mask = np.where(preds > self.ratio, 1, 0)
        acc = (preds_mask == truths).mean()
        results.append(acc)

        TP = np.sum((truths == 1) & (preds_mask == 1))
        FP = np.sum((truths == 0) & (preds_mask == 1))
        TN = np.sum((truths == 0) & (preds_mask == 0))
        FN = np.sum((truths == 1) & (preds_mask == 0))

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        results.append(f1)

        self.V.append(results)
        self.N += 1

    def measure(self):
        assert self.N == len(self.V)
        return np.array(self.V).mean(0)

    def write(self, writer, global_step, prefix="", suffix=""):
        writer.add_scalar(os.path.join(prefix, "raydrop error"), self.measure()[0], global_step)

    def report(self):
        return f"Rdrop_error (RMSE, Acc, F1) = {self.measure()}"


class IntensityMeter:
    def __init__(self, scale, lpips_fn=None):
        self.V = []
        self.N = 0
        self.scale = scale
        self.lpips_fn = lpips.LPIPS(net='alex').eval()

    def clear(self):
        self.V = []
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds = preds / self.scale
        truths = truths / self.scale
        preds, truths = self.prepare_inputs(
            preds, truths
        )  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        # simplified since max_pixel_value is 1 here.
        intensity_error = self.compute_intensity_errors(truths, preds)

        intensity_error = list(intensity_error)
        self.V.append(intensity_error)
        self.N += 1

    def compute_intensity_errors(
            self, gt, pred, min_intensity=1e-6, max_intensity=1.0,
    ):
        pred[pred < min_intensity] = min_intensity
        pred[pred > max_intensity] = max_intensity
        gt[gt < min_intensity] = min_intensity
        gt[gt > max_intensity] = max_intensity

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        medae = np.median(np.abs(gt - pred))

        if gt.shape[-2] >= 32:
            lpips_loss = self.lpips_fn(torch.from_numpy(pred).squeeze(0),
                                       torch.from_numpy(gt).squeeze(0), normalize=True).item()
        else:
            lpips_loss = 1

        ssim = structural_similarity(
            pred.squeeze(0), gt.squeeze(0), data_range=np.max(gt) - np.min(gt)
        )

        psnr = 10 * np.log10(max_intensity ** 2 / np.mean((pred - gt) ** 2))

        return rmse, medae, lpips_loss, ssim, psnr

    def measure(self):
        assert self.N == len(self.V)
        return np.array(self.V).mean(0)

    def write(self, writer, global_step, prefix="", suffix=""):
        writer.add_scalar(
            os.path.join(prefix, f"intensity error{suffix}"), self.measure()[0], global_step
        )

    def report(self):
        return f"Inten_error = {self.measure()}"


class SemanticMeter:
    def __init__(self, num_classes: int, ignore_index: int = -1):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def clear(self):
        self.confusion.fill(0)

    @staticmethod
    def _to_numpy(tensor):
        if torch.is_tensor(tensor):
            tensor = tensor.detach().cpu().numpy()
        return tensor

    def update(self, preds, truths):
        preds = self._to_numpy(preds).reshape(-1)
        truths = self._to_numpy(truths).reshape(-1)

        # Filter out ignore_index and out-of-range values
        # Only filter based on GT, but clamp preds to valid range
        valid_mask = (truths != self.ignore_index) & (truths >= 0) & (truths < self.num_classes)
        
        if valid_mask.sum() == 0:
            return
        
        preds = preds[valid_mask]
        truths = truths[valid_mask]
        
        # Clamp predictions to valid range [0, num_classes)
        preds = np.clip(preds, 0, self.num_classes - 1)
        
        # Ensure values are non-negative integers for bincount
        preds = preds.astype(np.int64)
        truths = truths.astype(np.int64)
        
        combined = truths * self.num_classes + preds
        counts = np.bincount(combined, minlength=self.num_classes ** 2)
        self.confusion += counts.reshape(self.num_classes, self.num_classes)

    def measure(self):
        total = self.confusion.sum()
        if total == 0:
            return 0.0, 0.0

        diag = np.diag(self.confusion)
        accuracy = diag.sum() / total

        # Calculate IoU per class
        # IoU = TP / (TP + FP + FN) = diag / (row_sum + col_sum - diag)
        row_sum = self.confusion.sum(axis=1)  # Ground truth counts per class
        col_sum = self.confusion.sum(axis=0)    # Prediction counts per class
        denom = row_sum + col_sum - diag
        valid = denom > 0
        iou = np.zeros(self.num_classes, dtype=np.float64)
        iou[valid] = diag[valid] / denom[valid]
        # Only average over classes that appear in GT (row_sum > 0)
        classes_in_gt = row_sum > 0
        if classes_in_gt.any():
            miou = iou[classes_in_gt].mean()
        else:
            miou = 0.0
        return float(accuracy), float(miou)
