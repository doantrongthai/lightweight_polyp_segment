import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from utils.dataloader import TestDataset
from utils.metrics import compute_all_metrics, aggregate_metrics
from utils.checkpoint import load_model_weights
from utils.utils import split_model_output, fuse_outputs

TEST_DATASETS = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--test_path', type=str, default='./dataset/TestDataset')
    parser.add_argument('--test', dest='test_dataset', type=str, default='all',
                        choices=['CVC-300', 'CVC-ClinicDB', 'Kvasir',
                                 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'all'])
    parser.add_argument('--testsize', type=int, default=352,
                        help='fixed test/input size; forced to 352')
    parser.add_argument('--save_maps', action='store_true')
    parser.add_argument('--save_path', type=str, default='./result_map')
    parser.add_argument('--threshold', type=float, default=0.5)
    return parser.parse_args()


@torch.no_grad()
def run_dataset(model, test_path, dataset, testsize, device,
                threshold=0.5, save_maps=False, map_dir=None):
    data_path = os.path.join(test_path, dataset)
    image_root = os.path.join(data_path, 'images', '')
    gt_root = os.path.join(data_path, 'masks', '')
    loader = TestDataset(image_root, gt_root, testsize)
    model.eval()

    if save_maps and map_dir:
        os.makedirs(map_dir, exist_ok=True)

    metrics_list = []
    for _ in range(loader.size):
        image, gt, name = loader.load_data()
        gt_np = np.asarray(gt, np.float32)
        gt_np = gt_np / (gt_np.max() + 1e-8)

        image = image.to(device)
        out = model(image)
        main_out, aux_outputs = split_model_output(out)
        main = fuse_outputs(main_out, aux_outputs)

        res = F.interpolate(main, size=gt_np.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        if save_maps and map_dir:
            cv2.imwrite(os.path.join(map_dir, name), (res * 255).astype(np.uint8))

        metrics_list.append(compute_all_metrics(res, gt_np, threshold))

    return aggregate_metrics(metrics_list)


def _print_header():
    cols = f'{"Dataset":<26} {"Dice":>6} {"IoU":>6} {"Prec":>6} {"Rec":>6} ' \
           f'{"Spec":>6} {"MAE":>6} {"Fw":>6} {"Sm":>6} {"Em":>6} {"HD95":>7}'
    print('\n' + '=' * 85)
    print(cols)
    print('=' * 85)


def _print_row(dataset, m):
    print(f'{dataset:<26} {m["dice"]:>6.4f} {m["iou"]:>6.4f} {m["precision"]:>6.4f} '
          f'{m["recall"]:>6.4f} {m["specificity"]:>6.4f} {m["mae"]:>6.4f} '
          f'{m["fw"]:>6.4f} {m["sm"]:>6.4f} {m["em"]:>6.4f} {m["hd95"]:>7.2f}')


def main():
    opt = parse_args()
    opt.testsize = 352

    from models import get_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(opt.model).to(device)
    ckpt = torch.load(opt.weights, map_location=device, weights_only=False)
    load_model_weights(model, ckpt)
    model.eval()

    datasets = TEST_DATASETS if opt.test_dataset == 'all' else [opt.test_dataset]
    _print_header()

    all_metrics = {}
    for ds in datasets:
        map_dir = os.path.join(opt.save_path, opt.model, ds) if opt.save_maps else None
        m = run_dataset(model, opt.test_path, ds, opt.testsize, device,
                        threshold=opt.threshold, save_maps=opt.save_maps, map_dir=map_dir)
        all_metrics[ds] = m
        _print_row(ds, m)

    print('=' * 85)

    if len(datasets) > 1:
        avg = aggregate_metrics(list(all_metrics.values()))
        _print_row('MEAN', avg)
        print('=' * 85)

    if opt.save_maps:
        print(f'Prediction maps saved to: {os.path.join(opt.save_path, opt.model)}')


if __name__ == '__main__':
    main()
