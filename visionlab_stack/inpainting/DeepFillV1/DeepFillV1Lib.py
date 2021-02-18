import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(__file__)))

import torch
import numpy as np
import cv2

import DeepFillV1Model


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'INPAINT torch {torch.__version__} '  # string
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'
    return torch.device('cuda:0' if cuda else 'cpu')


class DeepFillV1(object):
    def __init__(self,
                 pretrained_model=None,
                 device='cpu'):
        self.device = select_device(device)

        self.deepfill = DeepFillV1Model.Generator(device=device).to(device)
        model_weight = torch.load(pretrained_model,map_location=device)
        self.deepfill.load_state_dict(model_weight, strict=True)
        self.deepfill.eval()
        print('Load Deepfill Model from', pretrained_model)

    def forward(self, img, mask, image_shape, res_shape):

        img, mask, small_mask = self.data_preprocess(img, mask, size=image_shape)

        image = torch.stack([img])
        mask = torch.stack([mask])
        small_mask = torch.stack([small_mask])

        with torch.no_grad():
            _, inpaint_res, _ = self.deepfill(image.to(self.device), mask.to(self.device), small_mask.to(self.device))

        res_complete = self.data_proprocess(image, mask, inpaint_res, res_shape)

        return res_complete

    def data_preprocess(self, img, mask, enlarge_kernel=0, size=[512, 960]):
        img = img / 127.5 - 1
        mask = (mask > 0).astype(np.int)
        img = cv2.resize(img, (size[1], size[0]))
        if enlarge_kernel > 0:
            kernel = np.ones((enlarge_kernel, enlarge_kernel), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask = (mask > 0).astype(np.uint8)

        small_mask = cv2.resize(mask, (size[1] // 8, size[0] // 8), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)

        if len(mask.shape) == 3:
            mask = mask[:, :, 0:1]
        else:
            mask = np.expand_dims(mask, axis=2)

        if len(small_mask.shape) == 3:
            small_mask = small_mask[:, :, 0:1]
        else:
            small_mask = np.expand_dims(small_mask, axis=2)

        img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
        mask = torch.from_numpy(mask).permute(2, 0, 1).contiguous().float()
        small_mask = torch.from_numpy(small_mask).permute(2, 0, 1).contiguous().float()

        return img*(1-mask), mask, small_mask

    def data_proprocess(self, img, mask, res, res_shape):
        img = img.cpu().data.numpy()[0]
        mask = mask.data.numpy()[0]
        res = res.cpu().data.numpy()[0]

        res_complete = res * mask + img * (1. - mask)
        res_complete = (res_complete + 1) * 127.5
        res_complete = res_complete.transpose(1, 2, 0)
        if res_shape is not None:
            res_complete = cv2.resize(res_complete, (res_shape[1], res_shape[0]))

        return res_complete


class DeepFillV1Class:
    
    def __init__(self,
                    pretrained_model="./data/models/inpainting/DeepFillV1/imagenet_deepfill.pth",
                    image_shape=[512, 960],
                    res_shape=[512, 960],
                    device='cpu'
                ):
        self.deepfill = DeepFillV1(
                            pretrained_model=pretrained_model,
                            device = device
                        )

    def doInpainting(self, image, mask,image_shape, res_shape):        
        with torch.no_grad():
            res = self.deepfill.forward(image, mask,image_shape,res_shape)
        return res


def parse_arges():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_shape', type=int, nargs='+',
                        default=[512, 960])
    parser.add_argument('--res_shape', type=int, nargs='+',
                        default=None)
    parser.add_argument('--pretrained_model', type=str,
                        default='/home/chengao/Weight/imagenet_deepfill.pth')
    parser.add_argument('--test_img', type=str,
                        default='/work/cascades/chengao/DAVIS-540/bear_540p/00000.png')
    parser.add_argument('--test_mask', type=str,
                        default='/work/cascades/chengao/DAVIS-540-baseline/mask_540p.png')
    parser.add_argument('--output_path', type=str,
                        default='/home/chengao/res_00000.png')

    args = parser.parse_args()

    return args


def main():

    args = parse_arges()

    deepfill = DeepFillv1(pretrained_model=args.pretrained_model,
                          image_shape=args.image_shape,
                          res_shape=args.res_shape)

    test_image = cv2.imread(args.test_img)
    mask = cv2.imread(args.test_mask, cv2.IMREAD_UNCHANGED)

    with torch.no_grad():
        img_res = deepfill.forward(test_image, mask)

    cv2.imwrite(args.output_path, img_res)
    print('Result Saved')


if __name__ == '__main__':
    main()
