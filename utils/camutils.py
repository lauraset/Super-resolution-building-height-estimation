import torch
import torch.nn.functional as F

def multi_scale_cam(model, inputs, scales):
    cam_list = []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)
            else:
                inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)

            _cam = model(inputs_cat, cam_only=True)
            _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
            _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

            cam_list.append(F.relu(_cam))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5
    return cam


def multi_scale_puzzle(model, inputs, scales):
    cam_list = []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)

        _, _cam = model(inputs_cat, with_cam=True)

        _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

        cam_list = [F.relu(_cam)]

        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _, _cam = model(inputs_cat, with_cam=True)

                _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

                cam_list.append(F.relu(_cam))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5
    return cam


# 2022.11.6
def multi_scale_seam(model, inputs, scales, pos=0):
    cam_list = []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)

        _cam = model(inputs_cat)
        _cam = _cam[pos]

        _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

        if pos==0:
            _cam = F.relu(_cam)

        cam_list = [_cam]

        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam = model(inputs_cat)
                _cam = _cam[pos]

                _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

                if pos == 0:
                    _cam = F.relu(_cam)

                cam_list.append(_cam)

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5
    return cam


## 2022.11.2: add multistage cam
def multi_scale_cam_multistage(model, inputs, scales):
    cam_list = []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)

        cam_all = model(inputs_cat, cam_only=True)
        # all stages
        for _cam in cam_all:
            _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
            _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
            cam_list.append(F.relu(_cam))

        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                cam_all = model(inputs_cat, cam_only=True)
                # all stages
                for _cam in cam_all:
                    _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
                    _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
                    cam_list.append(F.relu(_cam))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5
    return cam


## 2022.11.4: revise the fusion method
def multi_scale_cam_multistagev2(model, inputs, scales, weights=(1,1,1,1)):
    cam_list = [[], [], [], []] # 4 stages
    b, c, h, w = inputs.shape
    with torch.no_grad():
        for s in scales:
            if s==1:
                inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
            else:
                _inputs = F.interpolate(inputs, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

            cam_all = model(inputs_cat, cam_only=True)
            numcam = len(cam_all)
            # all stages
            for i, _cam in enumerate(cam_all):
                _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
                cam_list[i].append(F.relu(_cam))

        # sum for each cam, consider weights
        cam_normlist = []
        for i in range(numcam):
            cam = torch.sum(torch.stack(cam_list[i], dim=0), dim=0)
            cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
            cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5
            cam_normlist.append(cam*weights[i])
        # sum
        cam = torch.sum(torch.stack(cam_normlist, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5

    return cam


## 2022.11.3: from puzzle-net, seam
def make_cam(x, epsilon=1e-5):
    # relu(x) = max(x, 0)
    x = F.relu(x)

    b, c, h, w = x.size()

    flat_x = x.view(b, c, (h * w))
    max_value = flat_x.max(dim=-1)[0].view((b, c, 1, 1))

    return F.relu(x - epsilon) / (max_value + epsilon)
