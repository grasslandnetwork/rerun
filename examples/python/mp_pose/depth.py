import numpy as np
import os
import torch
import utilio
import cv2
from torchvision.transforms import Compose

from dpt.models import DPTDepthModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet




def run(frame, optimize=True):
    """Run MonoDepthNN to compute depth maps.

    Args:
        frame (str): path to input image
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)


    #model_type == "dpt_hybrid_nyu":
    net_w = 640
    net_h = 480

    model = DPTDepthModel(
        path="weights/dpt_hybrid_nyu-2ce69ec7.pt",
        scale=0.000305,
        shift=0.1378,
        invert=True,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )

    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()

    if optimize == True and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)

    # print("start processing {} )".format(frame))

    # if frame.ndim == 2:
    #     print("ndim")
    #     frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0

    
    frame_trnsf = transform({"image": frame})["image"]

    # compute
    with torch.no_grad():
        sample = torch.from_numpy(frame_trnsf).to(device).unsqueeze(0)

        if optimize == True and device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        

        
        # prediction *= 1000.0
        # print("prediction", prediction)
        # utilio.write_depth("depthmap", prediction, bits=2)
        
        return prediction
        


        

