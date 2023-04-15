#!/usr/bin/env python3
"""Use the MediaPipe Pose solution to detect and track a human pose in video."""
import argparse
import logging
import os
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterator, Optional

import cv2 as cv
import mediapipe as mp
import numpy as np
import numpy.typing as npt
import requests
import rerun as rr

from scipy.spatial.transform import Rotation as R


EXAMPLE_DIR: Final = Path(os.path.dirname(__file__))
DATASET_DIR: Final = EXAMPLE_DIR / "dataset" / "pose_movement"
DATASET_URL_BASE: Final = "https://storage.googleapis.com/rerun-example-datasets/pose_movement"


# PyTorch Hub
import torch

yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s') 
#since we are only intrested in detecting person
yolo_model.classes=[0]

#we need some extra margin bounding box for human crops to be properly detected
MARGIN=10


def track_pose(video_path: str, segment: bool) -> None:

    mp_pose = mp.solutions.pose


    # # Use a separate annotation context for the segmentation mask.
    # rr.log_annotation_context(
    #     "video/mask",
    #     [rr.AnnotationInfo(id=0, label="Background"), rr.AnnotationInfo(id=1, label="Person", color=(0, 0, 0))],
    # )
    
    rr.log_view_coordinates("world", up="-Y", timeless=True)

    # rr.log_cleared("world/camera")

    # It's a non-moving camera so it doesn't go in the for loop
    camera_from_world = get_camera_pose_from_world()


    rr.log_rigid3(
        "camera",
        child_from_parent=camera_from_world,
        timeless=True,
    )

    # Log camera intrinsics
    intrinsics = get_camera_intrinsic_matrix()
    rr.log_pinhole(
        "camera/video",
        child_from_parent=intrinsics,
        width=1280,
        height=720,
        timeless=True,
    )

    rr.log_annotation_context(
        "/",
        rr.ClassDescription(
            info=rr.AnnotationInfo(label="Person"),
            keypoint_annotations=[rr.AnnotationInfo(id=lm.value, label=lm.name) for lm in mp_pose.PoseLandmark],
            keypoint_connections=mp_pose.POSE_CONNECTIONS,
        ),
    )
    
    with closing(VideoSource(video_path)) as video_source:
            
        for bgr_frame in video_source.stream_bgr():
            rgb = cv.cvtColor(bgr_frame.data, cv.COLOR_BGR2RGB)
            rr.set_time_seconds("time", bgr_frame.time)
            rr.set_time_sequence("frame_idx", bgr_frame.idx)
            rr.log_image("camera/video/rgb", rgb)

            h, w, _ = rgb.shape

            # use yolov5 to detect person in the frame
            yolo_result = yolo_model(rgb)
            person_id = 0
            for (xmin, ymin, xmax,   ymax,  confidence,  clas) in yolo_result.xyxy[0].tolist():
                with mp_pose.Pose() as pose:
                    # take each detected person bounding box, crop the original image to the bounding box and have mediapipe detect the pose in the crop
                    results = pose.process(rgb[int(ymin)+MARGIN:int(ymax)+MARGIN,int(xmin)+MARGIN:int(xmax)+MARGIN:])

                    landmark_positions_2d = read_landmark_positions_2d(results, w, h, (xmin, ymin, xmax, ymax))
                    rr.log_points("camera/video/person/"+str(person_id)+"/pose/points", landmark_positions_2d, keypoint_ids=mp_pose.PoseLandmark)

                    landmark_positions_3d = read_landmark_positions_3d(results, (xmin, ymin, xmax, ymax))
                    rr.log_points("camera/person/"+str(person_id)+"/pose/points", landmark_positions_3d, keypoint_ids=mp_pose.PoseLandmark)


                    # move_land_mark_positions_3d(person_id, results)

                    # segmentation_mask = results.segmentation_mask
                    # if segmentation_mask is not None:
                    #     rr.log_segmentation_image("video/mask", segmentation_mask)

                    person_id += 1
                    

def read_landmark_positions_2d(
    results: Any,
    image_width: int,
    image_height: int,
    bbox: tuple,
) -> Optional[npt.NDArray[np.float32]]:
    if results.pose_landmarks is None:
        return None
    else:
        
        normalized_landmarks = [results.pose_landmarks.landmark[lm] for lm in mp.solutions.pose.PoseLandmark]
        # Log points as 3d points with some scaling so they "pop out" when looked at in a 3d view
        # Negative depth in order to move them towards the camera.

        # the normalized_landmarks are normalized to the cropped image, so we need to scale them back to the original image
        bbox_width = bbox[2]-bbox[0]
        bbox_height = bbox[3]-bbox[1]
        return np.array(
            [((bbox_width * lm.x) + bbox[0], (bbox_height * lm.y) + bbox[1], -(lm.z + 1.0) * 300.0) for lm in normalized_landmarks]
        )


def read_landmark_positions_3d(
    results: Any,
    bbox: tuple,
) -> Optional[npt.NDArray[np.float32]]:
    if results.pose_landmarks is None:
        return None
    else:
        landmarks = [results.pose_world_landmarks.landmark[lm] for lm in mp.solutions.pose.PoseLandmark]

        # print("world landmarks", landmarks[0])
        return np.array([(lm.x, lm.y, lm.z) for lm in landmarks])


def get_camera_pose_from_world():
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    translation_list = [-48.17233535998567,0.8712905040130878,52.1138980103538] # scale is 1 = 3.08 cm
    # convert translation list values to meters at scale 1 = 0.00308 m
    translation_list = [x * 0.00308 for x in translation_list]
    # convert to numpy array
    translation_array = np.array(translation_list)


    rotation_nested_list = [[0.21694192407016405,0.2359399679572982,0.9472425946403827],[-0.039333797430330386,0.9716767143068481,-0.23301762863259518],[-0.9753917438447207,0.01329264436285671,0.2200778308812537]]

    rotation_array = np.array(rotation_nested_list)
    r = R.from_matrix(rotation_array)
    r_quat = r.as_quat()

    return (translation_array,r_quat)


    
def get_camera_intrinsic_matrix():
    # intrinsic matrix
    intrinsic_list = [[856.7060693457354,0,349.4553311698114],[0,856.6579127840464,360.2700796586361],[0,0,1]]

    # convert to numpy array
    intrinsic_array = np.array(intrinsic_list)

    return intrinsic_array



    


def move_land_mark_positions_3d(
        person_id: int,
        results: Any
    ) -> None:
    # the landmark positions are person coordinates, so we need to move them to the world coordinates
    # take the metric distrance between person's left and right hip

    # print(mp.solutions.pose.PoseLandmark[23])
    print("Person "+str(person_id)+" camera: left hip", results.pose_landmarks.landmark[23])
    print("Person "+str(person_id)+" camera: right hip", results.pose_landmarks.landmark[24])
    print("Person "+str(person_id)+" world:left hip", results.pose_world_landmarks.landmark[23])
    print("Person "+str(person_id)+" world:right hip", results.pose_world_landmarks.landmark[24])

    # The left hip and the right hip are on both sides of the origin. Add the absolute value of the left hip and the absolute value of the right hip to get the total width of the person.

    
    
    



@dataclass
class VideoFrame:
    data: npt.NDArray[np.uint8]
    time: float
    idx: int


class VideoSource:
    def __init__(self, path: str):
        self.capture = cv.VideoCapture(path)

        if not self.capture.isOpened():
            logging.error("Couldn't open video at %s", path)

    def close(self) -> None:
        self.capture.release()

    def stream_bgr(self) -> Iterator[VideoFrame]:
        while self.capture.isOpened():
            idx = int(self.capture.get(cv.CAP_PROP_POS_FRAMES))
            is_open, bgr = self.capture.read()
            time_ms = self.capture.get(cv.CAP_PROP_POS_MSEC)

            if not is_open:
                break

            yield VideoFrame(data=bgr, time=time_ms * 1e-3, idx=idx)


def get_downloaded_path(dataset_dir: Path, video_name: str) -> str:
    video_file_name = f"{video_name}.mp4"
    destination_path = dataset_dir / video_file_name
    if destination_path.exists():
        logging.info("%s already exists. No need to download", destination_path)
        return str(destination_path)

    source_path = f"{DATASET_URL_BASE}/{video_file_name}"

    logging.info("Downloading video from %s to %s", source_path, destination_path)
    os.makedirs(dataset_dir.absolute(), exist_ok=True)
    with requests.get(source_path, stream=True) as req:
        req.raise_for_status()
        with open(destination_path, "wb") as f:
            for chunk in req.iter_content(chunk_size=8192):
                f.write(chunk)
    return str(destination_path)


def main() -> None:
    # Ensure the logging gets written to stderr:
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger().setLevel("INFO")

    parser = argparse.ArgumentParser(description="Uses the MediaPipe Pose solution to track a human pose in video.")
    parser.add_argument(
        "--video",
        type=str,
        default="backflip",
        choices=["backflip", "soccer"],
        help="The example video to run on.",
    )
    parser.add_argument("--dataset_dir", type=Path, default=DATASET_DIR, help="Directory to save example videos to.")
    parser.add_argument("--video_path", type=str, default="", help="Full path to video to run on. Overrides `--video`.")
    parser.add_argument("--no-segment", action="store_true", help="Don't run person segmentation.")
    rr.script_add_args(parser)

    args = parser.parse_args()
    rr.script_setup(args, "mp_pose")

    video_path = args.video_path  # type: str
    if not video_path:
        video_path = get_downloaded_path(args.dataset_dir, args.video)

    track_pose(video_path, segment=not args.no_segment)

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
