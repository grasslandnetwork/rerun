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
from depth import Depth
import utilio
import json

EXAMPLE_DIR: Final = Path(os.path.dirname(__file__))
DATASET_DIR: Final = EXAMPLE_DIR / "dataset" / "pose_movement"
DATASET_URL_BASE: Final = "https://storage.googleapis.com/rerun-example-datasets/pose_movement"


# PyTorch Hub
import torch

    
from ultralytics import YOLO

yolov8_model = YOLO('yolov8n.pt')

# initialize depth model
depth = Depth()

#we need some extra margin bounding box for human crops to be properly detected
MARGIN=10

import time

def current_milli_time():
    return round(time.time() * 1000)


import datetime
import pytz
def show_initial_frame_time(timestamp):
    # Convert the Unix timestamp in milliseconds to a datetime object
    dt = datetime.datetime.fromtimestamp(timestamp/1000.0, pytz.utc)

    # Convert to the user's local timezone
    local_tz = pytz.timezone('America/New_York')  # replace with your desired timezone
    local_dt = dt.astimezone(local_tz)

    # Format the time as a string with 12 hour format
    time_str = local_dt.strftime("%I:%M:%S %p")  # %I for 12 hour format, %p for AM/PM

    print("The initial time is:", time_str)



def track_pose(video_path: str, segment: bool) -> None:

    mp_pose = mp.solutions.pose


    # # Use a separate annotation context for the segmentation mask.
    # rr.log_annotation_context(
    #     "video/mask",
    #     [rr.AnnotationInfo(id=0, label="Background"), rr.AnnotationInfo(id=1, label="Person", color=(0, 0, 0))],
    # )

    rr.log_annotation_context(
        "/",
        rr.ClassDescription(
            info=rr.AnnotationInfo(label="Person"),
            keypoint_annotations=[rr.AnnotationInfo(id=lm.value, label=lm.name) for lm in mp_pose.PoseLandmark],
            keypoint_connections=mp_pose.POSE_CONNECTIONS,
        ),
    )

    rr.log_view_coordinates("3dpeople", up="-Y", timeless=True)

    intrinsics = get_camera_intrinsic_matrix()

    # primary_camera_from_world = get_camera_pose_from_world(primary=True)
    # rr.log_rigid3(
    #     "camera0",
    #     parent_from_child=primary_camera_from_world,
    #     timeless=True,
    # )

    # rr.log_pinhole(
    #     "camera0/image",
    #     child_from_parent=intrinsics,
    #     width=1280,
    #     height=720,
    #     timeless=True,
    # )

    
    # It's a non-moving camera so it doesn't go in the for loop
    camera_from_world = get_camera_pose_from_world(primary=True)

    rr.log_rigid3(
        "camera",
        child_from_parent=camera_from_world,
        timeless=True,
    )


    # Log camera intrinsics
    rr.log_pinhole(
        "camera/image",
        child_from_parent=intrinsics,
        width=1280,
        height=720,
        timeless=True,
    )

    rr.log_pinhole(
        "camera/depth",
        child_from_parent=intrinsics,
        width=1280,
        height=720,
        timeless=True,
    )
    
    
    
    with closing(VideoSource(video_path)) as video_source:
        file_name = os.path.basename(video_path)+"_poses.json"
        # Open the file in write a blank list to it
        with open(file_name, 'w') as f:
            json.dump([], f)


        # set initial frame timestamp for the start of the video file 
        frame_timestamp = current_milli_time()

        show_initial_frame_time(frame_timestamp)
            
        for bgr_frame in video_source.stream_bgr():
            rgb = cv.cvtColor(bgr_frame.data, cv.COLOR_BGR2RGB)
            rr.set_time_seconds("time", bgr_frame.time)
            rr.set_time_sequence("frame_idx", bgr_frame.idx)
            rr.log_image("camera/image", rgb) # don't put camera/image/rgb or it won't show the keypoints

            h, w, _ = rgb.shape

            # use yolov8 to detect person in the frame
            # since we are only intrested in detecting person, we use classes=[0]
            yolov8_results = yolov8_model(rgb, stream=True, classes=[0])
            
            # extract xmin, ymin, xmax,   ymax,  confidence,  clas from yolov8_results
                        
            depth_map = depth.run(rgb)

            # rgb_depth = cv.imread("depthmap.png")
            depth_image = utilio.depth_to_numpy_array(depth_map)

            rr.log_image("camera/depth", depth_image) # don't put camera/image/rgb or it won't show the keypoints
            

            # for (xmin, ymin, xmax, ymax,  confidence,  clas) in yolo_result.xyxy[0].tolist():

            # Add the current position of the video file in milliseconds to the frame_timestamp
            frame_timestamp += round(bgr_frame.time)
            
            multiple_poses = []

            for yolov8_result in yolov8_results:
                for person_id, subresult in enumerate(yolov8_result.boxes.xyxy):
                    (xmin, ymin, xmax, ymax) = subresult.tolist()
                    confidence = yolov8_result.boxes.conf.tolist()
                    this_color = get_color(person_id)
                    rr.log_scalar("confidence/person/"+str(person_id), confidence[person_id], color=this_color)

                    with mp_pose.Pose() as pose:
                        # take each detected person bounding box, crop the original image to the bounding box and have mediapipe detect the pose in the crop
                        results = pose.process(rgb[int(ymin)+MARGIN:int(ymax)+MARGIN,int(xmin)+MARGIN:int(xmax)+MARGIN:])

                        landmark_positions_2d = read_landmark_positions_2d(results, depth_map, w, h, (xmin, ymin, xmax, ymax))

                        if confidence[person_id] >= 0.68:
                            rr.log_points("camera/image/2dpeople/"+str(person_id)+"/pose/keypoints", landmark_positions_2d, keypoint_ids=mp_pose.PoseLandmark)

                            rr.log_points("camera/depth/2dpeople/"+str(person_id)+"/pose/keypoints", landmark_positions_2d, keypoint_ids=mp_pose.PoseLandmark)

                            landmark_positions_3d = read_landmark_positions_3d(results, landmark_positions_2d, depth_map, (xmin, ymin, xmax, ymax))
                            rr.log_points("3dpeople/"+str(person_id)+"/pose/keypoints", landmark_positions_3d, keypoint_ids=mp_pose.PoseLandmark)
                            

                            # if it's not none, add it to the list of poses in multiple_poses
                            if landmark_positions_3d is not None:
                                pose = [{"x":lm[0], "y":lm[1], "z":lm[2], "timestamp": frame_timestamp, "frame_idx": bgr_frame.idx} for lm in landmark_positions_3d]
                                multiple_poses.append(pose)

                        else:
                            rr.log_points("camera/image/lowconf/"+str(person_id)+"/pose/keypoints", landmark_positions_2d, keypoint_ids=mp_pose.PoseLandmark)


            # reorganize the list of poses in this frame so that they are connected according to the connections in mp_pose.POSE_CONNECTIONS
            new_connected_multiple_poses = []
            for pose in multiple_poses:
                keypoints3D = pose
                for start_idx, end_idx in mp_pose.POSE_CONNECTIONS:
                    start_position = keypoints3D[start_idx]
                    end_position = keypoints3D[end_idx]
                    if start_position and end_position:
                        new_connected_multiple_poses.append({'start': start_position.copy(), 'end': end_position.copy()})

                    

            # Load existing JSON data from file
            with open(file_name, 'r') as f:
                existing_connected_multiple_poses = json.load(f)

            # Append new existing_connected_multiple_poses to array
            existing_connected_multiple_poses.extend(new_connected_multiple_poses)

            # Write modified JSON existing_connected_multiple_poses back to file
            with open(file_name, "w") as f:
                json.dump(existing_connected_multiple_poses, f, indent=4)


def read_landmark_positions_2d(
    results: Any,
    depth_map: Any,
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
        #depth_map[(bbox_width * lm.x) + bbox[0], (bbox_height * lm.y) + bbox[1]]

        pixel_landmarks = []
        idx = 0
        for lm in normalized_landmarks:
            x = (bbox_width * lm.x) + bbox[0]
            y = (bbox_height * lm.y) + bbox[1]
            z = 0
                
            pixel_landmarks.append([x, y, z])
            idx += 1
            
        return np.array(pixel_landmarks)
    
        # return np.array(
        #     [((bbox_width * lm.x) + bbox[0], (bbox_height * lm.y) + bbox[1], -(lm.z + 1.0) * 300.0) for lm in normalized_landmarks]
        # )



def read_landmark_positions_3d(
    results: Any,
    landmark_positions_2d: Any,
    depth_map: Any,
    bbox: tuple,
) -> Optional[npt.NDArray[np.float32]]:
    if results.pose_landmarks is None:
        return None
    else:

        # these landmarks only provide the relative depth, one against the other. With the zero point being midway between the hips
        world_landmarks = [results.pose_world_landmarks.landmark[lm] for lm in mp.solutions.pose.PoseLandmark]

        # give me a python list between 0 and 33 but without the numbers in a particular list

                
        # get the real world depth of particular landmark tht we'll use as a reference landmark
        ideal_ref_lm_list = [23, 24, 11, 12, 0, 31] #try for the left hip, right hip, left shoulder, right shoulder, nose, and left toe first
        
        alt_rf_lm_list = [i for i in range(33) if i not in ideal_ref_lm_list] # otherwise, try any other landmark
        ref_lm_list = ideal_ref_lm_list + alt_rf_lm_list
        
        for idx, ref_lm_num in enumerate(ref_lm_list):
            # sometimes the landmark is outside the image, so we need try till we get one that's inside the image
            try:
                # take the landmark's pixel coordinate and use that to find the corresponding depth value
                rwd_ref_lm = depth_map[int(landmark_positions_2d[ref_lm_num,1]), int(landmark_positions_2d[ref_lm_num,0])]
                break 
            except IndexError:
                if idx == len(ref_lm_list)-1:
                    print("couldn't find any real world landmark depths")
                    # bubble up the error
                    raise IndexError
        
                
        # create a numpy array from  world_landmarks to find the relative depth of the chosen reference landmark
        world_landmarks_np = np.array([(lm.x, lm.y, lm.z) for lm in world_landmarks])
        reld_ref_lm = world_landmarks_np[ref_lm_num,2]                              

                                      
        # Intrinsic matrix K
        K = get_camera_intrinsic_matrix()
        
        pixel_landmarks = []
        idx = 0

        # now we'll convert the world_landmark's to real world coordinates                            
        for lm in world_landmarks:

            #lm.z = rwd_left_hip + lm.z
            
            # get the landmark's (lm) real world z coordinate 
            # the formula is rwd(lm) = rwd(ref_lm) + (reld(lm) - reld(ref_lm))
            lm.z = rwd_ref_lm + (lm.z - reld_ref_lm)

            # get the landmark's real world x and y coordinates using the intrinsic matrix and it's real world z coordinate
            u = landmark_positions_2d[idx,0]
            v = landmark_positions_2d[idx,1]
            # 2D image coordinates (u, v)
            image_coordinates = np.array([u, v, 1])
            
            # Compute the 3D world space coordinates
            K_inv = np.linalg.inv(K)

            homogeneous_world_coordinates = lm.z * K_inv @ image_coordinates
            X_world, Y_world, _ = homogeneous_world_coordinates

            lm.x = X_world
            lm.y = Y_world
                
            pixel_landmarks.append([lm.x, lm.y, lm.z])
            # print("idx", idx)
            idx += 1


        
        # return the new world_landmarks
        return np.array([(lm.x, lm.y, lm.z) for lm in world_landmarks])


def get_camera_pose_from_world(primary=False):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    translation_list = [-48.17233535998567,0.8712905040130878,52.1138980103538] # scale is 1 = 3.08 cm
    if primary:
        translation_list = [0.0,0.0,0.0]
    # convert translation list values to meters at scale 1 = 0.00308 m
    translation_list = [x * 0.00308 for x in translation_list]
    # convert to numpy array
    translation_array = np.array(translation_list)


    rotation_nested_list = [
        [0.21694192407016405, 0.2359399679572982, 0.9472425946403827],
        [-0.039333797430330386, 0.9716767143068481, -0.23301762863259518],
        [-0.9753917438447207, 0.01329264436285671, 0.2200778308812537]
    ]

    if primary:
        rotation_nested_list = [1.0, 0.0, 0.0 ],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]

    rotation_array = np.array(rotation_nested_list)
    r = R.from_matrix(rotation_array)
    r_quat = r.as_quat()

    return (translation_array,r_quat)


    
def get_camera_intrinsic_matrix():
    # intrinsic matrix
    intrinsic_list = [
        [856.7060693457354, 0, 349.4553311698114],
        [0, 856.6579127840464, 360.2700796586361],
        [0, 0, 1]
    ]
    # intrinsic_list = [
    #     [500, 0, 100],
    #     [0, 500, 100],
    #     [0, 0, 1]
    # ]

    # convert to numpy array
    intrinsic_array = np.array(intrinsic_list)

    return intrinsic_array



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


def get_color(num):
    colors = [
        [255, 0, 0],
        [255, 165, 0],
        [255, 255, 0],
        [0, 255, 0],
        [0, 127, 255],
        [0, 0, 255],
        [139, 0, 255],
        [255, 0, 255],
        [0, 0, 0]
    ]
    
    return colors[num]


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
