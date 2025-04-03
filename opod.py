#!/usr/bin/env python3

import argparse
import multiprocessing
import queue
import sys
import threading
from functools import lru_cache

import cv2
import numpy as np

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)

# ------------------------------------------------
# SAFETY TIPS from your second code
# ------------------------------------------------
SAFETY_TIPS = {
    "bird": "Most birds are harmless, but avoid feeding or touching them to prevent disease transmission.",
    "cat": "Approach cats slowly and let them come to you. Avoid sudden movements or loud noises.",
    "dog": "Do not approach unfamiliar dogs. Let them sniff you first and avoid direct eye contact.",
    "horse": "Stay calm and avoid standing behind a horse where you can't be seen — they may kick.",
    "sheep": "Sheep are generally gentle; avoid chasing or cornering them.",
    "cow": "Keep a respectful distance from cows, especially if calves are present — they can become protective.",
    "elephant": "Elephants are powerful and unpredictable in the wild. Keep a long distance and stay quiet.",
    "bear": "Never run from a bear. Back away slowly and make yourself look big. Avoid eye contact.",
    "zebra": "Zebras can kick if threatened. Do not approach or try to feed them in the wild.",
    "giraffe": "Giraffes are usually peaceful, but avoid getting too close or standing underneath them."
}

# We’ll keep track of which labels we’ve already printed
seen_detections = set()

class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


def parse_detections(metadata: dict):
    """Parse the output tensor into a list of Detection objects, scaled to the ISP output."""
    bbox_normalization = intrinsics.bbox_normalization
    threshold = args.threshold
    iou = args.iou
    max_detections = args.max_detections

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return None

    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = \
            postprocess_nanodet_detection(
                outputs=np_outputs[0],
                conf=threshold,
                iou_thres=iou,
                max_out_dets=max_detections
            )[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h
        # Split the boxes into x,y,w,h
        boxes = np.array_split(boxes, 4, axis=1)
        boxes = zip(*boxes)

    detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > threshold
    ]
    return detections


@lru_cache
def get_labels():
    labels = intrinsics.labels
    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels


def draw_detections(jobs):
    """Draw frames on an OpenCV window, but with bounding boxes/commented out."""
    labels = get_labels()

    # We'll keep track of the last detections in case new metadata is None
    last_detections = []

    while (job := jobs.get()) is not None:
        request, async_result = job
        detections = async_result.get()
        if detections is None:
            detections = last_detections
        else:
            last_detections = detections

        with MappedArray(request, 'main') as m:
            # Terminal printing for each new detection
            for detection in detections:
                label_str = labels[int(detection.category)]
                conf_str = detection.conf
                # If not yet seen this label, print detection + safety tip
                if label_str not in seen_detections:
                    print(f"Detected: {label_str} with confidence {conf_str:.2f}")
                    if label_str in SAFETY_TIPS:
                        print(f"Safety Tip: {SAFETY_TIPS[label_str]}")
                    seen_detections.add(label_str)

            # -----------------------------------------------
            # If you want to comment out the bounding boxes and text,
            # simply do NOT draw them on the frame:
            #
            # # for detection in detections:
            # #     x, y, w, h = detection.box
            # #     label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"
            # #     # ... code to draw rect or text ...
            #
            # We'll skip all that so the live feed has no overlays.
            # -----------------------------------------------

            # If preserving aspect ratio, you can also skip drawing ROI here:
            # if intrinsics.preserve_aspect_ratio:
            #     b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
            #     color = (255, 0, 0)
            #     # skip rectangle or text

            # Show the raw frame with no overlays in an OpenCV window
            cv2.imshow('IMX500 Object Detection', m.array)
            cv2.waitKey(1)

        # Release the request so Picamera2 can reuse the buffer
        request.release()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model",
                        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction, help="Normalize bbox")
    parser.add_argument("--threshold", type=float, default=0.55, help="Detection threshold")
    parser.add_argument("--iou", type=float, default=0.65, help="Set iou threshold")
    parser.add_argument("--max-detections", type=int, default=10, help="Set max detections")
    parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction, help="Remove '-' labels ")
    parser.add_argument("--postprocess", choices=["", "nanodet"],
                        default=None, help="Run post process of type")
    parser.add_argument("-r", "--preserve-aspect-ratio", action=argparse.BooleanOptionalAction,
                        help="Preserve the pixel aspect ratio of the input tensor")
    parser.add_argument("--labels", type=str, help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="Print JSON network_intrinsics then exit")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Must be done before Picamera2 instantiation
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        sys.exit(1)

    # Override intrinsics from args
    for key, value in vars(args).items():
        if key == 'labels' and value is not None:
            with open(value, 'r') as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    # If no labels provided, default to coco
    if intrinsics.labels is None:
        with open("assets/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        sys.exit(0)

    # Create and start the camera
    picam2 = Picamera2(imx500.camera_num)
    main = {'format': 'RGB888'}

    # Use a preview configuration but no built-in preview
    config = picam2.create_preview_configuration(
        main,
        controls={"FrameRate": intrinsics.inference_rate},
        buffer_count=12
    )

    # Optional progress bar
    imx500.show_network_fw_progress_bar()

    # Start camera (no built-in PiCamera2 preview)
    picam2.start(config, show_preview=False)

    # If preserving aspect ratio
    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    # Create a multiprocessing pool for parse_detections
    pool = multiprocessing.Pool(processes=4)

    # Queue to pass requests + parse results to the drawing thread
    jobs = queue.Queue()

    # Separate thread to handle drawing in order
    thread = threading.Thread(target=draw_detections, args=(jobs,))
    thread.start()

    # Main loop: capture requests, parse metadata in parallel
    while True:
        request = picam2.capture_request()
        metadata = request.get_metadata()
        if metadata:
            async_result = pool.apply_async(parse_detections, (metadata,))
            jobs.put((request, async_result))
        else:
            request.release()
