import cv2
from pathlib import Path
from glob import glob
import argparse
import json
import sys
import os
#from tqdm import tqdm
import pandas as pd
import logging
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--InputParameters", help="Path of input JSON parameter file")
args = parser.parse_args()


def stream_logger_setup(params):
    logger = logging.getLogger()
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(params['log']['log_level'])
    formatter = logging.Formatter(params['log']['log_format'])
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.setLevel(params['log']['log_level'])

    return logger

def main(params: dict):
    logger = stream_logger_setup(params)
    img_format = 'jpg'
    
    sentinel = 0
    prev_frame = None

    df_lst = []
    try:
        cap = cv2.VideoCapture(params['input_video_path'])

        video_props = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            'height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            'fourcc': encoding(int(cap.get(cv2.CAP_PROP_FOURCC))),

        }
        
        empty_frame = np.zeros((int(video_props['height']), int(video_props['width']), 3), dtype=np.uint8)

        logger.info(f"Reading video with resolution: {video_props['width']}x{video_props['height']} at fps: {video_props['fps']} and encoding {video_props['fourcc']}")
        if('log' in params['log']['trace_modes']):
            with open(params['log']['output_log_path'], 'w') as fh:
                fh.write("ts,motion,motion_sma,exceeded1,exceeded2\r\n")
        
        outVid = None
        if('video' in params['log']['trace_modes']):
            logger.info("Creating diagnostic video in videos folder")
            outVid = cv2.VideoWriter(os.path.join(params['output_video_dir'],'diagnostic.mkv'),cv2.VideoWriter_fourcc(*video_props['fourcc']), video_props['fps'], (int(video_props['width']),int(video_props['height'])))
        sentinel = True
        while sentinel:
            ret, frame = cap.read()
            if not ret:
                sentinel = False
                break
            seconds = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0

            if 'process_seconds' in params and params['process_seconds'] <= seconds:
                sentinel = False # finish this frame

            # Video consumer state machine
            if prev_frame is None:
                prev_frame = frame
                continue
            else:
                diff_im = frame_diff(frame_preprocess(prev_frame), frame_preprocess(frame), params['roi'])
                prev_frame = frame

                df_vals = get_df_vals(
                    seconds,
                    int(video_props['fps']),
                    df_lst,
                    diff_im.sum(),
                    params["thresholds"]["daytime"] if params["daytime"] else params["thresholds"]["nighttime"]
                )

                df_lst.append(df_vals)
                
                trace_diff(
                    frame,
                    diff_im,
                    df_vals,
                    params=params,
                    video=outVid,
                    empty_frame=empty_frame,
                    roi_rect=params['roi']
                )

        write_df(pd.DataFrame(df_lst), params)
        
    except Exception as e:
        logging.error(str(e), exc_info=e)
    finally:
        logger.info("Task completed")

def get_df_vals(seconds: float, fps: int, df_lst: list, sum, sma_params: dict):
    """
        fps: input video frame rate as an int
        seconds: input video timestamp
        df_list: reference output list for sma calculation
        params: parameters for sma and thresholds settings

    """
    sma = 0.0
    fps_n =  sma_params["interval_seconds"] * (fps) -1 
    if len(df_lst) >= fps_n:
        for i in range(len(df_lst)-1, (len(df_lst)-1)-fps_n, -1):
            sma = sma + df_lst[i]["motion_sum"]

    sma = (sma + sum) / float(fps_n)
    df_vals = {
        "sec": seconds,
        "motion_sum": sum,
        "motion_sma": sma,
        "motion_exceeded_35": sma >= sma_params["t0"], # 0.2[0.0, 152974.7, 305949.4, 458924.1, 611898.8, 764873.5, 917848.2, 1070822.9000000001, 1223797.6, 1376772.3, 1529747.0]
        "motion_exceeded_50": sma >= sma_params["t1"], # 0.25  # [1529747.0, 1682721.7, 1835696.4, 1988671.1, 2141645.8000000003, 2294620.5, 2447595.2, 2600569.9000000004, 2753544.6, 2906519.3, 3059494.0]
    }
    return df_vals


def frame_preprocess(frame_ref):
    return cv2.cvtColor(frame_ref, cv2.COLOR_BGR2RGB)

def region_of_interest(first, second, roi=None):
    if roi is not None:
        return (first[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]], second[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]])

    return (first, second)

def frame_diff(first, second, roi=None):
    """
        frame diff operation
    """
    
    first_roi, second_roi = region_of_interest(first, second, roi)
    diff_im = cv2.subtract(second_roi, first_roi)
    
    conv_gray = cv2.cvtColor(diff_im, cv2.COLOR_RGB2GRAY)
    ret, mask = cv2.threshold(conv_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    diff_im[mask!=255] = [0,0,255]
    
    return diff_im

def trace_diff(current_frame, diff_img, labels, params=None, video=None, empty_frame=None, roi_rect=None):
    """
    function to manage script outputs, depending on log options included in modes
        modes supported: 'logs', 'video'
    """
    for m in params['log']['trace_modes']:
        if m == 'log':
            with open(params['log']['output_log_path'], 'a') as fh:
                fh.write(",".join([str(labels[x]) for x in labels]) + "\n")
        elif m == "video" and not(video is None or empty_frame is None or roi_rect is None):
            roi = roi_rect
            color = (0, 255, 0)
            thickness = 2

            diff_img = cv2.rectangle(diff_img, (0, 0), (roi['x2'] - roi['x1'], roi['y2'] - roi['y1']), color, thickness)
            empty_frame[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]] = diff_img
            diff_img
            alpha = 0.3
            beta = 1 - alpha
            combined_frame = cv2.addWeighted(empty_frame, alpha, current_frame, beta, 0.0)
            video.write(frame_label(combined_frame, labels))


def frame_label(frame_ref, labels: dict):
    """
        adds diagnostic text to the frame
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    font_scale = 1
    color = (255,255,0)

    thickness = 2
    temp_frame = frame_ref
    line_height = 30
    y = 50
    for i in labels:
        org = (50, y)
        temp_frame = cv2.putText(frame_ref, str(f"{i}: {labels[i]}"), org, font, font_scale, color, thickness, cv2.LINE_AA)
        y = y + line_height

    return temp_frame

def write_df(df: pd.DataFrame, params):
    if Path(params['output_df_path']).suffix == ".parquet":
        df.to_parquet(params['output_df_path'])
    elif Path(params['output_df_path']).suffix == ".csv":
        df.to_csv(output_df_path)

def encoding(fourcc):
    """
        get encoding from video as *char
    """
    return chr(int(fourcc & 0xFF)) + chr(int((fourcc & 0xFF00) >> 8)) + chr(int((fourcc & 0xFF0000) >> 16)) + chr(int((fourcc & 0xFF000000) >> 24))


if __name__== "__main__":
    try:
        parameters = None
        with open(args.InputParameters, "r") as f:
            parameters = json.load(f)
                    
        main(parameters)

    except BaseException as e:
        print(f"Script arguments could not be read due to {e}")
        sys.exit()
