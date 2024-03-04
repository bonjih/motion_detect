import cv2
import os


def save_video(params, video_props):
    out_vid = cv2.VideoWriter(os.path.join(params['output_video_dir'], 'diagnostic.mkv'), video_props['fourcc'],
                              video_props['fps'], (int(video_props['width']), int(video_props['height'])))

    return out_vid
