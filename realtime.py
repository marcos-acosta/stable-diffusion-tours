from diffusionify import Diffusioner
from image_processing import center_crop

import cv2
import numpy as np

def run_realtime(diffusioner: Diffusioner, camera_id: int):
    cap = cv2.VideoCapture(camera_id)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = center_crop(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            diffusioned_frame = diffusioner.diffusionify(image_arr=frame_rgb)
        except Exception as e:
            print(f"[ERROR]: {e}")
            return
        diffusioned_frame_bgr = cv2.cvtColor(diffusioned_frame, cv2.COLOR_RGB2BGR)
        frame_512 = cv2.resize(frame, (512, 512))
        concatenated_frames = np.concatenate((frame_512, diffusioned_frame_bgr), axis=1)
        cv2.imshow("Camera", concatenated_frames)
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# run_realtime(None, 0)