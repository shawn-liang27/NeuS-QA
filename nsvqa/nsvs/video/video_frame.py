from typing import List
import numpy as np
import cv2


class VideoFrame:
    """Frame class."""
    def __init__(
        self,
        frame_idx_list: list[int],
        object_of_interest: dict
    ):
        self.frame_idx_list = frame_idx_list
        self.object_of_interest = object_of_interest

    def thresholded_detected_objects(self, threshold) -> dict:
        """Get all detected object."""
        detected_obj = {}
        for prop in self.object_of_interest.keys():
            probability = self.object_of_interest[prop].get_detected_probability()
            if probability > threshold:
                detected_obj[prop] = probability
        return detected_obj



