class FramesofInterest:
    def __init__(self):
        self.foi_list = []
        self.frame_buffer = [] # Now stores (pixel_data, original_idx)
    def add_frame(self, frame_idx_list):
        self.frame_buffer.append(frame_idx_list)

    def flush_frame_buffer(self):
        """Flush the buffer by capturing the exact indices tracked."""
        if self.frame_buffer:
            for frame_obj in self.frame_buffer:
                if frame_obj.frame_idx_list:
                    self.foi_list.append(frame_obj.frame_idx_list)

            self.frame_buffer = []

    def compile_foi(self):
        flat_list = [i for sub in self.foi_list for i in sub]
        return sorted(list(set(flat_list)))