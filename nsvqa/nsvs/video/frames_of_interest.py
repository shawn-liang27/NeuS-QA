# class FramesofInterest:
#     def __init__(self, num_of_frame_in_sequence, frame_step):
#         self.num_of_frame_in_sequence = num_of_frame_in_sequence
#         self.frame_step = frame_step
#         self.foi_list = []
#         self.frame_buffer = []

#     def flush_frame_buffer(self):
#         """Flush frame buffer to frame of interest."""
#         if self.frame_buffer:
#             frame_interval = [frame.frame_idx for frame in self.frame_buffer]
#             total_step = self.num_of_frame_in_sequence * self.frame_step
#             self.foi_list.append([
#                 i*total_step + j 
#                 for i in frame_interval 
#                 for j in range(total_step)
#             ])
#             self.frame_buffer = []
#     def compile_foi(self):
#         return [i for sub in self.foi_list for i in sub]

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
        # Use a set to automatically remove duplicates, then sort
        flat_list = [i for sub in self.foi_list for i in sub]
        return sorted(list(set(flat_list)))