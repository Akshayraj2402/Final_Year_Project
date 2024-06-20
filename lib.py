from IPython.display import HTML
import PIL.Image
from base64 import b64encode
import seaborn as sns
import ffmpeg
import numpy as np
import os
import imageio
import cv2
def show_video(video_path, video_width = "fill"):
  """
  video_path (str): The path to the video
  video_width: Width for the window the video will be shown in 
  """
  video_file = open(video_path, "r+b").read()

  video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
  return HTML(f"""<video width={video_width} controls><source src="{video_url}"></video>""")


def create_video(frames_pattern='Track/%03d.png', video_file='movie.mp4', framerate=25):
    """
    frames_pattern (str): The pattern to use to find the frames. The default pattern looks for frames in a folder called Track. The frames should be named 001.png, 002.png, ..., 999.png
    video_file (str): The file the video will be saved in
    framerate (float): The framerate for the video
    """
    # Assuming frames are numbered from 1 to 999
    frame_files = [frames_pattern % i for i in range(1, 60) if os.path.exists(frames_pattern % i)]

    # Check if there are frames before proceeding
    if not frame_files:
        print("No frames found. Cannot create the video.")
        return
    
    # Read the first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape

    # Initialize video writer
    video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), framerate, (width, height))

    # Write frames to video
    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        video_writer.write(frame)

    # Release video writer
    video_writer.release()

'''
def create_video(frames_patten='Track/%03d.png', video_file = 'movie.mp4', framerate=25):
  """
  frames_patten (str): The patten to use to find the frames. The default patten looks for frames in a folder called Track. The frames shoud be named 001.png, 002.png, ..., 999.png
  video_file (str): The file the video will be saved in 
  framerate (float): The framerate for the video
  """
  ffmpeg_executable = r'D:\Python\Projects\Ak\ffmpeg-6.1.1-essentials_build\bin\ffmpeg.exe'
  if os.path.exists(video_file):
      os.remove(video_file)
  ffmpeg.input(frames_patten, framerate=framerate).output(video_file, ffmpeg_path=ffmpeg_executable).run()
  #ffmpeg.input(frames_patten, framerate=framerate).output(video_file).run() 
'''
def textsize(text):
    im = PIL.Image.new(mode="P", size=(0, 0))
    draw = PIL.ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text)
    return width, height

class VisTrack:
    def __init__(self, unique_colors=400):
        """
        unique_colors (int): The number of unique colors (the number of unique colors dos not need to be greater than the max id)
        """
        self._unique_colors = unique_colors
        self._id_dict = {}
        self.p = np.zeros(unique_colors)
        self._colors = (np.array(sns.color_palette("hls", unique_colors))*255).astype(np.uint8)

    def _get_color(self, i):
        return tuple(self._colors[i])

    def _color(self, i):
        if i not in self._id_dict:
            inp = (self.p.max() - self.p ) + 1 
            if any(self.p == 0):
                nzidx = np.where(self.p != 0)[0]
                inp[nzidx] = 0
            soft_inp = inp / inp.sum()

            ic = np.random.choice(np.arange(self._unique_colors, dtype=int), p=soft_inp)
            self._id_dict[i] = ic

            self.p[ic] += 1

        ic = self._id_dict[i]
        return self._get_color(ic)

    def draw_bounding_boxes(self, im: PIL.Image, bboxes: np.ndarray, ids: np.ndarray,
                        scores: np.ndarray) -> PIL.Image:
        """
        im (PIL.Image): The image 
        bboxes (np.ndarray): The bounding boxes. [[x1,y1,x2,y2],...]
        ids (np.ndarray): The id's for the bounding boxes
        scores (np.ndarray): The scores's for the bounding boxes
        """
        im = im.copy()
        draw = PIL.ImageDraw.Draw(im)

        for bbox, id_, score in zip(bboxes, ids, scores):
            color = self._color(id_)
            draw.rectangle((*bbox.astype(np.int64),), outline=color)

            text = f'{id_}: {int(100 * score)}%'
            text_w, text_h = textsize(text)
            draw.rectangle((bbox[0], bbox[1], bbox[0] + text_w, bbox[1] + text_h), fill=color, outline=color)
            draw.text((bbox[0], bbox[1]), text, fill=(0, 0, 0))

        return im
