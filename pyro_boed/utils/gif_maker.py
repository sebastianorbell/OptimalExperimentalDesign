import glob
from PIL import Image
import os


def make_gif(name='ode', folder='/figs'):
    frame_folder = os.getcwd() + folder
    images = glob.glob(f"{frame_folder}/*.png")
    images.sort()
    frames = [Image.open(image) for image in images]
    frame_one = frames[0]
    frame_one.save(f"{os.getcwd()}/gifs/{name}.gif", format="GIF", append_images=frames,
                   save_all=True, duration=500, loop=0)
