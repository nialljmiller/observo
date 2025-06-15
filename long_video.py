#!/usr/bin/env python3
"""Create a combined timelapse video from plant and weather station images.

Usage:
    python long_video.py <days_back>
The script collects all JPG images from the last `days_back` days in
both /media/bigdata/plant_station/images and
/media/bigdata/weather_station/images, sorts them chronologically and
produces an MP4 video.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import tempfile

PLANT_IMAGE_DIR = Path('/media/bigdata/plant_station/images')
WEATHER_IMAGE_DIR = Path('/media/bigdata/weather_station/images')

FRAMERATE = 30  # frames per second for the output video


def gather_images(days_back):
    """Return a sorted list of image paths from the last `days_back` days."""
    cutoff = datetime.now() - timedelta(days=days_back)
    images = []
    for directory in [PLANT_IMAGE_DIR, WEATHER_IMAGE_DIR]:
        if directory.is_dir():
            for path in directory.glob('*.jpg'):
                if datetime.fromtimestamp(path.stat().st_mtime) >= cutoff:
                    images.append(path)
    images.sort(key=lambda p: p.stat().st_mtime)
    return images


def build_video(image_paths, output_path):
    if not image_paths:
        print('No images found for the specified period.')
        return
    with tempfile.NamedTemporaryFile('w', delete=False) as list_file:
        for img in image_paths:
            list_file.write(f"file '{img}'\n")
        list_path = list_file.name
    cmd = [
        'ffmpeg', '-y', '-r', str(FRAMERATE),
        '-f', 'concat', '-safe', '0', '-i', list_path,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(output_path)
    ]
    subprocess.run(cmd, check=True)


def main():
    if len(sys.argv) != 2:
        print('Usage: python long_video.py <days_back>')
        sys.exit(1)
    try:
        days_back = int(sys.argv[1])
    except ValueError:
        print('days_back must be an integer')
        sys.exit(1)

    images = gather_images(days_back)
    start = (datetime.now() - timedelta(days=days_back)).strftime('%Y%m%d')
    end = datetime.now().strftime('%Y%m%d')
    output_name = f'combined_{start}_to_{end}.mp4'
    build_video(images, output_name)
    print(f'Created {output_name} with {len(images)} images.')


if __name__ == '__main__':
    main()
