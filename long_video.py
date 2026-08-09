#!/usr/bin/env python3
"""Create a combined timelapse video from plant and weather station images.

Usage:
    python long_video.py <days_back>
The script collects all JPG images from the last ``days_back`` days in
both ``/media/bigdata/plant_station/images`` and
``/media/bigdata/weather_station/images``. Images are merged chronologically
and written to a temporary list for ffmpeg so memory use stays modest even for
large ranges. The output video encoder is chosen based on available ffmpeg
support.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import tempfile
import shutil
import heapq

PLANT_IMAGE_DIR = Path('/media/bigdata/plant_station/images')
WEATHER_IMAGE_DIR = Path('/media/bigdata/weather_station/images')

FRAMERATE = 30  # frames per second for the output video


def gather_images(days_back):
    """Yield image paths from the last ``days_back`` days sorted by timestamp."""
    cutoff_ts = (datetime.now() - timedelta(days=days_back)).timestamp()
    sources = []
    for directory in (PLANT_IMAGE_DIR, WEATHER_IMAGE_DIR):
        if directory.is_dir():
            imgs = [p for p in directory.glob('*.jpg') if p.stat().st_mtime >= cutoff_ts]
            imgs.sort(key=lambda p: p.stat().st_mtime)
            sources.append(imgs)
    return heapq.merge(*sources, key=lambda p: p.stat().st_mtime)


def choose_encoder():
    """Return a supported video encoder for ffmpeg."""
    if not shutil.which('ffmpeg'):
        return None
    try:
        result = subprocess.run(
            ['ffmpeg', '-v', '0', '-hide_banner', '-encoders'],
            capture_output=True, text=True, check=True
        )
        encoders = result.stdout
    except subprocess.SubprocessError:
        return 'mpeg4'
    for cand in ('libx264', 'libxvid', 'mpeg4'):
        if cand in encoders:
            return cand
    return 'mpeg4'


def build_video(image_iter, output_path):
    if not shutil.which('ffmpeg'):
        print('ffmpeg is not installed or not found in PATH.')
        return
    with tempfile.NamedTemporaryFile('w', delete=False) as list_file:
        count = 0
        for img in image_iter:
            list_file.write(f"file '{img}'\n")
            count += 1
        list_path = list_file.name
    if count == 0:
        print('No images found for the specified period.')
        return
    encoder = choose_encoder() or 'mpeg4'
    cmd = [
        'ffmpeg', '-y', '-r', str(FRAMERATE),
        '-f', 'concat', '-safe', '0', '-i', list_path,
        '-c:v', encoder, '-pix_fmt', 'yuv420p', str(output_path)
    ]
    subprocess.run(cmd, check=True)
    print(f'Created {output_path} with {count} images.')


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


if __name__ == '__main__':
    main()
