"""
RTSP Stream Simulator

Utility for turning a local video file (e.g. MP4) into a live RTSP stream
using ffmpeg. This allows you to develop and test the edge controller
against a "camera-like" stream before real cameras are available.

IMPORTANT: This requires an RTSP server to be running. For development, you can:

Option 1 - Use rtsp-simple-server (recommended):
    1. Install: winget install aler9.rtsp-simple-server
    2. Start it (runs on port 8554 by default)
    3. Run this script with --rtsp-url rtsp://127.0.0.1:8554/test

Option 2 - Use file directly (simplest):
    Just set CAMERA_SOURCES=camera_0:C:\path\to\video.mp4
    No streaming needed - your camera manager already supports files!

Example usage:

    python -m services.rtsp_simulator ^
        --input test_video.mp4 ^
        --rtsp-url rtsp://127.0.0.1:8554/test

Then, in your `.env`:

    CAMERA_SOURCES=camera_0:rtsp://127.0.0.1:8554/test

The edge controller will treat this exactly like an IP camera.
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
from typing import List

logger = logging.getLogger(__name__)




def build_ffmpeg_command_rtsp(
    input_path: str,
    rtsp_url: str,
    loop: bool = True,
) -> List[str]:
    """
    Build the ffmpeg command to stream a file as RTSP.

    This follows the options you described:

        ffmpeg -re -i "<input>" -c:v libx264 -preset veryfast -tune zerolatency
               -g 30 -bf 0 -c:a aac -f rtsp <rtsp_url>

    Note: This requires an RTSP server (like rtsp-simple-server) to be running.
    We additionally add -stream_loop -1 (optional) so the video repeats forever.
    """
    cmd: List[str] = ["ffmpeg", "-re"]

    # Loop forever so the "camera" is always on during dev, unless disabled.
    if loop:
        cmd += ["-stream_loop", "-1"]

    cmd += [
        "-i",
        input_path,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-tune",
        "zerolatency",
        "-g",
        "30",
        "-bf",
        "0",
        "-c:a",
        "aac",
        "-f",
        "rtsp",
        rtsp_url,
    ]

    return cmd


def run_ffmpeg(
    input_path: str,
    rtsp_url: str,
    loop: bool = True,
) -> int:
    """
    Run ffmpeg as a subprocess and block until it exits.

    Returns the process' exit code.
    """
    if not os.path.isfile(input_path):
        logger.error(f"Input file does not exist: {input_path}")
        return 1

    if not rtsp_url:
        logger.error("--rtsp-url must be specified")
        logger.info("Note: For simple testing, you can use the file directly:")
        logger.info("  Set CAMERA_SOURCES=camera_0:%s", input_path)
        logger.info("  No streaming needed!")
        return 1

    cmd = build_ffmpeg_command_rtsp(input_path, rtsp_url, loop=loop)
    logger.info("Starting ffmpeg RTSP simulator")
    logger.warning("IMPORTANT: Make sure an RTSP server is running!")
    logger.warning("  Install: winget install aler9.rtsp-simple-server")
    logger.warning("  Then start it before running this script")
    logger.info("Command: %s", " ".join(cmd))

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError:
        logger.error(
            "ffmpeg executable not found. Please install ffmpeg and ensure it is "
            "available on your PATH."
        )
        return 1

    # Handle Ctrl+C gracefully and forward to ffmpeg
    def handle_sigint(signum, frame):  # type: ignore[override]
        logger.info("Received interrupt, stopping ffmpeg...")
        if proc.poll() is None:
            proc.terminate()

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    # Stream ffmpeg logs to our logger
    assert proc.stdout is not None
    for line in proc.stdout:
        logger.info("[ffmpeg] %s", line.rstrip())

    proc.wait()
    logger.info("ffmpeg exited with code %s", proc.returncode)
    return int(proc.returncode or 0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate a camera by streaming a local video file over RTSP using ffmpeg. "
            "Requires an RTSP server (like rtsp-simple-server) to be running.\n\n"
            "For simple testing without RTSP, just use the file directly:\n"
            "  CAMERA_SOURCES=camera_0:/path/to/video.mp4"
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input video file (e.g. MP4).",
    )
    parser.add_argument(
        "--rtsp-url",
        required=True,
        help="RTSP URL to publish to, e.g. rtsp://127.0.0.1:8554/test (requires RTSP server)",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Do not loop the video; stop when the file ends.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_args(argv)
    loop = not args.no_loop
    return run_ffmpeg(args.input, args.rtsp_url, loop=loop)


if __name__ == "__main__":
    raise SystemExit(main())



