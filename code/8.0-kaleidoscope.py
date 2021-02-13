"""
Experimenting with kaleidoscope effects
2x2 grid
"""

from art import Net, create_input, generate_one_art
from sampler import LatentSpaceSampler
import torch
import torch.onnx
import cv2
import time
from timeit import default_timer as timer
import sys
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("frames", help="Total frames to run", type=int)
parser.add_argument(
    "--nodisplay",
    help="Hide the display for performance benchmark purposes",
    action="store_true",
)
parser.add_argument(
    "--XDIM", help="Dimension of X, defaults to 320", type=int, default=320
)
parser.add_argument(
    "--YDIM", help="Dimension of Y, defaults to 320", type=int, default=320
)
parser.add_argument(
    "--scale", help="Scale factor to resize the image by", type=float, default=None
)
args = parser.parse_args()
print(args)

if __name__ == "__main__":
    # Divide both XDIM and YDIM by 2
    XDIM = args.XDIM // 2
    YDIM = args.YDIM // 2

    lss = LatentSpaceSampler(min_coord=[-2, -2, -2], max_coord=[2, 2, 2], stepsize=0.05)
    net = Net(num_hidden_layers=2, num_neurons=64)
    # max_fps = 30 # no max FPS cap!

    if args.nodisplay is False:
        # Format output window
        # https://stackoverflow.com/a/49095781
        # Removes toolbar and status bar
        window_name = "Generative Art with neural networks"
        cv2.namedWindow(window_name, flags=cv2.WINDOW_GUI_NORMAL)

    # Main loop
    frame_count = 0
    while frame_count <= args.frames:

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) <= 0:
            # Was going to just be `while True`
            # But OpenCV windows don't stay closed when you close them in the GUI!
            # Explicitly checking that the window is closed to end the loop
            break

        out = generate_one_art(
            net,
            latent_vec=lss.sample(),
            input_config={"img_width": XDIM, "img_height": YDIM},
        )

        # Kaleidoscope!
        # `out` is top-left
        # Flip the array accordingly
        # axis=0 is rows, which are y-dim
        top_right = np.flip(out, axis=(1))
        bot_right = np.flip(out, axis=(0, 1))
        bot_left = np.flip(out, axis=(0))

        # Join them
        left_col = np.concatenate((out, bot_left), axis=0)
        right_col = np.concatenate((top_right, bot_right), axis=0)
        out = np.concatenate((left_col, right_col), axis=1)

        # Convert to BGR
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        # Upscale
        if args.scale is not None:
            out = cv2.resize(
                src=out,
                dsize=(0, 0),
                fx=args.scale,
                fy=args.scale,
                interpolation=cv2.INTER_LINEAR,
            )

        if args.nodisplay is False:
            # Display
            cv2.imshow(window_name, out)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_count % 10 == 0:
            if frame_count == 0:
                stopwatch = timer()
            else:
                duration = timer() - stopwatch
                print(f"Frame {frame_count}: Avg FPS is {10/duration:.2f}")
                stopwatch = timer()

        frame_count += 1

    else:
        if args.nodisplay is False:
            # After breaking while loop
            cv2.destroyAllWindows()
