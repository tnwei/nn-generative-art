"""
Run `python cli.py` to generate abstract art using neural networks on the fly.
"""

from art import Net, generate_one_art, create_input
from sampler import LatentSpaceSampler, BezierSampler
import cv2
import time
import numpy as np
from timeit import default_timer as timer
import sys
from PIL import Image
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--XDIM", help="Dimension of X, defaults to 160", type=int, default=160
)
parser.add_argument(
    "--YDIM", help="Dimension of Y, defaults to 90", type=int, default=90
)
parser.add_argument(
    "--sampler",
    help="Choose 'random' sampler or 'bezier' sampler to explore the latent space",
    type=str,
    default="random",
)
parser.add_argument(
    "--scale",
    help="Scale factor to resize the image, defaults to 4",
    type=float,
    default=4,
)
parser.add_argument(
    "--noframelimit",
    help="If passed, lifts cap on max frames per second (24fps), not recommended",
    action="store_true",
)
parser.add_argument(
    "--kaleidoscope",
    help="If passed, generates top-left quadrant of the image and mirrors across x-y axes for a kaleidoscope effect. Note: Doubles width and height before scale factor is applied",
    action="store_true",
)
parser.add_argument(
    "--include_R",
    help="TODO: Reword: Incorporates minimum distance to each pixel in a reference array",
    action="store_true",
)

args = parser.parse_args()
assert args.sampler in ["random", "bezier"], f"--sampler received {args.sampler}"

if __name__ == "__main__":
    XDIM, YDIM = args.XDIM, args.YDIM

    # Setup latent space sampler
    if args.sampler == "random":
        lss = LatentSpaceSampler(
            min_coord=[-2, -2, -2], max_coord=[2, 2, 2], stepsize=0.05
        )
        latent_len = 3
    elif args.sampler == "bezier":
        lss = BezierSampler(num_ctrl_points=5, steps=1000)
        latent_len = 2
    else:
        raise ValueError

    # Incorporate min dist R
    if args.include_R:

        def im_png_to_arr(im):
            # ref: https://stackoverflow.com/a/33507138/13095028
            background = Image.new("RGBA", im.size, (255, 255, 255))
            alpha_composite = Image.alpha_composite(background, im)
            return 1 - np.array(alpha_composite.convert("L")) / 255

        R = im_png_to_arr(Image.open("160x90-smiley.png"))
        include_R = True
    else:
        R = None
        include_R = False

    # Setup net input in advance
    # This actually only needs to be calculated once!
    if args.kaleidoscope is True:
        input_config = {
            "img_width": XDIM,
            "img_height": YDIM,
            "xs_start": -1,
            "xs_stop": 0,
            "ys_start": -1,
            "ys_stop": 0,
            "R": R,
        }
    else:
        input_config = {"img_width": XDIM, "img_height": YDIM, "R": R}

    net_input = create_input(**input_config)

    print(f"net_input.shape: {net_input.shape}")

    net = Net(
        num_hidden_layers=2,
        num_neurons=64,
        latent_len=latent_len,
        include_R=args.include_R,
    )

    max_fps = 24
    max_frame_duration = 1 / max_fps

    if args.noframelimit is False:
        print(f"Frame limit set at {max_fps} FPS. Use --noframelimit to disable")

    # Format output window
    # https://stackoverflow.com/a/49095781
    # Removes toolbar and status bar
    window_name = "Generative Art with neural networks"
    cv2.namedWindow(window_name, flags=cv2.WINDOW_GUI_NORMAL)

    frame_count = 0

    try:
        while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) > 0:
            # Was going to just be `while True`
            # But OpenCV windows don't stay closed when you close them in the GUI!
            # Explicitly checking that the window is closed to end the loop
            frame_start = timer()

            if args.kaleidoscope is True:
                out = generate_one_art(
                    net,
                    latent_vec=lss.sample(),
                    net_input=net_input,
                    input_config=input_config,
                )
                # Kaleidoscope ------------------------------
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

                # End kaleidoscope --------------------------

            else:
                out = generate_one_art(
                    net,
                    latent_vec=lss.sample(),
                    net_input=net_input,
                    input_config=input_config,
                )

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

            # Cap FPS to 24
            frame_end = timer()

            if args.noframelimit is False:
                time_passed = frame_end - frame_start
                if time_passed < max_frame_duration:
                    time.sleep(max_frame_duration - time_passed)

        # After breaking while loop
        cv2.destroyAllWindows()

    except KeyboardInterrupt:
        # Exit if Ctrl-C in terminal
        cv2.destroyAllWindows()
        sys.exit()
