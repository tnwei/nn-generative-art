"""
Run `python cli.py` to generate abstract art using neural networks on the fly.
"""

from code.art import Net, generate_one_art
from code.sampler import LatentSpaceSampler
import cv2
import time
from timeit import default_timer as timer
import sys

if __name__ == "__main__":

    lss = LatentSpaceSampler(min_coord=[-2, -2, -2], max_coord=[2, 2, 2], stepsize=0.05)
    net = Net(num_hidden_layers=2, num_neurons=64)
    max_fps = 30

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

            out = generate_one_art(
                net,
                latent_vec=lss.sample(),
                input_config={"img_width": 640, "img_height": 640},
            )

            # Already averages < 1.5 FPS on my laptop
            # But including an FPS cap here for fast machines
            time.sleep(1/max_fps)

            # Convert to BGR
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
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

        # After breaking while loop
        cv2.destroyAllWindows()

    except KeyboardInterrupt:
        # Exit if Ctrl-C in terminal
        cv2.destroyAllWindows()
        sys.exit()
