"""
Script to benchmark runtime over different configurations. Expect frequent changes.
"""

from art import Net, create_input
from sampler import LatentSpaceSampler
import torch
import torch.onnx
import cv2
import time
from timeit import default_timer as timer
import sys
import argparse
import numpy as np


def generate_one_art(
    net: Net,
    latent_vec: torch.Tensor,
    input_config={"img_width": 320, "img_height": 320},
    dtype: torch.dtype = torch.float32,
):
    """
    Wrapper function to generate a single image output from the given network.

    Input
    -----
    net: Net
    latent_vec: torch.Tensor
    input_config: dict
        Dict of parameters to be passed to `create_input` as kwargs.

    Output
    ------
    net_output: np.ndarray
        Should have shape (y, x, 3) or (y, x, 1)
    """
    # Create input to net, and convert from ndarray to torch.FloatTensor
    net_input = torch.tensor(create_input(**input_config)).to(dtype)

    # Create input array from latent_vec, and convert from ndarray to torch.FloatTensor
    latent_vec = np.expand_dims(latent_vec, axis=0)
    latent_vec = np.repeat(latent_vec, repeats=net_input.shape[0], axis=0)
    latent_vec = torch.tensor(latent_vec).to(dtype)

    assert net_input.shape == latent_vec.shape, (
        "Shape of net_input is "
        f"{net_input.shape} while shape of latent_vec is {latent_vec.shape}"
    )

    # Run input through net
    net_output = net(net_input, latent_vec).detach().numpy()

    # Reshape into (y, x, 3) for plotting in PIL
    net_output = net_output.reshape(
        input_config["img_height"], input_config["img_width"], -1
    )

    # Re-format to color output
    # Scale to range 0 to 255, and set type to int
    net_output = (net_output * 255).astype(np.uint8)
    return net_output


parser = argparse.ArgumentParser()
parser.add_argument("--frames", help="Total frames to run", type=int)
parser.add_argument(
    "--nodisplay",
    help="Hide the display for performance benchmark purposes",
    action="store_true",
)
parser.add_argument(
    "--usetorchscript",
    help="Compile to Torchscript before running",
    action="store_true",
)
parser.add_argument(
    "--useonnx", help="Compile to ONNX before running", action="store_true"
)
parser.add_argument("--dtype", help="torch.dtype to use", type=str)
args = parser.parse_args()

torch_dtypes = {
    "torch.float64": torch.float64,
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
    "torch.int8": torch.int8,
}


if __name__ == "__main__":

    lss = LatentSpaceSampler(min_coord=[-2, -2, -2], max_coord=[2, 2, 2], stepsize=0.05)
    net = Net(num_hidden_layers=2, num_neurons=64)
    net.to(torch_dtypes.get(args.dtype))

    if args.usetorchscript is True:
        net = torch.jit.script(net)

    elif args.useonnx is True:

        import onnx
        import onnxruntime
        from art import create_input
        import numpy as np

        # Compile to ONNX
        arr_in = torch.tensor(create_input(640, 640)).float()

        latent_vec = np.random.normal(size=(3,))
        latent_vec = np.expand_dims(latent_vec, axis=0)
        latent_vec = np.repeat(latent_vec, repeats=arr_in.shape[0], axis=0)
        latent_vec = torch.tensor(latent_vec).float()

        torch.onnx.export(
            net,
            (arr_in, latent_vec),
            "exported-model.onnx",
            export_params=True,
            opset_version=10,
            do_constant_folding=True,
            input_names=["spatial-input", "latent-vec"],
            output_names=["output-rgb"],
        )

        # Verify ONNX is valid
        onnx_model = onnx.load("exported-model.onnx")
        onnx.checker.check_model(onnx_model)

        # Then load ONNX
        ort_session = onnxruntime.InferenceSession("exported-model.onnx")

    # max_fps = 30 # no max FPS cap!

    if args.nodisplay is False:
        # Format output window
        # https://stackoverflow.com/a/49095781
        # Removes toolbar and status bar
        window_name = "Generative Art with neural networks"
        cv2.namedWindow(window_name, flags=cv2.WINDOW_GUI_NORMAL)

    if args.nodisplay is True:
        frame_count = 0
        while frame_count <= args.frames:

            if args.useonnx is True:
                # compute ONNX Runtime output prediction
                arr_in = create_input(640, 640)
                latent_vec = lss.sample()
                latent_vec = np.expand_dims(latent_vec, axis=0)
                latent_vec = np.repeat(latent_vec, repeats=arr_in.shape[0], axis=0)

                ort_inputs = {
                    "spatial-input": arr_in.astype(np.float32),
                    "latent-vec": latent_vec.astype(np.float32),
                }
                ort_outs = ort_session.run(None, ort_inputs)
                out = ort_outs[0]
            else:
                out = generate_one_art(
                    net,
                    latent_vec=lss.sample(),
                    input_config={"img_width": 640, "img_height": 640},
                    dtype=torch_dtypes.get(args.dtype),
                )

            # Convert to BGR
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

            if frame_count % 10 == 0:
                if frame_count == 0:
                    stopwatch = timer()
                else:
                    duration = timer() - stopwatch
                    print(f"Frame {frame_count}: Avg FPS is {10/duration:.2f}")
                    stopwatch = timer()

            frame_count += 1

    else:
        frame_count = 0
        while (
            cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) > 0
        ) and frame_count <= args.frames:
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
            # time.sleep(1 / max_fps)

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
        else:
            # After breaking while loop
            cv2.destroyAllWindows()
