from code.art import Net, create_input
import numpy as np
import torch
import torch.onnx
import onnx
import onnxruntime

# Keep track of all parameters here
latent_len = 3
XDIM = 160
YDIM = 90
input_config = {
    "img_width": XDIM,
    "img_height": YDIM,
    "xs_start": -1,
    "xs_stop": 0,
    "ys_start": -1,
    "ys_stop": 0,
}

# Init net
net = Net(num_hidden_layers=2, num_neurons=64, latent_len=latent_len)

# Create input examples for torch tracing for compilation to ONNX
net_input = torch.FloatTensor(create_input(**input_config))
latent_vec = np.random.normal(size=(latent_len,))
latent_vec = np.expand_dims(latent_vec, axis=0)
latent_vec = np.repeat(latent_vec, repeats=net_input.shape[0], axis=0)
latent_vec = torch.FloatTensor(latent_vec)

torch.onnx.export(
    net,
    (net_input, latent_vec),
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

# Verify that the input used is valid
ort_session = onnxruntime.InferenceSession("exported-model.onnx")

ort_inputs = {
    "spatial-input": net_input.numpy().astype(np.float32),
    "latent-vec": latent_vec.numpy().astype(np.float32),
}

ort_outs = ort_session.run(None, ort_inputs)
out = ort_outs[0]
