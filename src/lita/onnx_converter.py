import subprocess
from typing import Optional
import tempfile

from optimum.onnxruntime import ORTOptimizer, ORTModelForCausalLM
from optimum.onnxruntime.configuration import OptimizationConfig

def optimum_export(
    model: str,
    output: str,
    task: Optional[str] = None,
    monolith: bool = False,
    device: str = "cpu",
    opset: Optional[int] = None,
    atol: Optional[float] = None,
    framework: Optional[str] = None,
    pad_token_id: Optional[int] = None,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = False,
    no_post_process: bool = False,
    batch_size: Optional[int] = None,
    sequence_length: Optional[int] = None,
    num_choices: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    num_channels: Optional[int] = None,
    feature_size: Optional[int] = None,
    nb_max_frames: Optional[int] = None,
    audio_sequence_length: Optional[int] = None,
    
    optimize: Optional[int] = None,
    fp16: Optional[bool] = False
):
    """
    Convert a model to ONNX format using optimum-cli with all available options.

    Args:
    - model (str): Model ID on huggingface.co or path on disk to load model from.
    - output (str): Path indicating the directory where to store generated ONNX model.
    - task (Optional[str]): The task to export the model for.
    - monolith (bool): Whether to export as a single ONNX file.
    - device (str): The device to use for the export.
    - opset (Optional[int]): ONNX opset version to use.
    - atol (Optional[float]): Absolute difference tolerance for validation.
    - framework (Optional[str]): Framework to use for ONNX export.
    - pad_token_id (Optional[int]): Padding token ID for the model.
    - cache_dir (Optional[str]): Cache directory for models.
    - trust_remote_code (bool): Trust remote code for custom model repositories.
    - no_post_process (bool): Disable post-processing of exported ONNX models.
    - optimize (Optional[str]): ONNX Runtime optimization level (O1, O2, O3, O4).
    - batch_size, sequence_length, num_choices, width, height, num_channels, feature_size, nb_max_frames, audio_sequence_length: Model input dimensions.
    """
    
    # Construct the CLI command
    command = [
        "optimum-cli", "export", "onnx",
        "-m", model,
        output
    ]
    
    # Add optional arguments
    if task:
        command.extend(["--task", task])
    if monolith:
        command.append("--monolith")
    if device:
        command.extend(["--device", device])
    if opset:
        command.extend(["--opset", str(opset)])
    if atol:
        command.extend(["--atol", str(atol)])
    if framework:
        command.extend(["--framework", framework])
    if pad_token_id is not None:
        command.extend(["--pad_token_id", str(pad_token_id)])
    if cache_dir:
        command.extend(["--cache_dir", cache_dir])
    if trust_remote_code:
        command.append("--trust-remote-code")
    if no_post_process:
        command.append("--no-post-process")
    if batch_size:
        command.extend(["--batch_size", str(batch_size)])
    if sequence_length:
        command.extend(["--sequence_length", str(sequence_length)])
    if num_choices:
        command.extend(["--num_choices", str(num_choices)])
    if width:
        command.extend(["--width", str(width)])
    if height:
        command.extend(["--height", str(height)])
    if num_channels:
        command.extend(["--num_channels", str(num_channels)])
    if feature_size:
        command.extend(["--feature_size", str(feature_size)])
    if nb_max_frames:
        command.extend(["--nb_max_frames", str(nb_max_frames)])
    if audio_sequence_length:
        command.extend(["--audio_sequence_length", str(audio_sequence_length)])
            
    # Execute the CLI command using subprocess
    try:
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            subprocess.run(command, check=True, text=True, stdout=temp_file)
        print("ONNX model conversion successful!")
        
    except subprocess.CalledProcessError as e:
        print("Error occurred during ONNX model conversion:")
        print(e.stderr)

    if optimize or fp16:
        try:
            print("Starting ONNX model optimization...")
            
            # load exported onnx model
            model = ORTModelForCausalLM.from_pretrained(output)
            
            # initialize the optimum optimizer
            optimizer = ORTOptimizer.from_pretrained(model)
            optimization_config = OptimizationConfig(
                                                        optimization_level=int(optimize),
                                                        fp16=fp16
                                                    )
            
            # optimize the onnx model and save
            optimizer.optimize(
                save_dir=output + "-optimized",
                optimization_config=optimization_config,
            )
            
            print("ONNX model optimization successful!")
            
        except Exception as opt_error:
            print("Error occurred during ONNX model optimization:")
            print(opt_error)
