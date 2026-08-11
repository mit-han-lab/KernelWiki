import torch
from task import TestSpec
from reference import ref_kernel, generate_input
from attempt12 import custom_kernel

torch.cuda.set_device(0)

def run_test(spec: TestSpec, warmup_rounds: int = 1):
    """
    Run a single test case based on the provided specification.

    Args:
        spec: A dictionary containing the test parameters (m, k, l, seed).
        warmup_rounds: Number of warmup iterations before timing.

    Returns:
        None
    """
    print(f"Running test with spec: {spec}")

    # Generate input tensors
    inputs = generate_input(spec["m"], spec["k"], spec["l"], spec["seed"])

    # Clone inputs for reference and custom kernels
    ref_inputs = tuple(tensor.clone() for tensor in inputs)
    custom_inputs = tuple(tensor.clone() for tensor in inputs)

    # Warmup rounds for reference kernel
    for _ in range(warmup_rounds):
        _ = ref_kernel(ref_inputs)  # type: ignore
    torch.cuda.synchronize()

    # Timing for reference kernel
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    ref_output = ref_kernel(ref_inputs)  # type: ignore
    end_event.record()
    torch.cuda.synchronize()
    ref_time_us = start_event.elapsed_time(end_event) * 1000  # Convert ms to us

    # Warmup rounds for custom kernel
    for _ in range(warmup_rounds):
        _ = custom_kernel(custom_inputs)  # type: ignore
    torch.cuda.synchronize()

    # Timing for custom kernel
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    custom_output = custom_kernel(custom_inputs)  # type: ignore
    end_event.record()
    torch.cuda.synchronize()
    custom_time_us = start_event.elapsed_time(end_event) * 1000  # Convert ms to us

    # Compare outputs
    if torch.allclose(ref_output, custom_output, atol=1e-3, rtol=1e-3):
        print("✓ Test passed!")
    else:
        print("✗ Test failed: Outputs do not match.")
        # print("Reference output:", ref_output)
        # print("Custom output:", custom_output)

    # Report performance
    #print(f"Reference kernel time: {ref_time_us:.2f} us")
    print(f"Custom kernel time: {custom_time_us:.2f} us")
    print("---")


def main():
    """
    Main function to run all tests.
    """
    # test_specs = [
    #     {"m": 128, "k": 256, "l": 1, "seed": 1111},
    #     {"m": 128, "k": 1536, "l": 1, "seed": 1111},
    #     {"m": 128, "k": 3072, "l": 1, "seed": 1111},
    #     {"m": 256, "k": 7168, "l": 1, "seed": 1111},
    #     {"m": 2432, "k": 4608, "l": 2, "seed": 1111},
    #     {"m": 384, "k": 7168, "l": 2, "seed": 1111},
    #     {"m": 512, "k": 512, "l": 2, "seed": 1111},
    #     {"m": 512, "k": 4096, "l": 2, "seed": 1111},
    #     {"m": 512, "k": 1536, "l": 2, "seed": 1111},
    # ]

    test_specs = [
        {"m": 7168, "k": 16384, "l": 1, "seed": 1111},
        {"m": 4096, "k": 7168, "l": 8, "seed": 1111},
        {"m": 7168, "k": 2048, "l": 4, "seed": 1111},
        {"m": 4608, "k": 2432, "l": 2, "seed": 1111},
        {"m": 7168, "k": 384, "l": 2, "seed": 1111}
    ]

    for spec in test_specs:
        run_test(spec)  # type: ignore


if __name__ == "__main__":
    main()
