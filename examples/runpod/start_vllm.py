from dotenv import load_dotenv
load_dotenv()

import sys
import time
import atexit
from aivoxplay.deploy.deploy_runpod import DeployRunPod, DEFAULT_START_CMD
import os
rp = DeployRunPod(model="unsloth/orpheus-3b-0.1-ft")
HF_TOKEN=os.getenv("HF_TOKEN")

def cleanup():
    """Ensure pod is stopped and deleted on exit."""
    try:
        if rp.pod_id:
            # print(">> Stopping pod…")
            # rp.stop_pod()
            # print(">> Terminating pod…")
            # rp.terminate_pod()
            pass
    except Exception as e:
        print("Cleanup error:", e)

atexit.register(cleanup)


def main():
    # 1. Optional: list GPUs
    try:
        print("Available GPUs:", rp.get_available_gpu_types())
    except Exception as e:
        print("GPU list failed:", e)

    # 2. Create template
    tpl_id = rp.create_template(
        name="orpheus-template-h100-4",
        image="vllm/vllm-openai:latest",
        start_command=DEFAULT_START_CMD,
        ports=["8000/http"],
        volume_gb=30,
        mount_path="/root/.cache/huggingface",
        env={
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_TOKEN": HF_TOKEN
        },
        serverless=False,
    )
    
    #tpl_id = "wrce975nbo"
    print("Template ID:", tpl_id)

    # 3. Start pod
    pod_id = rp.start_pod(
        name="orpheus-pod-2",
        gpu_type="NVIDIA A100 80GB PCIe",
        gpu_count=1,
        cloud_type="SECURE",
        volume_gb=30,
        ports=["8000/http"])
    print("Pod ID:", pod_id)

    # 4. Simulate workload
    print("Pod running... Sleeping 5 seconds.")
    time.sleep(300)

    print("Python executable:", sys.executable)
    print("Python version:", sys.version)

if __name__ == "__main__":
    try:
        main()
    finally:
        pass  # atexit handles cleanup
