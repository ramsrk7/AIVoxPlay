from dotenv import load_dotenv
load_dotenv()
from aivoxplay.deploy.deploy_runpod import DeployRunPod

runpod = DeployRunPod(model="meta-llama/Llama-3-8b")


# Create template with selected GPU
gpus = runpod.get_available_gpu_types()

print(f"Available GPUs - {gpus}")
# Launch and check
# pod_id = runpod.launch_pod()
# status = runpod.get_pod_status()

import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)