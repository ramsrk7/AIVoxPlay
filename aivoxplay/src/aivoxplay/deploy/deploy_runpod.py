# src/aivoxplay/deploy/deploy_runpod.py
import os
import requests
import runpod

RUNPOD_API = "https://rest.runpod.io/v1"
TOKEN = os.getenv("RUNPOD_API_KEY")
runpod.api_key = os.getenv("RUNPOD_API_KEY")

def _req(method: str, path: str, **kwargs):
    headers = kwargs.pop("headers", {})
    headers["Authorization"] = f"Bearer {TOKEN}"
    headers["Content-Type"] = "application/json"
    url = f"{RUNPOD_API}{path}"
    r = requests.request(method, url, headers=headers, **kwargs)
    r.raise_for_status()
    return r.json() if r.text else {}

DEFAULT_START_CMD = (
    "python -m vllm.entrypoints.openai.api_server "
    "--model unsloth/orpheus-3b-0.1-ft --dtype auto --host 0.0.0.0 --port 8000 "
    "--trust-remote-code --max-model-len 2048 --quantization fp8"
)


class DeployRunPod:
    def __init__(self, model: str, source: str = "unsloth"):
        self.model = model
        self.source = source
        self.template_id: str | None = None
        self.pod_id: str | None = None

    # ---------- OPTIONAL: GPU list (simple wrapper if you still need it) ----------
    def get_available_gpu_types(self):
        return runpod.get_gpus()  # adjust if endpoint differs

    # ---------- TEMPLATE ----------
    def create_template(
        self,
        name: str = "orpheus-3b-openai-api",
        image: str = "vllm/vllm-openai:latest",
        start_command: str = DEFAULT_START_CMD,
        ports: list[str] = None,
        volume_gb: int = 30,
        mount_path: str = "/workspace",
        env: dict | None = None,
        container_disk_gb: int = 50,
        serverless: bool = False,
        is_public: bool = False,
        category: str = "NVIDIA",
        readme: str = "",
    ) -> str:
        payload = {
            "category": category,
            "containerDiskInGb": container_disk_gb,
            "dockerEntrypoint": [],
            "dockerStartCmd": [
                "--model", "unsloth/orpheus-3b-0.1-ft",
                "--dtype", "auto",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--trust-remote-code",
                "--max-model-len", "2048",
                "--quantization", "fp8"
            ],
            "env": env or {},
            "imageName": image,
            "isPublic": is_public,
            "isServerless": serverless,
            "name": name,
            "ports": ports or ["8000/http"],
            "readme": readme,
            "volumeInGb": volume_gb,
            "volumeMountPath": mount_path,
        }
        resp = _req("POST", "/templates", json=payload)
        self.template_id = resp["id"]
        return self.template_id

    # ---------- POD (ENDPOINT) ----------
    def start_pod(
        self,
        name: str = "my pod",
        gpu_type: str = "NVIDIA H100 80GB HBM3",
        gpu_count: int = 1,
        cloud_type: str = "SECURE",
        compute_type: str = "GPU",
        volume_gb: int = 10,
        mount_path: str = "/workspace",
        ports: list[str] = None,
        env: dict | None = None,
        container_disk_gb: int = 20,
        interruptible: bool = False,
        template_id: str | None = None,
        vcpu_count: int | None = None,
        min_vcpu_per_gpu: int | None = None,
        min_ram_per_gpu: int | None = None,
        **extra,
    ) -> str:
        payload = {
            "name": name,
            "cloudType": cloud_type,
            "computeType": compute_type,
            "gpuCount": gpu_count,
            "gpuTypeIds": [gpu_type],
            "imageName": extra.pop("imageName", None),  # optional override
            "interruptible": interruptible,
            "templateId": template_id or self.template_id,
            "ports": ports or ["8000/http"],
            "env": env or {},
            "volumeInGb": volume_gb,
            "volumeMountPath": mount_path,
            "containerDiskInGb": container_disk_gb,
            # Optional CPU/Network tuning
            "vcpuCount": vcpu_count,
            "minVCPUPerGPU": min_vcpu_per_gpu,
            "minRAMPerGPU": min_ram_per_gpu,
        }
        # Remove None fields to keep payload tidy
        payload = {k: v for k, v in payload.items() if v is not None}

        resp = _req("POST", "/pods", json=payload)
        self.pod_id = resp["id"]
        return self.pod_id

    def stop_pod(self, pod_id: str | None = None):
        _req("POST", f"/pods/{pod_id or self.pod_id}/stop")

    def terminate_pod(self, pod_id: str | None = None):
        _req("DELETE", f"/pods/{pod_id or self.pod_id}")
