import time, torch, numpy as np
from snac import SNAC

device = "mps"
model  = torch.compile(SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
                       .eval().to(device), mode="reduce-overhead")

# one dummy 28-token window
tokens = np.random.randint(0, 4096, size=28, dtype=np.int32).tolist()

from aivoxplay.tts.factory import OrpheusAudioProcessor

OAP = OrpheusAudioProcessor()
# warm-up
OAP.convert_to_audio(tokens, 28)

t0 = time.perf_counter()
for _ in range(100):
    OAP.convert_to_audio(tokens, 28)
t1 = time.perf_counter()

print(f"Avg latency: {(t1-t0)/100*1e3:.2f} ms")   # expect ~25–35 ms on M1
