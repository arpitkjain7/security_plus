from infer.inference import infer
from datetime import datetime

batch_id = str(int(datetime.now().timestamp() * 1000))
infer(
    "/Users/arpitkjain/Desktop/Data/POC/security_plus/test-data/abhilash-1.jpeg",
    batch_id=batch_id,
)
