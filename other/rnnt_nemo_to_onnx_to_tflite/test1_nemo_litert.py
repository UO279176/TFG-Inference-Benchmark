import torch
import litert_torch
from nemo.collections.asr.models import EncDecRNNTBPEModel

# Cargar modelo
nemo_path = "/mnt/d/Users/Saulete/Documents/TFG-Inference-Benchmark/src/data/models/rnnt/parakeet-rnnt-1.1b.nemo"
device = torch.device("cpu")
asr_model = EncDecRNNTBPEModel.restore_from(restore_path=nemo_path, map_location=device)
asr_model.eval()
asr_model._prepare_for_export()

# Aislar encoder
nemo_encoder = asr_model.encoder
nemo_encoder.to(device)
nemo_encoder.eval()

# Wrapper para el encoder de NeMo
class NeMoEncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, audio_signal):
        with torch.no_grad():
            encoded, _ = self.encoder(audio_signal=audio_signal, length=None)
        return encoded

# Instanciar el wrapper
wrapper_encoder = NeMoEncoderWrapper(nemo_encoder)
wrapper_encoder.to(device)
wrapper_encoder.eval()

# Inputs
batch_size = 1
mel_features = 80
time_steps = 128

sample_audio_signal = torch.randn(batch_size, mel_features, time_steps, device=device)
# sample_lengths = torch.tensor([time_steps], dtype=torch.int32, device=device)

sample_inputs = (sample_audio_signal,)

# Conversión
try:
    with torch.no_grad():
        edge_encoder = litert_torch.convert(wrapper_encoder, sample_inputs)
    edge_encoder.export("rnnt_encoder.tflite")
    print("Model converted and exported successfully.")
except Exception as e:
    print(f"Error during conversion: {e}")
    