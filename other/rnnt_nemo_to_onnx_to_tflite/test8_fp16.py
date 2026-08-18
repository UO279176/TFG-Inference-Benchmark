import torch
from nemo.collections.asr.models import EncDecRNNTBPEModel

# 1. Cargar modelo
nemo_path = "D:/Users/Saulete/Documents/TFG-Inference-Benchmark/src/data/models/rnnt/parakeet-rnnt-1.1b.nemo"
device = torch.device("cpu")
asr_model = EncDecRNNTBPEModel.restore_from(restore_path=nemo_path, map_location=device)
asr_model.eval()
asr_model._prepare_for_export()

# Convertir toda la red a FP16
asr_model.encoder.half()

# 2. Wrapper
class NeMoEncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, audio_signal):
        encoded, _ = self.encoder(audio_signal=audio_signal, length=None)
        return encoded

wrapper_encoder = NeMoEncoderWrapper(asr_model.encoder).to(device).eval()

# 3. Input
sample_audio_signal = torch.randn(1, 80, 128, dtype=torch.float16, device=device)

# 4. Exportar a ONNX
onnx_path = "rnnt_encoder_fp16.onnx"
print("Exportando a ONNX FP16...")
torch.onnx.export(
    wrapper_encoder,
    (sample_audio_signal,),
    onnx_path,
    input_names=["audio_signal"],
    output_names=["encoded"],
    opset_version=17
)
print(f"ONNX en FP16 exportado en: {onnx_path}")