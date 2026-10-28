from nemo.collections.asr.models import EncDecRNNTBPEModel 

print("Cargando modelo FP32")
model = EncDecRNNTBPEModel.restore_from("../src/data/models/rnnt/parakeet-rnnt-1.1b.nemo", map_location="cpu")

print("Convirtiendo pesos a FP16")
model = model.half()

print("Empaquetando nuevo archivo .nemo")
model.save_to("parakeet-rnnt-1.1b_fp16.nemo")

print("Conversión terminada")