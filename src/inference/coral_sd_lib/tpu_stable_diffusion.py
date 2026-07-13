# most of the code is from Keras CV implementation, 
# https://github.com/keras-team/keras-cv/blob/v0.4.1/keras_cv/models/stable_diffusion/stable_diffusion.py,
# not fully implmented, just to show that converted models work

# also borrow constants from keras cv
from inference.coral_sd_lib.stable_diffusion_constants import _UNCONDITIONAL_TOKENS, _ALPHAS_CUMPROD
from inference.coral_sd_lib.keras_simple_tokenizer import SimpleTokenizer

import math
import tensorflow as tf
from tensorflow import keras
import numpy as np

from tflite_runtime.interpreter import load_delegate, Interpreter

class StableDiffusionTFLite:
    def __init__(self, img_width, img_height, base_model_path):
        self.img_width = img_width
        self.img_height = img_height
        self.base_model_path = base_model_path
        
        self.MAX_PROMPT_LENGTH = 77
        delegate = load_delegate("libedgetpu.so.1")
        
        self.text_encoder = Interpreter(
            model_path=str(self.base_model_path / "sd_text_encoder_qint8_edgetpu.tflite"),
            experimental_delegates=[delegate]
        )
        self.text_encoder.allocate_tensors()
        
        self.decoder = Interpreter(
            model_path=str(self.base_model_path / "sd_decoder_qint8_modded_edgetpu.tflite"),
            experimental_delegates=[delegate]
        )
        
        self.diffusion_model_first = Interpreter(
            model_path=str(self.base_model_path / "diffusion_model_first_qint8_modded_edgetpu.tflite"),
            experimental_delegates=[delegate]
        )
        
        self.diffusion_model_second = Interpreter(
            model_path=str(self.base_model_path / "diffusion_model_second_qint8_modded_edgetpu.tflite"),
            experimental_delegates=[delegate]
        )
        
        self.tokenizer = SimpleTokenizer()

    def _get_initial_alphas(self, timesteps):
        alphas = [_ALPHAS_CUMPROD[t] for t in timesteps]
        alphas_prev = [1.0] + alphas[:-1]

        return alphas, alphas_prev
    
    def _get_initial_diffusion_noise(self, batch_size, seed):
        if seed is not None:
            return tf.random.stateless_normal(
                (batch_size, self.img_height // 8, self.img_width // 8, 4),
                seed=[seed, seed],
            )
        else:
            return tf.random.normal(
                (batch_size, self.img_height // 8, self.img_width // 8, 4)
            )
        
    def _get_timestep_embedding(self, timestep, batch_size, dim=320, max_period=10000):
        half = dim // 2
        freqs = tf.math.exp(
            -math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half
        )
        args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
        embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
        embedding = tf.reshape(embedding, [1, -1])
        return tf.repeat(embedding, batch_size, axis=0)

    def _get_pos_ids(self):
        return tf.convert_to_tensor([list(range(self.MAX_PROMPT_LENGTH))], dtype=tf.int32)

    def encoded_token_padded(self, prompt):
        inputs = self.tokenizer.encode(prompt)
        phrase = inputs + [49407] * (self.MAX_PROMPT_LENGTH - len(inputs))
        phrase = tf.convert_to_tensor([phrase], dtype=tf.int32)

        return phrase, self._get_pos_ids()
    
    def encode_text(self, prompt):
        input_details = self.text_encoder.get_input_details()
        output_details = self.text_encoder.get_output_details()

        token, pos = self.encoded_token_padded(prompt)
        self.text_encoder.set_tensor(input_details[0]['index'], token)
        self.text_encoder.set_tensor(input_details[1]['index'], pos)

        self.text_encoder.invoke()

        output_data = self.text_encoder.get_tensor(output_details[0]['index'])

        return output_data
    
    def encode_text_2(self, token, pos):
        input_details = self.text_encoder.get_input_details()
        output_details = self.text_encoder.get_output_details()

        # token, pos = encoded_token_padded(prompt)
        self.text_encoder.set_tensor(input_details[0]['index'], token)
        self.text_encoder.set_tensor(input_details[1]['index'], pos)

        self.text_encoder.invoke()

        output_data = self.text_encoder.get_tensor(output_details[0]['index'])

        return output_data


    def _expand_tensor(self, text_embedding, batch_size):
        """Extends a tensor by repeating it to fit the shape of the given batch size."""
        text_embedding = tf.squeeze(text_embedding)
        if text_embedding.shape.rank == 2:
            text_embedding = tf.repeat(
                tf.expand_dims(text_embedding, axis=0), batch_size, axis=0
            )
        return text_embedding
    
    def _get_unconditional_context(self):
        unconditional_tokens = tf.convert_to_tensor(
            [_UNCONDITIONAL_TOKENS], dtype=tf.int32
        )
        unconditional_context = self.encode_text_2(unconditional_tokens, self._get_pos_ids())

        return unconditional_context
    
    def get_index_of_name(self, tensors, name):
        for a in tensors:
            if a['name'] == name:
                return a['index']

    def diffusion_model(self, latent, t_emb, unconditional_context):
        input_names = [
            "serving_default_args_0:0",
            "serving_default_args_0_1:0",
            "serving_default_args_0_2:0",
            "serving_default_args_0_3:0",
            "serving_default_args_0_4:0",
            "serving_default_args_0_5:0",
            "serving_default_args_0_6:0",
            "serving_default_args_0_7:0",
            "serving_default_args_0_8:0",
            "serving_default_args_0_9:0",
            "serving_default_args_0_10:0",
            "serving_default_args_0_11:0",
            "serving_default_args_0_12:0",
        ]
        
        output_names = [
            "StatefulPartitionedCall:6",
            "StatefulPartitionedCall:4",
            "StatefulPartitionedCall:0",
            "StatefulPartitionedCall:12",
            "serving_default_input_1:0",
            "StatefulPartitionedCall:11",
            "StatefulPartitionedCall:3",
            "StatefulPartitionedCall:10",
            "StatefulPartitionedCall:9",
            "StatefulPartitionedCall:5",
            "StatefulPartitionedCall:8",
            "StatefulPartitionedCall:7",
            "StatefulPartitionedCall:2",
        ]

        first_input_details = self.diffusion_model_first.get_input_details()
        first_output_details = self.diffusion_model_first.get_output_details()
        
        self.diffusion_model_first.resize_tensor_input(first_input_details[0]['index'], unconditional_context.shape)
        self.diffusion_model_first.resize_tensor_input(first_input_details[1]['index'], latent.shape)
        self.diffusion_model_first.resize_tensor_input(first_input_details[2]['index'], t_emb.shape)
        self.diffusion_model_first.allocate_tensors()        
        self.diffusion_model_first.set_tensor(first_input_details[0]['index'], unconditional_context)
        self.diffusion_model_first.set_tensor(first_input_details[1]['index'], latent)
        self.diffusion_model_first.set_tensor(first_input_details[2]['index'], t_emb)

        self.diffusion_model_first.invoke()

        second_input_details = self.diffusion_model_second.get_input_details()
        second_output_details = self.diffusion_model_second.get_output_details()        

        for i in range(13):
            self.diffusion_model_second.resize_tensor_input(
                self.get_index_of_name(second_input_details, input_names[i]),
                self.diffusion_model_first.get_tensor(self.get_index_of_name(first_output_details, output_names[i])).shape)

        self.diffusion_model_second.allocate_tensors()

        for i in range(13):
            self.diffusion_model_second.set_tensor(
                self.get_index_of_name(second_input_details, input_names[i]),
                self.diffusion_model_first.get_tensor(self.get_index_of_name(first_output_details, output_names[i])))

        self.diffusion_model_second.invoke()
        output_data = self.diffusion_model_second.get_tensor(second_output_details[0]['index'])
        return output_data
    
    def decode(self, encoded_image):
        input_details = self.decoder.get_input_details()
        output_details = self.decoder.get_output_details()
        
        self.decoder.resize_tensor_input(input_details[0]['index'], encoded_image.shape)
        self.decoder.allocate_tensors()
        self.decoder.set_tensor(input_details[0]['index'], encoded_image)

        self.decoder.invoke()

        output_data = self.decoder.get_tensor(output_details[0]['index'])

        return output_data

    def generate_image(
        self,
        encoded_text,
        batch_size,
        num_steps,
        unconditional_guidance_scale,
        seed,
        negative_prompt=None,
        diffusion_noise=None,
    ):
        context = self._expand_tensor(encoded_text, batch_size)

        if negative_prompt is None:
            unconditional_context = tf.repeat(
                self._get_unconditional_context(), batch_size, axis=0
            )
        else:
            unconditional_context = self.encode_text(negative_prompt)
            unconditional_context = self._expand_tensor(
                unconditional_context, batch_size
            )
        if diffusion_noise is not None:
            diffusion_noise = tf.squeeze(diffusion_noise)
            if diffusion_noise.shape.rank == 3:
                diffusion_noise = tf.repeat(
                    tf.expand_dims(diffusion_noise, axis=0), batch_size, axis=0
                )
            latent = diffusion_noise
        else:
            latent = self._get_initial_diffusion_noise(batch_size, seed)

        timesteps = tf.range(1, 1000, 1000 // num_steps)
        alphas, alphas_prev = self._get_initial_alphas(timesteps)
        progbar = keras.utils.Progbar(len(timesteps))
        iteration = 0
        for index, timestep in list(enumerate(timesteps))[::-1]:
            latent_prev = latent  # Set aside the previous latent vector
            t_emb = self._get_timestep_embedding(timestep, batch_size)
            unconditional_latent = self.diffusion_model(latent, t_emb, unconditional_context)
            
            latent = self.diffusion_model(latent, t_emb, context)
            latent = unconditional_latent + unconditional_guidance_scale * (
                latent - unconditional_latent
            )
            a_t, a_prev = alphas[index], alphas_prev[index]
            pred_x0 = (latent_prev - math.sqrt(1 - a_t) * latent) / math.sqrt(a_t)
            latent = latent * math.sqrt(1.0 - a_prev) + math.sqrt(a_prev) * pred_x0
            iteration += 1
            progbar.update(iteration)

        # Decoding stage
        decoded = self.decode(latent)
        decoded = ((decoded + 1) / 2) * 255
        return np.clip(decoded, 0, 255).astype("uint8")