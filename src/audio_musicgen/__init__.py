import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.io.wavfile import write
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODELO_POR_DEFECTO = "facebook/musicgen-small"
TOKENS_POR_SEGUNDO = 50  # MusicGen genera 50 tokens de audio por segundo
DURACION_MAXIMA = 30  # segundos; limite del modelo en una sola generacion

# MusicGen no conoce nombres de videojuegos: se describe el sonido, no el titulo.
ESTILOS = {
    "noir": (
        "Dark smoky noir jazz instrumental, late night in a rainy city. Slow "
        "swing groove around 90 BPM with brushed drums, deep upright walking "
        "bass, moody Rhodes electric piano chords, a muted trumpet playing a "
        "mysterious melody and a warm tenor saxophone answering. Cool, "
        "mysterious and stylish, vintage studio recording with vinyl warmth."
    ),
    "lounge": (
        "Relaxed smooth lounge jazz instrumental for a card game night in a "
        "smoky back room. Laid-back swing at 100 BPM, soft brushed drums, "
        "walking upright bass, a mellow jazz piano trio comping and soloing, "
        "a warm saxophone melody with a bluesy feel. Cozy, playful and "
        "confident, intimate live club atmosphere, analog warm mix."
    ),
    "salsa": (
        "An upbeat modern salsa song with a bright piano montuno, powerful "
        "brass section with trumpets and trombones, energetic congas, "
        "timbales and bongos, a walking bass line and cowbell keeping the "
        "clave rhythm. Fast tempo, joyful and festive, warm studio mix."
    ),
}


def elegir_dispositivo():
    # cuda - nvidia | cpu - cualquier equipo
    return "cuda" if torch.cuda.is_available() else "cpu"


def leer_argumentos():
    parser = argparse.ArgumentParser(
        description="Genera musica a partir de texto con MusicGen."
    )
    parser.add_argument("--estilo", choices=ESTILOS, default="noir",
                        help="estilo predefinido (por defecto: noir)")
    parser.add_argument("--prompt", help="texto propio; reemplaza al estilo")
    parser.add_argument("--duracion", type=int, default=DURACION_MAXIMA,
                        help=f"segundos de musica, 1 a {DURACION_MAXIMA} (por defecto {DURACION_MAXIMA})")
    parser.add_argument("--modelo", default=MODELO_POR_DEFECTO,
                        help="small (rapido), medium (mejor calidad, ~6 GB) o un id de Hugging Face")
    parser.add_argument("--guidance", type=float, default=3.0,
                        help="fidelidad al prompt (por defecto 3.0)")
    parser.add_argument("--semilla", type=int, help="semilla para repetir el mismo resultado")
    parser.add_argument("--salida", default="musica.wav", help="archivo de salida (por defecto musica.wav)")
    args = parser.parse_args()

    if not 1 <= args.duracion <= DURACION_MAXIMA:
        parser.error(f"--duracion debe estar entre 1 y {DURACION_MAXIMA}")
    if args.modelo in ("small", "medium", "large"):
        args.modelo = f"facebook/musicgen-{args.modelo}"
    return args


def main():
    args = leer_argumentos()
    prompt = args.prompt or ESTILOS[args.estilo]
    dispositivo = elegir_dispositivo()

    if args.semilla is not None:
        torch.manual_seed(args.semilla)

    print(f"Cargando {args.modelo} en {dispositivo}...")
    processor = AutoProcessor.from_pretrained(args.modelo)
    model = MusicgenForConditionalGeneration.from_pretrained(args.modelo).to(dispositivo)
    model.eval()

    print(f"Prompt: {prompt}")
    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(dispositivo)

    print(f"Generando {args.duracion} s de musica...")
    with torch.inference_mode():
        audio = model.generate(
            **inputs,
            do_sample=True,
            guidance_scale=args.guidance,
            max_new_tokens=args.duracion * TOKENS_POR_SEGUNDO,
        )

    sample_rate = model.config.audio_encoder.sampling_rate
    onda = audio[0, 0].cpu().float().numpy()
    # Normaliza el volumen y guarda en PCM de 16 bits (compatible con cualquier reproductor)
    onda = onda / max(float(np.abs(onda).max()), 1e-9) * 0.95
    destino = Path(args.salida)
    write(destino, sample_rate, (onda * 32767).astype(np.int16))
    print(f"Musica guardada en {destino} ({len(onda) / sample_rate:.1f} s)")


if __name__ == "__main__":
    main()
