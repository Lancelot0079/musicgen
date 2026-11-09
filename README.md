# MusicGen: generación de música con IA

Genera música a partir de texto con [MusicGen](https://huggingface.co/facebook/musicgen-small) (Meta) usando `transformers` y PyTorch. Incluye estilos de jazz listos para usar, pensados para sonar como el jazz nocturno y de club de videojuegos como *Deadlock* o *Poker Night at the Inventory*.

> MusicGen no entiende nombres de juegos, así que los estilos describen el sonido (instrumentos, tempo, ambiente) en lugar de citar los títulos.

## Requisitos

- [uv](https://docs.astral.sh/uv/) y Python 3.12 o superior
- Conexión a internet la primera vez (descarga el modelo)
- Funciona en CPU; usa GPU NVIDIA si hay una disponible

## Ejecución

```bash
uv sync
uv run audio-musicgen
```

Por defecto genera 30 segundos de jazz *noir* en `musica.wav`.

### Opciones

| Opción | Descripción |
|---|---|
| `--estilo {noir,lounge,salsa}` | Estilo predefinido (por defecto `noir`) |
| `--prompt "texto"` | Texto propio; reemplaza al estilo |
| `--duracion N` | Segundos de música, de 1 a 30 (por defecto 30) |
| `--modelo {small,medium,large}` | `small` es rápido; `medium` suena mejor pero pesa ~6 GB y es más lento |
| `--guidance X` | Fidelidad al prompt (por defecto 3.0) |
| `--semilla N` | Repite exactamente el mismo resultado |
| `--salida archivo.wav` | Archivo de salida |

Ejemplos:

```bash
uv run audio-musicgen --estilo lounge --duracion 20
uv run audio-musicgen --estilo noir --modelo medium --salida jazz_medium.wav
uv run audio-musicgen --prompt "slow jazz ballad with piano and double bass"
```

## Estilos

- **noir**: jazz oscuro y humeante de noche lluviosa. Swing lento (~90 BPM), escobillas, contrabajo, piano Rhodes, trompeta con sordina y saxo tenor.
- **lounge**: jazz relajado de sala de cartas. Swing tranquilo (~100 BPM), trío de piano, contrabajo, escobillas y saxo con aire de blues.
- **salsa**: el estilo original del proyecto.

## Consejos para mejores resultados

- Escribe los prompts en inglés y describe instrumentos, tempo, ambiente y tipo de grabación.
- Prueba varias semillas (`--semilla 1`, `2`, `3`...) y quédate con la mejor toma: la generación es aleatoria.
- El modelo `medium` mejora claramente la calidad; en CPU puede tardar varios minutos por 30 s.
- MusicGen genera como máximo 30 s por toma.

## Ejemplo

[`ejemplos/jazz_noir.wav`](ejemplos/jazz_noir.wav) fue generado con `uv run audio-musicgen --estilo noir --semilla 21`.

## Estructura

```
├── pyproject.toml
├── uv.lock
├── src/audio_musicgen/__init__.py   # función main() y estilos
└── ejemplos/jazz_noir.wav           # audio de ejemplo
```
