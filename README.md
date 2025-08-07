# Dreamspace Co-Pilot: AI + BCI Co-Piloted Visual Dreamspace

## Overview

Dreamspace Co-Pilot is a hybrid Brain-Computer Interface (BCI) and Artificial Intelligence (AI) system that enables users to co-pilot and explore imaginative visual experiences in a way that feels like a dream: semi-intentional, emotionally responsive, yet partially out of one's control. The system blends generative visual models (e.g., Kandinsky 2.1, Stable Diffusion) with passive and active EEG-based BCI signals to steer imagery in latent space, allowing users to influence the unfolding of story-like or symbolic visual narratives. An AI co-pilot spins a narrative which drives the overall arch of the image sequence, and much like in a dream, the human user may not be aware of that narrative, but simply experiences it as a shift of themes throughout the sequence. The AI does not reveal the narrative until after the session.

## Key Features

- 🔮 **Latent-space continuity**: Images evolve gradually to maintain narrative and visual cohesion.
- 🧠 **EEG/BCI input support**: Use EEG signals or manual keyboard proxies to steer the generation.
- ✨ **LLM-driven storytelling**: Asynchronous prompts update the narrative and influence image evolution.
- 🖼️ **Diffusion-based generation**: Uses models like Kandinsky 2.1 or Stable Diffusion via Hugging Face.
- 🧪 **Exploratory modes**: Switch between different control paradigms — e.g., prompt morphing, latent vector traversal, img2img evolution.
- 🌈 **Async Multi-Prompt Sequences**: Generate smooth interpolation sequences between multiple prompts with server-side processing and progressive PNG saving.

## New: Async Multi-Prompt Generation

Generate cinematic sequences that smoothly interpolate between multiple prompts:

```bash
# Basic usage
PYTHONPATH=src python examples/async_multi_prompt.py \
    --prompts "serene mountain lake" "vibrant autumn forest" "mystical desert dunes" \
    --output-dir ./my_sequence

# High quality
PYTHONPATH=src python examples/async_multi_prompt.py \
    --prompts "ethereal garden" "cosmic nebula" "ancient temple" \
    --output-dir ./hq_sequence \
    --width 1024 --height 1024 --batch-size 12 \
    --model sd21_server --seed 42
```

Features:
- **Asynchronous processing**: Client exits immediately, server continues generation
- **Progressive saving**: PNG files saved as generated for progress monitoring  
- **Memory efficient**: Images saved to disk immediately to reduce memory usage
- **Consistent composition**: Shared latent cookies maintain visual coherence
- **Multiple models**: Support for SD 1.5, SD 2.1, and Kandinsky 2.1

See [docs/async-multi-prompt.md](docs/async-multi-prompt.md) for detailed documentation.

