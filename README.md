# Vapoursynth-ATWT

**Vapoursynth-ATWT** is a VapourSynth plugin that implements the À Trous Wavelet Transform for frequency separation. It decomposes a video clip into multiple frequency layers (Detail) and a residual layer (Base).

## Core Plugin API

The plugin exports two low-level functions.

### `atwt.ExtractFrequency(clip, radius=1)`

Extracts a detail layer from the input clip.
*   **Formula**: $Detail = Src - Blur(Src)$
*   **clip**: Input clip.
*   **radius**: The dilation factor of the wavelet kernel. Step size = $2^{(radius-1)}$. Default is 1.

### `atwt.ReplaceFrequency(base, detail)`

Recombines a base layer with a detail layer.
*   **Formula**: $Output = Base + (Detail - Neutral)$
*   **base**: The low-frequency clip.
*   **detail**: The high-frequency clip (result from `ExtractFrequency`).
*   **Note**: This function automatically handles neutral grey offsets for integer formats.

---

## Python Helper Scripts

To modify specific levels, one cannot simply call `ExtractFrequency(radius=2)` on the source, as that would include Level 1 details as well. One must peel the layers recursively.

```python
import vapoursynth as vs
core = vs.core

def atwt_decompose(clip: vs.VideoNode, levels: int = 2) -> list[vs.VideoNode]:
    """
    [Level_1, Level_2, ..., Level_N, Base(Residual)]
    
    layers[0]  -> Level 1
    layers[1]  -> Level 2
    layers[-1] -> Base
    """
    details = []
    current_base = clip
    
    for i in range(1, levels + 1):
        d = core.atwt.ExtractFrequency(current_base, radius=i)
        next_base = core.std.MakeDiff(current_base, d)
        details.append(d)
        current_base = next_base

    return details + [current_base]

def atwt_recombine(layers: list[vs.VideoNode]) -> vs.VideoNode:
    current_clip = layers[-1]
    details_reversed = layers[:-1][::-1]
    
    for detail in details_reversed:
        current_clip = core.atwt.ReplaceFrequency(base=current_clip, detail=detail)
    
    return current_clip
```

## Compilation

```
meson setup builddir
ninja -C builddir
ninja -C builddir install
```