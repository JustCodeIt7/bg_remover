# Image Background Remover + Upscaler (Streamlit)

Streamlit app to remove image backgrounds (via rembg) and optionally upscale images using EDSR from the super-image package.

## Features

- Remove background from PNG/JPG/JPEG images
- Auto-resize large inputs for safer processing
- Download processed PNG
- Optional super-resolution upscaling (×2/×3/×4) using EDSR

## Install

Create/activate a Python 3.9+ environment, then install dependencies:

```bash
pip install -r requirements.txt
```

If you only need upscaling later, you can install it separately:

```bash
pip install super-image
```

## Run

```bash

- Background removal: <https://github.com/danielgatis/rembg>
- EDSR super-resolution: <https://github.com/eugenesiow/super-image>

## Usage
1. Upload an image (PNG/JPG/JPEG)
2. Download the background-removed result from the sidebar
3. (Optional) Open the “Upscale” expander, enable upscaling, choose scale and source (Fixed or Original), then download the upscaled output

Notes:
- Upscaling is compute-intensive and slower on CPU; a safety cap limits extremely large outputs
- Default demo images `zebra.jpg` and `wallaby.png` are optional; add them next to the script if desired

## Credits
- Background removal: https://github.com/danielgatis/rembg
- EDSR super-resolution: https://github.com/eugenesiow/super-image
```
