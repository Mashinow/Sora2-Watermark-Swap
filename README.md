# Sora2-Watermark-Swap

This repository provides a fast and efficient solution for replacing the default Sora watermark with your own social media watermark. The best part? **No neural networks or AI required** - just simple, effective processing.

## How It Works

The script automatically detects the standard Sora watermark position and replaces it with your custom social media handle/watermark using traditional computer vision techniques.

## Quick Start

1. Place your video file named `input.mp4` in the project root directory
2. Run the watermark replacement script:
   ```bash
   python hide_watermark.py
3. Your processed video will be saved as `output.mp4` in the output folder

## Example Results

Check out these demonstration files included in the repository:

### **Original Video** (with default Sora watermark):

https://github.com/user-attachments/assets/d9751b1e-8363-483a-b572-47360bda8d62


### **Processed Video** (with your social media watermark):

https://github.com/user-attachments/assets/30110b06-6da1-43ce-8307-e0c035dc6b15






## Features

- Lightning-fast processing (no heavy AI models)
- Simple one-command operation
- Preserves original video quality
- Easy customization of watermark text/position
- No external dependencies or API keys required

## Customization

Edit the `hide_watermark.py` file to customize:
- Your social media handle/text
- Watermark position, size, and style
- Font properties and opacity
