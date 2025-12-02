# GlyphPay

## Setup

1.  **Install Python 3.x**.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

### 1. Run the Pipeline

Run the main script with an input image:

```bash
python main.py --image data/test_hand.jpg
```

## Outputs

The script saves intermediate steps in the `data/` folder:

- **`step1_cropped_input.jpg`**: Input image with wrist removed.
- **`step2_roi_750px_fixed.jpg`**: The extracted 750x750 ROI.
- **`step3_clahe_output.jpg`**: Grayscale veins after enhancement (before binarization).
- **`step4_final_binary.jpg`**: Final output - Black vein strokes on white background.
- **`step5_visualization.png`**: Debug visualization showing geometric lines and detection points.
