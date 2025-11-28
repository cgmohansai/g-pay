# gpay palm preprocessing

Pipeline for extracting palm ROIs and vein features from near-infrared hand images.

## Features
- interactive person selection (defaults to `007`)
- wrist cropping, Gaussian blur, and binarisation
- contour extraction with refined convexity-defect pruning
- geometric ROI definition aligned with finger valleys
- Gabor + CLAHE enhancement and final binary mask
- matplotlib visualisation of every stage
- automatic export of square ROIs (`dataset/img_out`)

## Requirements
Install Python 3.9+ and the packages listed in `req.txt`:

```bash
pip install -r req.txt
```

## Dataset layout
Expect the repository root to contain:
```
dataset/
  images/
    <person>_<hand>_<spectrum>_<idx>.jpg
  img_out/        # created automatically if missing
src/
  preprocessing/
    preprocess.py
```

Images are addressed by pattern `PERSON_HAND_SPECTRUM_*.jpg`, e.g. `007_l_940_01.jpg`.

## Running the script
From the repo root:

```bash
python src/preprocessing/preprocess.py
```

You will be prompted for the person ID. Press Enter to keep the default. The script then:
1. loads up to three matching frames,
2. shows a 6×3 grid of preprocessing steps plus a 4×3 grid of ROI refinement,
3. writes the rectified ROI (`*_roi_raw.png`) and its binary counterpart (`*_roi_binary.png`) to `dataset/img_out/`.

## Configuration tips
- change `selected_hand` or `selected_spectrum` near the top of `preprocess.py` to target other captures.
- adjust the Gaussian blur size or threshold for different sensors.
- tweak the convexity suppression radius (`min_distance`) if fingers sit closer/farther in your dataset.

## Troubleshooting
- **No images found**: verify filenames under `dataset/images` follow the expected pattern.
- **No defects detected**: relax cropping, try different threshold, or confirm the hand isn’t touching image borders.
- **Matplotlib windows blocked**: ensure you run inside an environment with GUI support (e.g. local Python, not headless server).

## License
Add your preferred license statement here.

