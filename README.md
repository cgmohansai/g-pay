
```bash
pip install -r req.txt
```

##Inside the images

the raw palm vein images are stored initially when the user is trying to register for the first time.
then preprocess is being processed and output is saved in the img_out folder


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

'{person_id}_{left or right}_{spectrum range}_{copies}.jpg

## Running the preprocessing script
From the repo root:

```bash
python src/preprocessing/preprocess.py
```

