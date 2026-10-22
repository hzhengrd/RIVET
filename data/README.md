# Datasets

Download links and the subsets used in the paper. The preparation scripts in
this directory convert each download into the layout the trainer reads; the
end-to-end steps are in the top-level [README](../README.md#pipeline).

## HA-ViD

- **Download**: <https://iai-hrc.github.io/ha-vid>
- **Subset used**: all three synchronized views (side, front, and top/overhead),
  with both hands pooled into a single model. Official split (41 task videos
  held out per view): 4,835 training clips (2,476 left-hand, 2,359 right-hand)
  and 584 left-hand + 596 right-hand test clips per view. Long-tail
  augmentation raises the training set to 5,826 rows (`train_aug`).
- **Scripts**: `havid.py`, `havid_side.py`, `crop_havid.py`, `pool.py`,
  `model/augment.py`

## Assembly101

- **Download**: <https://assembly-101.github.io/> — recommended mirror:
  <https://huggingface.co/datasets/cvml-nus/assembly101> — download scripts:
  <https://github.com/assembly-101/assembly101-download-scripts>
- **Subset used**: the single fixed view **v4** (`C10119_rgb`), workspace-cropped
  (crop `(440,10,1880,1050)`), training head capped at 200 examples per class
  (`train_aug200`). Raw v4 counts: 47,651 training / 15,699 validation /
  21,934 test clips.
- **Scripts**: `assembly101.py`, `crop.py`, `model/subsample.py`,
  `model/augment.py`

## IKEA ASM

- **Download**: <https://ikeaasm.github.io/> (Google Drive folder); processing
  code: <https://github.com/IkeaASM/IKEA_ASM_Dataset>
- **Subset used**: the **top** view (`dev3`), workspace-cropped, official
  environment-based split: 5,729 training / 2,589 test clips over 33 classes.
- **Scripts**: `ikea.py`, `crop.py`, `workspace.py`, `model/augment.py`
