# 🚀 Quick Start - Train SAMCell on Ratcliff Lab Data

## ✅ Datasets Ready!

Two datasets have been prepared and are ready for training:

| Dataset | Samples | Size | Location |
|---------|---------|------|----------|
| RatcliffLab-model | 9 images | 9.0 MB | `datasets/RatcliffLab-model/` |
| RatcliffLab-analyze | 12 images | 12 MB | `datasets/RatcliffLab-analyze/` |

---

## 🎯 Train Now (Copy & Paste)

### Option 1: Train on RatcliffLab-analyze (12 images)

```bash
cd /Users/saahilsanganeriya/Documents/Saahil/SAMCell/SAMCell_dev/SAMCell/training

python train.py \
    --datasets ../datasets/RatcliffLab-analyze \
    --batch-size 2 \
    --num-epochs 40 \
    --learning-rate 1e-4 \
    --sam-model facebook/sam-vit-base
```

### Option 2: Train on Both Datasets Combined (21 images)

```bash
cd /Users/saahilsanganeriya/Documents/Saahil/SAMCell/SAMCell_dev/SAMCell/training

python train.py \
    --datasets ../datasets/RatcliffLab-model ../datasets/RatcliffLab-analyze \
    --batch-size 4 \
    --num-epochs 50 \
    --learning-rate 1e-4 \
    --patience 10 \
    --sam-model facebook/sam-vit-base
```

### Option 3: Fast Training (Freeze Vision Encoder)

```bash
cd /Users/saahilsanganeriya/Documents/Saahil/SAMCell/SAMCell_dev/SAMCell/training

python train.py \
    --datasets ../datasets/RatcliffLab-analyze \
    --batch-size 4 \
    --num-epochs 40 \
    --no-finetune-vision \
    --sam-model facebook/sam-vit-base
```

---

## 📊 What to Expect

- **Training time**: 2-4 hours (depending on GPU)
- **Checkpoints saved**: Every 5 epochs + best model
- **Output location**: `SAMCell/checkpoints/samcell-{dataset}-sam-vit-base/`
- **Monitoring**: WandB dashboard (automatic)

---

## 📁 Files Created

```
SAMCell/
├── datasets/
│   ├── RatcliffLab-model/
│   │   ├── imgs.npy (9 samples, 512x512, green channel)
│   │   └── dist_maps.npy (distance transform maps)
│   └── RatcliffLab-analyze/
│       ├── imgs.npy (12 samples, 512x512, green channel)
│       └── dist_maps.npy (distance transform maps)
└── dataset_processing/
    ├── process_ratcliff_lab_data.py (conversion script)
    └── RATCLIFF_LAB_DATASET_README.md (detailed guide)
```

---

## 🎓 Training Tips

1. **Small dataset?** Use `--patience 10` for early stopping
2. **GPU memory issues?** Reduce `--batch-size` to 1 or 2
3. **Want faster training?** Add `--no-finetune-vision`
4. **Need more data?** Combine both datasets
5. **Track experiments?** WandB enabled by default

---

## 📖 Full Documentation

- **Detailed guide**: `SAMCell/dataset_processing/RATCLIFF_LAB_DATASET_README.md`
- **Training options**: `SAMCell/training/README.md`
- **Re-process data**: `SAMCell/dataset_processing/process_ratcliff_lab_data.py --help`

---

## ✨ After Training

Your trained model will be saved as:
- `checkpoints/samcell-{datasets}-sam-vit-base/best_model.pt`

Use it for inference on new mitotic spindle images!

**Ready to train? Pick an option above and run it!** 🚀
