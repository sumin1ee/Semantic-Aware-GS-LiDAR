# LiSenCE: LiDAR Semantic Gaussian Splatting with Coherent Embedding Field
> **The Project on Advanced Computer Vision 2025**

## 🛠️ Pipeline
<div align="center">
  <img src="assets/overview.pdf"/>
</div><br/>

## Get started
### Environment
```
# Clone the repo.
# NOTE: Repository will be made public after the project release.
git clone ...

# Make a conda environment.
conda create --name lisence python=3.9
conda activate lisence

# Install PyTorch according to your CUDA version
# CUDA 11.7 (Ampere and earlier)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu117

# CUDA 11.8 (Ada / Blackwell)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install requirements.
pip install -r requirements.txt

# Install simple-knn
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git
pip install ./simple-knn

# compile packages in chamfer
cd chamfer/chamfer3D
python setup.py install
cd ../..
```

### 📁 Dataset
#### SemanticKITTI dataset ([Download](http://www.semantic-kitti.org/))
Download the official SemanticKITTI release (sequences, labels, poses) and organize as follows:
```bash
Semantic-Aware-GS-LiDAR
└── data
    └── SemanticKITTI
        └── sequences
            └── 00
                ├── velodyne
                ├── labels
                ├── calib.txt
                └── poses.txt  # copy from poses/00.txt in the official release
```

1. Copy each `poses/{seq}.txt` file into `sequences/{seq}/poses.txt`; the loader expects poses next to the LiDAR frames.
2. Set `sequence_id` (e.g., `00`) in `configs/semantickitti_nvs.yaml` or override it via the CLI.
3. (Optional) Adjust `semantic_label_map_path` if you use a custom remapping; by default we rely on `configs/semantickitti_label_map.yaml`.

#### nuScenes dataset
We support nuScenes LiDAR training via the `NuScenes` scene loader. The expected layout is:

```bash
Semantic-Aware-GS-LiDAR
└── data
    └── nuscenes -> /data/nuscenes  # symbolic link to the official nuScenes dataroot
```

1. Create the symbolic link (adjust the absolute path to your local dataset if needed):
   ```bash
   cd Semantic-Aware-GS-LiDAR
   mkdir -p data
   ln -s /path/to/nuscenes data/nuscenes
   ```
2. Select a scene (e.g. `scene-0103`) and update `configs/nuscenes_nvs.yaml`:
   ```bash
   scene_name: "scene-0103"
   source_path: "/home/sumin/projects/ACV/Semantic-Aware-GS-LiDAR/data/nuscenes"
   ```
3. Launch training:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python train.py \
       --config configs/nuscenes_nvs.yaml \
       source_path=/home/sumin/projects/ACV/Semantic-Aware-GS-LiDAR/data/nuscenes \
       model_path=eval_output/nuscenes_reconstruction/scene-0103
   ```

The loader expects `nuscenes-devkit` and `pyquaternion`, which are included in `requirements.txt`.


### Training
```
# SemanticKITTI
CUDA_VISIBLE_DEVICES=0 python train.py \
--config configs/semantickitti_nvs.yaml \
source_path=/home/sumin/projects/ACV/Semantic-Aware-GS-LiDAR/data/SemanticKITTI \
sequence_id=00 \
model_path=eval_output/semantickitti_reconstruction/seq00

# nuScenes
CUDA_VISIBLE_DEVICES=0 python train.py \
--config configs/nuscenes_nvs.yaml \
source_path=/home/sumin/projects/ACV/Semantic-Aware-GS-LiDAR/data/nuscenes \
scene_name=scene-0103 \
model_path=eval_output/nuscenes_reconstruction/scene-0103
```

After training, evaluation results can be found in `{EXPERIMENT_DIR}/eval_output` directory.
The training logs will be saved in `log.txt`. If you need to display them in the terminal, please use the `--show_log` option.

### Evaluating
You can also use the following command to evaluate using pre-trained checkpoints.
```
# SemanticKITTI
CUDA_VISIBLE_DEVICES=0 python train.py \
--config configs/semantickitti_nvs.yaml \
source_path=/home/sumin/projects/ACV/Semantic-Aware-GS-LiDAR/data/SemanticKITTI \
sequence_id=00 \
model_path=eval_output/semantickitti_reconstruction/seq00 \
--test_only

# nuScenes
CUDA_VISIBLE_DEVICES=0 python train.py \
--config configs/nuscenes_nvs.yaml \
source_path=/home/sumin/projects/ACV/Semantic-Aware-GS-LiDAR/data/nuscenes \
scene_name=scene-0103 \
model_path=eval_output/nuscenes_reconstruction/scene-0103 \
--test_only
```

## 📜 BibTeX
``` bibtex
@inproceedings{jiang2025gslidar,
  title={GS-LiDAR: Generating Realistic LiDAR Point Clouds with Panoramic Gaussian Splatting},
  author={Jiang, Junzhe and Gu, Chun and Chen, Yurui and Zhang, Li},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```