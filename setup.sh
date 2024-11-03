# Environment
python -m venv tijepa
source tijepa/bin/activate
pip install -r requirements-tijepa.txt

# Download COCO
wget -P src/datasets/train.zip http://images.cocodataset.org/zips/train2017.zip
unzip src/datasets/train.zip
wget -P src/datasets/annotations.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip src/datasets/annotations.zip

# Download MVSA
gdown 1FRI3TR-Z2jEr9WLwWhJkliG5q0oxJIRd -O src/datasets/mvsa.zip
unzip src/datasets/mvsa.zip

# Download checkpoints from ijepa
wget https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar
