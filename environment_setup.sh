apt-get update && apt-get install -y screen nvtop htop
conda update -n base conda
conda env create --file environment.yaml
conda init bash
echo "conda activate distillery" >> ~/.bashrc