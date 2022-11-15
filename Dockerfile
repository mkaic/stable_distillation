# docker build -t distillery:dev .
# docker run -it -d -v $PWD:/workspace --gpus all --name distillery --ipc=host -p 8888  distillery:dev


FROM nvcr.io/nvidia/pytorch:22.10-py3
WORKDIR /buildfiles
COPY environment.yaml /buildfiles  
RUN apt-get update && apt-get install -y screen nvtop htop && \
    conda update -n base conda && \
    conda env create --file environment.yaml && \
    conda init bash && \
    echo "conda activate distillery" >> ~/.bashrc 

