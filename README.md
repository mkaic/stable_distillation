# Instructions
1. On the server you want to use for training, clone this repo.
2. Inside the repo, create a file called `token.txt` and paste your HuggingFace token into it.
3. Run `bash environment_setup.sh` to install utilities and packages and create an appropriate conda environment.
4. Run `conda activate distillery` to activate the environment
5. Run stage 1 distillation with `accelerate run stage_1_distillation.py`
6. Run stage 2 distillation with `accelerate run stage_2_distillation.py`
7. Repeat step 6 until the model is distilled to your liking.
8. The distilled model weights can be loaded and used with the DistilledUNet2DConditionModel class. Enjoy your fast sampling!