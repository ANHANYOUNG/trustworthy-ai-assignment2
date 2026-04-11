git clone https://github.com/peikexin9/deepxplore

conda create -n trustworthy-ai-a2 python=3.10 -y

pip install torch torchvision torchaudio
pip install numpy matplotlib pillow tqdm opencv-python pandas scikit-learn

conda activate trustworthy-ai-a2
pip install ipykernel jupyter
python -m ipykernel install --user --name trustworthy-ai-a2 --display-name "Python (trustworthy-ai-a2)"
