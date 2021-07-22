set -e

conda remove --name sim2real_ad --all -y
echo "Creating conda environmnet: sim2real_ad"
conda env create -f environment_dependencies.yml

echo "Clone and install gym-duckietown"
git clone --branch v6.1.16 --single-branch --depth 1 https://github.com/duckietown/gym-duckietown.git ./gym-duckietown
conda run -vn sim2real_ad pip install -e gym-duckietown

echo "conda environment setup complet
Run  $ conda activate sim2real_ad"


