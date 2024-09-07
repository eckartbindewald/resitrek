#!/bin/bash

# Set paths
ROOT_DIR="/notebooks"
# GOOGLE_DRIVE_DIR=$ROOT_DIR # "/content/drive/MyDrive"
MINICONDA_DIR="$ROOT_DIR/miniconda"
CONDA_ENV_NAME="protrek"
CONDA_ENV_DIR="$MINICONDA_DIR/conda_envs/$CONDA_ENV_NAME"
REPO_DIR='ProTrek' # ${GOOGLE_DRIVE_DIR}/repos/ProTrek'
# echo 'Creating directory ${REPO_DIR}'
# mkdir -p $REPO_DIR

export PATH=$PATH":$MINICONDA_DIR/bin"
# Function to install Miniconda if not installed
install_miniconda() {
    if [ ! -d "$MINICONDA_DIR" ]; then
        echo "Downloading Miniconda installer..."
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$ROOT_DIR/miniconda.sh"
        
        echo "Making installer executable..."
        chmod +x "$ROOT_DIR/miniconda.sh"
        
        echo "Installing Miniconda..."
        bash "$ROOT_DIR/miniconda.sh" -b -p "$MINICONDA_DIR"
        
        echo "Initializing Conda..."
        "$MINICONDA_DIR/bin/conda" init bash
        
        echo "Cleaning up installer..."
        rm "$ROOT_DIR/miniconda.sh"
        
        echo "Sourcing .bashrc to activate conda..."
        source ~/.bashrc
    else
        echo "Miniconda is already installed."
    fi
}

# Function to ensure Conda is available
ensure_conda() {
    if ! command -v conda &> /dev/null; then
        echo "Conda command not found, attempting to source..."
        source "$MINICONDA_DIR/bin/activate"
    fi
}

# Function to create Conda environment if not exists
create_conda_env() {
    if [ ! -d "$CONDA_ENV_DIR" ]; then
        echo "Creating Conda environment named $CONDA_ENV_NAME at $CONDA_ENV_DIR..."
        "$MINICONDA_DIR/bin/conda" create --prefix "$CONDA_ENV_DIR" python=3.10 --yes
    else
        echo "Conda environment $CONDA_ENV_NAME already exists at $CONDA_ENV_DIR."
    fi
}

# Function to activate Conda environment
activate_conda_env() {
    echo "Activating Conda environment $CONDA_ENV_NAME at $CONDA_ENV_DIR..."
    source "$MINICONDA_DIR/bin/activate"
    conda activate "$CONDA_ENV_DIR"
}

# Function to install Conda packages
install_conda_packages() {
    echo "Installing Conda packages..."
    conda install pytorch::faiss-gpu --yes
}

clone_protrek() {
    echo "cloning https://github.com/westlake-repl/ProTrek.git"
    git clone https://github.com/westlake-repl/ProTrek.git
}

# Function to install pip requirements
install_pip_requirements() {
    echo "Installing pip requirements..."
    pip install --no-cache-dir -r "$REPO_DIR/requirements.txt"
    pip install bioservices
    pip install biopandas
}

# Function to set permissions for foldseek
install_foldseek() {
    echo "Retrieving foldseek..."
    wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1B_9t3n_nlj8Y3Kpc_mMjtMdY0OPYa7Re' -O ProTrek/bin/foldseek
    echo "Setting execute permissions for foldseek..."
    chmod +x "$REPO_DIR/bin/foldseek"
}

fetch_weights() {
    huggingface-cli download westlake-repl/ProTrek_650M_UniRef50 \
                         --repo-type model \
                         --local-dir $REPO_DIR/weights/ProTrek_650M_UniRef50
}

# Main script execution
# install_miniconda
# ensure_conda
clone_protrek
fetch_weights
# create_conda_env
# activate_conda_env
install_pip_requirements
install_conda_packages
install_foldseek

echo "Setup complete. The ProTrek installation directory is $(realpath ProTrek)"

