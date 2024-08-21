# Installation

## Method 1: Install Using pip
We recommend using conda to manage your environment and installing dependencies with pip:

1. Create a conda environment (optional)
   ```shell
   # It is recommended to use Python 3.10
   conda create -n evalscope python=3.10
   # Activate the conda environment
   conda activate evalscope
   ```

2. Install dependencies using pip
   ```shell
   pip install evalscope                # Install Native backend (default)
   # Additional options
   pip install evalscope[opencompass]   # Install OpenCompass backend
   pip install evalscope[vlmeval]       # Install VLMEvalKit backend
   pip install evalscope[all]           # Install all backends (Native, OpenCompass, VLMEvalKit)
   ```

````{warning}
As the project has been renamed to `evalscope`, for versions `v0.4.3` or earlier, you can install using the following command:
```shell
pip install llmuses<=0.4.3
```
To import relevant dependencies using `llmuses`:
``` python
from llmuses import ...
```
````

## Method 2: Install from Source
1. Download the source code
   ```shell
   git clone https://github.com/modelscope/evalscope.git
   ```

2. Install dependencies
   ```shell
   cd evalscope/
   pip install -e .                  # Install Native backend
   # Additional options
   pip install -e '.[opencompass]'   # Install OpenCompass backend
   pip install -e '.[vlmeval]'       # Install VLMEvalKit backend
   pip install -e '.[all]'           # Install all backends (Native, OpenCompass, VLMEvalKit)
   ```