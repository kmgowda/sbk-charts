# Custom AI Implementations
This directory contains custom AI implementations for the sbk-charts application.
Each implementation should inherit from the SbkGenAI base class and implement the required methods.

At present, The Hugging Face and NoAI implementations are available.

The Hugging Face implementation uses the Hugging Face API to generate the AI analysis.
make sure that you have the Hugging Face API key set in the environment variable HUGGING_FACE_API_KEY.

The NoAI implementation returns error messages for all analysis methods.

The example command to use the Hugging Face implementation is:

```
(sbk-charts-venv) kmg@Mac-Studio sbk-charts % sbk-charts -i ./samples/charts/sbk-file-read.csv,./samples/charts/sbk-rocksdb-read.csv huggingface

   _____   ____    _  __            _____   _    _              _____    _______    _____
  / ____| |  _ \  | |/ /           / ____| | |  | |     /\     |  __ \  |__   __|  / ____|
 | (___   | |_) | | ' /   ______  | |      | |__| |    /  \    | |__) |    | |    | (___
  \___ \  |  _ <  |  <   |______| | |      |  __  |   / /\ \   |  _  /     | |     \___ \
  ____) | | |_) | | . \           | |____  | |  | |  / ____ \  | | \ \     | |     ____) |
 |_____/  |____/  |_|\_\           \_____| |_|  |_| /_/    \_\ |_|  \_\    |_|    |_____/

Sbk Charts Version : 3.26.X.0
Input Files :  ./samples/charts/sbk-file-read.csv,./samples/charts/sbk-rocksdb-read.csv
Output File :  out.xlsx
SBK logo image found: /Users/kmg/projects/sbk-charts/images/sbk-logo.png
xlsx file : out.xlsx created
Time Unit : NANOSECONDS
Reading : FILE, ROCKSDB
file : out.xlsx updated with graphs
AI analysis. Please wait....
Completed 1/4 tasks
Completed 2/4 tasks
Completed 3/4 tasks
Completed 4/4 tasks
File updated with graphs and AI documentation: out.xlsx
```
The -help option can be used to get the help message for the Hugging Face implementation.

```
sbk-charts-venv) kmg@Mac-Studio sbk-charts % sbk-charts -i ./samples/charts/sbk-file-read.csv,./samples/charts/sbk-rocksdb-read.csv huggingface -help
usage: sbk-charts huggingface [-h] [-id MODEL_ID]

options:
  -h, --help            show this help message and exit
  -id, --model_id MODEL_ID
                        Hugging Face model ID; default model: google/gemma-2-2b-it
```


The example command to use the NoAI implementation is:

```
sbk-charts -i ./samples/charts/sbk-file-read.csv,./samples/charts/sbk-rocksdb-read.csv noai
```

