# Custom AI Implementations
This directory contains custom AI implementations for the sbk-charts application.
Each implementation should inherit from the SbkGenAI base class and implement the required methods.

An example implementation of hugging face is here : [Hugging face](./hugging_face/README.md) directory


The example command to use the NoAI implementation is:

```
sbk-charts -i ./samples/charts/sbk-file-read.csv,./samples/charts/sbk-rocksdb-read.csv noai
```

The Custom AI can be extended by creating a new class that inherits from the SbkGenAI base class and implements the required methods.
As of today only Hugging Face and NoAI implementations are available. But, the custom AI can be extended to use other AI models.

