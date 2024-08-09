
# SAILSS: The "Scouting AI Library for Social Sciences"

## Estimate the likelihood of sentences (or parts of sentences) based on a Large Language Model (LLM)

This library is described in a series of posts on [my blog](https://lardel.li/2024/07/llm_language_model_library_social_sciences.html)

**Important note: this library is work in progress and still in a raw state.**

## Installation

This library needs the [MLX Examples](https://github.com/ml-explore/mlx-examples) to be installed:

```sh
pip install mlx-lm
```

Then you can download SAILSS (this repository):
```sh
git clone https://github.com/MarcoLardelli/sailss
```


## Downloading a Large Language Model (LLM)

mlx-lm supports a large number of Hugging Face large language models. You need to donwload at least one model from there.

- You must first create an account on [huggingface.co](https://huggingface.co)
- You must create a (read only) access token to be able to download models
- For some models you might have to accept a license agreement and/or provide your use case (sometimes you will have to wait 24hours for approval)
- You must install the huggingface comand line interface:

```sh
pip install -U "huggingface_hub[cli]"
```

Then you can log in to huggingface:
```sh
huggingface-cli login
```
Now you can download the model files:

```sh
git clone https://huggingface.co/mistralai/Meta-Llama-3-8B
```
(git will need the [git-lfs extension](https://git-lfs.com) for large files (download, run ./install.ch with sudo))

save the model folder into the *./models* directory.

Note:
**Models to be used for SAILSS do not need to be instruction tuned.**   
Models without instruction tuning might perform better.

# Documentation

The HTML documentation in the *docs* folder was created using [pdoc](https://pdoc.dev):

```sh
pdoc ./sailss.py -o ./docs
```

It can be found also on the corresponding [github pages page](https://marcolardelli.github.io/sailss/) of this repository.

# Examples

You can run the examples in the *examples* folder from the SAILSS directory:

```sh
python3 -m examples.example_1
```

Author: Marco Lardelli, Zurich/Switzerland   
Licence: MIT (see included license document)   
Copyright © 2024 Marco Lardelli   
