# Target
This repository is for our conference paper, now it is anonymous review.

# Environment
`pip install -r requirements.txt`
or
`conda env create -f environment.yml`
# Usage

`python main.py`

For version use attention memory module in our model, just verify `use_mem_network=True` in `main.py`. For the version with ability to class-incremental relation extraction, just verify `fix_labels=False` in `main.py`.

# Parameters

We provide default parameters with `config-tacred.ini` corresponding to **TACRED**, `config-fewrel.ini` corresponding to **FewRel**, change filename to `config.ini` and add your BERT path and run `python main.py` to reproduce results.# CRECL
