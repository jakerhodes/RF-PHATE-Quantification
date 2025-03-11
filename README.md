## RF-PHATE-Quantification

This repository provides the datasets and methodology used to quantify the embeddings in Rhodes, J.S., Aumon, A., Morin, S., et al.: Gaining Biological Insights through Supervised Data Visualization. *bioRxiv* (2024). [https://doi.org/10.1101/2023.11.22.568384](https://doi.org/10.1101/2023.11.22.568384).

To generate the raw results, download this repository, install the packages in the requirements.txt file, and run the `quantify_embeddings.py` module. Results will be stored in the *results* directory.

The noise_plots.py file is used to generate the basic scatterplots contrasting embeddings with and without added noise. 

`Note`: The random seeds used here are different from those used in the paper.  Those states were set via a computer time clock. Results will slightly vary from those presented in the paper.

Link to Code Repository: [RF-PHATE](https://github.com/jakerhodes/RF-PHATE)
