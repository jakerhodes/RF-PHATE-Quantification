# Unsupervised
from .phate import PHATE
from .phatet import PHATET
from .dm import DM
from .mds import MDS
from .tsne import TSNE
from .lle import LLE
from .isomap import ISOMAP
from .umap import UMAP
from .lapeig import LAPEIG
from .kpca import KPCA
from .pca import PCA
from .ucebra import UCEBRA

# Supervised
from .eslle import ESLLE
from .esisomap import ESISOMAP
from .sumap import SUMAP
from .estsne import ESTSNE
from .nca import NCA
from .snmf import SNMF
from .plsda import PLSDA
from .slapeig import SLAPEIG
from .spca import SPCA
from .kspca import KSPCA
from .cebra import CEBRA
from .sae import SAE
from .ssnp import SSNP
from .ce import CE


# RF-Based
from .rfphate import RFPHATE
from .rfdm import RFDM
from .rfkpca import RFKPCA
from .rfisomap import RFISOMAP
from .rfumap import RFUMAP
from .rfmds import RFMDS
from .rftsne import RFTSNE
from .rflapeig import RFLAPEIG

unsupervised_methods = [
    PHATE,
    PHATET,
    DM,
    MDS,
    TSNE,
    LLE,
    ISOMAP,
    UMAP,
    LAPEIG,
    KPCA,
    PCA,
    UCEBRA
]


supervised_methods = [
    ESTSNE,
    ESLLE,
    ESISOMAP,
    SUMAP,
    NCA,
    SNMF,
    PLSDA,
    SPCA,
    KSPCA,
    CEBRA,
    SAE,
    SSNP,
    CE
]

rf_methods = [
    RFPHATE,
    RFDM,
    RFMDS,
    RFTSNE,
    RFISOMAP,
    RFUMAP,
    RFKPCA,
    RFLAPEIG
]

regression_methods = unsupervised_methods + rf_methods + [PLSDA, CEBRA]