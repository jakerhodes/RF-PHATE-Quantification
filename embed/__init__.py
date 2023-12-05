# Unsupervised
from .phate import PHATE
from .dm import DM
from .mds import MDS
from .tsne import TSNE
from .lle import LLE
from .isomap import ISOMAP
from .umap import UMAP
from .lapeig import LAPEIG
from .kpca import KPCA
from .pca import PCA

# Supervised
from .eslle import ESLLE
from .esisomap import ESISOMAP
from .sumap import SUMAP
from .stsne import STSNE
from .nca import NCA
from .snmf import SNMF
from .plsda import PLSDA
from .spca import SPCA
from .kspca import KSPCA

# RF-Based
from .rfphate import RFPHATE
from .rfisomap import RFISOMAP
from .rfumap import RFUMAP
from .rfmds import RFMDS
from .rftsne import RFTSNE

unsupervised_methods = [
    PHATE,
    DM,
    MDS,
    TSNE,
    LLE,
    ISOMAP,
    UMAP,
    LAPEIG,
    KPCA,
    PCA
]


supervised_methods = [
    STSNE,
    ESLLE,
    ESISOMAP,
    SUMAP,
    NCA,
    SNMF,
    PLSDA,
    SPCA,
    KSPCA
]

rf_methods = [
    RFPHATE,
    RFMDS,
    RFTSNE,
    RFISOMAP,
    RFUMAP
]
