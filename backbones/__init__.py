from .SubNets.FeatureNets import BERTEncoder
from .FusionNets.MAG_BERT import MAG_BERT
from .FusionNets.MISA import MISA
from .FusionNets.MULT import MULT
from .FusionNets.TCS_Mamba import TCS_Mamba
from .FusionNets.DLF import DLF
from .FusionNets.DDSE import DDSE

text_backbones_map = {
                    'bert-base-uncased': BERTEncoder
                }

methods_map = {
    'mag_bert': MAG_BERT,
    'misa': MISA,
    'mult': MULT,
    'tcs_mamba': TCS_Mamba,
    'dlf': DLF,
    'ddse': DDSE,
}