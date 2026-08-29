from .TEXT.manager import TEXT
from .MISA.manager import MISA
from .MULT.manager import MULT
from .MAG_BERT.manager import MAG_BERT
from .TCS_Mamba.manager import TCS_Mamba
from .DLF.manager import DLF
from .DDSE.manager import DDSE

method_map = {
    'text': TEXT,
    'misa': MISA,
    'mult': MULT,
    'mag_bert': MAG_BERT,
    'tcs_mamba': TCS_Mamba,
    'dlf': DLF,
    'ddse': DDSE
}
