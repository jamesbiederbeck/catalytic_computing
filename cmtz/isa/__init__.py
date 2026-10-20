"""cmtz-core: a reusable processor for Cook-Mertz programs.

encoding  the ISA (opcodes, word layout); single source for every tool
asm       assembler for text programs and the runtime library in lib/
image     program image and core configuration
sim       instruction-level simulator, the reference for the RTL
"""

from .encoding import ISA_VERSION, Instr, encode, decode
from .asm import assemble, Assembled, AsmError
from .image import CoreConfig, ProgramImage, MatrixSlot
from .sim import Machine, SimError, run_image

__all__ = [
    "ISA_VERSION", "Instr", "encode", "decode",
    "assemble", "Assembled", "AsmError",
    "CoreConfig", "ProgramImage", "MatrixSlot",
    "Machine", "SimError", "run_image",
]
