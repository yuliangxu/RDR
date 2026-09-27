"""Repository configuration and default immutable/input and writable/output roots."""

from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SOURCE = Path("/cwork/yx306/RDR/JRSSB/ddim_diffusion_stylegan2_equal_fid")
REFERENCE = Path("/cwork/yx306/RDR/JRSSB/real_ddim_fid_matched")
DEFAULT_WORK = Path("/cwork/yx306/RDR/JRSSB/agent3_reproduction")
