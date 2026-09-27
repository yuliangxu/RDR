"""Compatibility imports for historical notebooks and source snapshots.

New code should import losses, networks, training, or diagnostics directly.
Implementations live in those modules; checkpoint class aliases remain here.
"""

from .losses import (
    Hellinger_loss,
    KL_loss,
    Chisq_loss,
    JS_loss,
)
from .networks import (
    BoundedSoftplus,
    BoundedSigmoid,
    MLP,
    ConvBlock,
    Down,
    Up,
    UNet,
    DREConvNet_DCGAN_MNIST,
    dcgan_weights_init,
    RatioNetCelebA64,
    ViewToMNIST,
)
from .training import (
    run_DRE_fdiv,
    run_DRE_fdiv_cnn,
    run_DRE_fdiv_precond,
    run_DRE_fdiv_embed,
    run_DRE_fdiv_SGD,
    run_DRE_fdiv_SGD_mix_in_loss,
    _u_from_model_out,
    _renorm_on_q,
    _model_device_dtype,
    _to_model,
    _extract_x,
    _match_batch_sizes,
    _flatten_if_needed,
    _ensure_each_side,
    _first_batch_dim,
)
from .diagnostics import (
    evaluate_model,
    evaluate_model_embed,
    _unpack_loader_batch,
)
