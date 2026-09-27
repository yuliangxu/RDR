#!/usr/bin/env python3
"""CelebA pixel RDR calibration using the shared frozen-score C.1/C.2 workflow."""

from CELEBA_feature_ci import main


if __name__ == "__main__":
    main(default_level="pixel")
