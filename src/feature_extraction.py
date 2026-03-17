import cv2

from .segmentation import IrisSegmenter
from .normalization import IrisNormalizer
from .encoding import IrisEncoder

class IrisFeatureExtractor:
    """
    High-level iris feature extraction pipeline.

    This class performs the complete iris recognition preprocessing:
    
        1. Iris segmentation
        2. Iris normalization (rubber sheet model)
        3. Iris feature encoding (log-Gabor filters)

    The output is a binary iris template and its corresponding noise mask.
    """

    def __init__(self):
        self.segmenter = IrisSegmenter()
        self.normalizer = IrisNormalizer()
        self.encoder = IrisEncoder()

    def extractFeature(self, path, eyelashes_threshold=80, multiprocess=True):
        """
        Extract iris features from an input image.

        Parameters
        ----------
        path : str
            Path to the iris image.

        eyelashes_threshold : int
            Threshold used for detecting eyelashes during segmentation.

        multiprocess : bool
            Whether to enable multiprocessing in segmentation.

        Returns
        -------
        tmpl : ndarray
            Binary iris template.

        msk : ndarray
            Noise mask indicating unreliable bits.

        path : str
            Image path (returned for bookkeeping).
        """
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        # Step 1: Iris segmentation
        # Detect pupil boundary, iris boundary, and noise regions

        ciriris, cirpupil, noise_img = self.segmenter.segment(
            img
        )

        # Step 2: Polar normalization
        # Convert iris region into polar coordinates
        # (Daugman's rubber sheet model)
        polar, noise = self.normalizer.normalize(
            noise_img,
            ciriris[1], ciriris[0], ciriris[2],     # iris circle
            cirpupil[1], cirpupil[0], cirpupil[2],  # pupil circle
            20,                                     # radial resolution
            240                                     # angular resolution
        )

        # Step 3: Iris encoding
        # Apply log-Gabor filters and quantize phase information
        tmpl, msk = self.encoder.multi_encode_iris(polar, noise)

        return tmpl, msk