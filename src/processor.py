from .feature_extraction import IrisFeatureExtractor
from .matching import IrisMatcher


class IrisProcessor:
    """
    High-level iris recognition interface.

    This class combines feature extraction and matching
    to compute similarity between two iris images.
    """

    def __init__(self, dataset_name:"Lamp"):
        self.dataset_name = dataset_name

        # Feature extraction pipeline
        self.extractor = IrisFeatureExtractor()

        # Template matcher
        self.matcher = IrisMatcher()

    def compute_score(self, img1_path, img2_path):
        """
        Compute the similarity score between two iris images.

        Parameters
        ----------
        img1_path : str
            Path to the first iris image.

        img2_path : str
            Path to the second iris image.

        Returns
        -------
        float
            Hamming distance between two iris templates.
            Lower values indicate higher similarity.
        """

        # Extract iris features from both images
        tmpl1, msk1= self.extractor.extractFeature(img1_path)
        tmpl2, msk2= self.extractor.extractFeature(img2_path)

        # Compute Hamming distance
        return self.matcher.HammingDistance(tmpl1, msk1, tmpl2, msk2)
