import numpy as np

class IrisNormalizer:

    def normalize(self, image, x_i, y_i, r_i, x_p, y_p, r_p, radpixels, angdiv):
        """
        Perform iris normalization using Daugman's rubber-sheet model.

        The circular iris region is unwrapped into a rectangular polar
        representation where:
            - rows correspond to radial positions
            - columns correspond to angular positions

        Parameters
        ----------
        image : ndarray
            Grayscale iris image.

        x_i, y_i, r_i : int
            Center (x, y) and radius of the iris boundary.

        x_p, y_p, r_p : int
            Center (x, y) and radius of the pupil boundary.

        radpixels : int
            Number of radial samples (resolution in radial direction).

        angdiv : int
            Number of angular divisions (resolution around the iris).

        Returns
        -------
        polar : ndarray
            Polar-normalized iris image.

        noise : ndarray (bool)
            Noise mask indicating unreliable pixels.
        """

        # ---------------------------------------------------------
        # 1. Generate polar coordinate sampling grid
        # ---------------------------------------------------------

        rp = radpixels + 2
        ad = angdiv - 1

        # radial positions from pupil boundary → iris boundary
        r = np.linspace(r_p, r_i, rp)[:, None]

        # angular positions around the iris
        theta = np.linspace(0, 2*np.pi, ad+1)

        # convert polar coordinates to Cartesian coordinates
        xs = np.round(x_p + r * np.cos(theta)).astype(int)
        ys = np.round(y_p - r * np.sin(theta)).astype(int)

        # ---------------------------------------------------------
        # 2. Clamp coordinates to image boundaries
        # ---------------------------------------------------------
        xs = np.clip(xs, 0, image.shape[1]-1)
        ys = np.clip(ys, 0, image.shape[0]-1)

        # ---------------------------------------------------------
        # 3. Sample image intensities
        # ---------------------------------------------------------
        polar = image[ys, xs] / 255.0

        # detect invalid samples
        noise = np.isnan(polar)

        # replace NaN values with mean intensity
        if np.any(noise):
            polar[noise] = np.nanmean(polar)

        # ---------------------------------------------------------
        # 4. Detect eyelash occlusions in polar domain
        # ---------------------------------------------------------

        # compute column statistics
        col_mean = np.nanmean(polar, axis=0)
        col_std  = np.nanstd(polar, axis=0)

        # eyelash regions are typically
        #   - very dark
        #   - high variance
        m_low = np.percentile(col_mean, 5)
        m_high = np.percentile(col_std, 95)

        bad_cols = np.where((col_mean < m_low) & (col_std > m_high))[0]

        # build mask for these columns
        mask2 = np.zeros_like(polar, dtype=bool)
        mask2[:, bad_cols] = True

        # update noise mask
        noise = noise | mask2
        
        # replace masked pixels with mean intensity
        polar[mask2] = np.nanmean(polar)

        return polar, noise