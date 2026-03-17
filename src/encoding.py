import numpy as np

class IrisEncoder:

    def gaborconvolve_f(self, img, minw, mult, sigma_f):
        """
        Apply a 1D log-Gabor filter to each row of the normalized iris image.

        Parameters
        ----------
        img : ndarray (rows, ndata)
            Polar-normalized iris image.
            Each row corresponds to one angular band of the iris.

        minw : float
            Minimum wavelength of the log-Gabor filter.
            Controls the center frequency.

        mult : float
            Scale multiplier (not used here but kept for compatibility
            with multi-scale filter implementations).

        sigma_f : float
            Bandwidth parameter of the log-Gabor filter.

        Returns
        -------
        fb : ndarray (rows, ndata), complex
            Complex filter responses for each row.
        """

        rows, ndata = img.shape

        # Construct log-Gabor filter in the frequency domain
        logGabor = np.zeros(ndata)

        # Create normalized frequency radius
        radius = np.arange(ndata//2 + 1) / (ndata/2) / 2
        radius[0] = 1

        # Center frequency
        fo = 1.0 / minw

        # Log-Gabor frequency response
        logGabor[:ndata//2 + 1] = np.exp(-((np.log(radius/fo))**2)
                                        / (2 * (np.log(sigma_f)**2)))
        
        # Remove DC component
        logGabor[0] = 0

        # Prepare output array
        fb = np.zeros((rows, ndata), dtype=complex)

        # Apply filter row-by-row in frequency domain
        for i in range(rows):

            # FFT of iris row
            row_fft = np.fft.fft(img[i, :])

            # Frequency filtering then inverse FFT
            fb[i, :] = np.fft.ifft(row_fft * logGabor)

        return fb

    def encode_gabor(self, polar, noise, minw, sigma_f):
        """
        Generate iris code using log-Gabor filter responses.

        The complex filter response is converted to binary bits
        using the sign of the real and imaginary components.

        Parameters
        ----------
        polar : ndarray (rows, cols)
            Polar-normalized iris image.

        noise : ndarray (rows, cols)
            Noise mask indicating occluded regions
            (eyelids, eyelashes, reflections).

        minw : float
            Minimum wavelength of the log-Gabor filter.

        sigma_f : float
            Bandwidth of the filter.

        Returns
        -------
        tmpl : ndarray (rows, 2*cols), bool
            Binary iris template.

        mask : ndarray (rows, 2*cols), bool
            Corresponding noise mask.
        """
        
        # Apply log-Gabor filtering
        fb = self.gaborconvolve_f(polar, minw, 1, sigma_f)
        rows, l = polar.shape

        # Two bits are generated per pixel (real + imaginary)
        tmpl = np.zeros((rows, 2 * l), dtype=bool)
        mask = np.zeros_like(tmpl)

        # Binary encoding of complex response
        H1 = np.real(fb) > 0
        H2 = np.imag(fb) > 0

        # Low magnitude responses are unreliable
        H3 = np.abs(fb) < 1e-4

        for i in range(l):

            # Bit 1: sign of real component
            tmpl[:, 2 * i] = H1[:, i]

            # Bit 2: sign of imaginary component
            tmpl[:, 2 * i + 1] = H2[:, i]

            # Mask invalid regions
            mask[:, 2 * i] = noise[:, i] | H3[:, i]
            mask[:, 2 * i + 1] = noise[:, i] | H3[:, i]
        return tmpl, mask



    def multi_encode_iris(self, polar, noise):
        """
        Multi-scale iris encoding.

        Multiple log-Gabor filters with different center frequencies
        are applied to capture texture information at different scales.

        The resulting binary codes are concatenated.

        Parameters
        ----------
        polar : ndarray
            Polar-normalized iris image.

        noise : ndarray
            Noise mask.

        Returns
        -------
        template : ndarray
            Concatenated iris code from multiple filter scales.

        mask : ndarray
            Combined noise mask.
        """

        # Multi-scale filter settings
        settings = [(18, 0.5), (9, 0.6), (27, 0.4)]
        tmpl_all, mask_all = [], []
        for minw, sigma in settings:
            tmpl, mask = self.encode_gabor(polar, noise, minw, sigma)

            tmpl_all.append(tmpl)
            mask_all.append(mask)

        # Concatenate features from different scales
        return (
            np.concatenate(tmpl_all, axis=1), 
            np.concatenate(mask_all, axis=1)
        )