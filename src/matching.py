import numpy as np

class IrisMatcher:

    def shiftbits_ham(self, tmpl, sh):
        """
        Circularly shift iris template along the angular direction.

        Parameters
        ----------
        tmpl : ndarray
            Iris template or mask.

        sh : int
            Number of bits to shift.

        Returns
        -------
        shifted template
        """

        return np.roll(tmpl, shift=int(sh), axis=1)

    def HammingDistance(self, t1, m1, t2, m2):
        """
        Compute the Hamming distance between two iris templates.

        The templates are circularly shifted to compensate for eye rotation,
        and the minimum Hamming distance across all shifts is returned.

        Parameters
        ----------
        t1, t2 : ndarray
            Iris templates (binary codes).

        m1, m2 : ndarray
            Corresponding noise masks.

        Returns
        -------
        best : float
            Minimum Hamming distance across all shifts.
        """
        
        best = np.nan

        # search over possible rotational shifts
        for sh in np.arange(-12, 12.5, 0.5):

            # rotate template and mask
            t1s = self.shiftbits_ham(t1, sh)
            m1s = self.shiftbits_ham(m1, sh)

            # valid bits are those not masked in either template
            valid = ~(m1s | m2)

            # XOR difference between templates
            diff = np.logical_xor(t1s, t2) & valid

            total = np.sum(valid)

            if total > 0:

                hd = np.sum(diff) / total

                if np.isnan(best) or hd < best:
                    best = hd
                    
        return best