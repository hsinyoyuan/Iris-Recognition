import numpy as np
import cv2
from scipy import signal

class IrisSegmenter:
    """
    Iris segmentation pipeline based on circular boundary search and
    simple eyelid / eyelash masking.

    Pipeline
    --------
    1. Detect inner boundary (pupil).
    2. Detect outer boundary (iris).
    3. Detect upper and lower eyelids using Hough line segments.
    4. Detect eyelashes using black-hat morphology.
    5. Return iris boundary, pupil boundary, and a noise-masked image.

    Notes
    -----
    - This implementation uses an integro-differential style search
      for the pupil and iris boundaries.
    - Eyelids are approximated by straight lines found via HoughLinesP.
    - Noise regions are marked with `np.nan` in the returned `noise_img`.
    """

    def findTopEyelid(self, imsz, imageiris, irl, icl, rowp, rp, ret_top=None):
        """
        Detect the upper eyelid and create a mask for the region above it.

        Parameters
        ----------
        imsz : tuple
            Shape of the full original image, i.e. (H, W).
        imageiris : ndarray
            Cropped iris region.
        irl : int
            Top row index of the crop in the original image.
        icl : int
            Left column index of the crop in the original image.
        rowp : int
            Row coordinate of the pupil center in the original image.
        rp : int
            Pupil radius.
        ret_top : list-like, optional
            Optional mutable container to store the output mask.

        Returns
        -------
        mask : ndarray of bool
            Boolean mask for the upper eyelid region in full-image coordinates.
        """
        
        # Region above the pupil where the upper eyelid is expected
        topeyelid = imageiris[0: rowp - irl - rp, :]
        lines = self.findline(
            topeyelid,
            canny_low=30,
            canny_high=100,
            hough_thresh=50
        )
       
        mask = np.zeros(imsz, dtype=bool)
        if lines.size > 0:
            xl, yl = self.linecoords(lines, topeyelid.shape)

            if len(yl) == 0:
                return mask
            
            # Map from crop coordinates back to original image coordinates
            yl = np.round(yl + irl - 1).astype(int)
            xl = np.round(xl + icl - 1).astype(int)

            # Fill a band above the detected eyelid line
            yla = np.max(yl)
            max_extend_top = 20                           
            y2 = np.arange(max(0, yla - max_extend_top), yla)
            yy, xx = np.meshgrid(y2, xl, indexing='ij')
            mask[yy, xx] = True

        if ret_top is not None:
            ret_top[0] = mask
        return mask
    
    def findBottomEyelid(self, imsz, imageiris, irl, icl, rowp, rp, ret_bot=None):
        """
        Detect the lower eyelid and create a mask for the region below it.

        Parameters
        ----------
        imsz : tuple
            Shape of the full original image, i.e. (H, W).
        imageiris : ndarray
            Cropped iris region.
        irl : int
            Top row index of the crop in the original image.
        icl : int
            Left column index of the crop in the original image.
        rowp : int
            Row coordinate of the pupil center in the original image.
        rp : int
            Pupil radius.
        ret_bot : list-like, optional
            Optional mutable container to store the output mask.

        Returns
        -------
        mask : ndarray of bool
            Boolean mask for the lower eyelid region in full-image coordinates.
        """
        # Region below the pupil where the lower eyelid is expected

        bottomeyelid = imageiris[rowp - irl + rp - 1 : imageiris.shape[0], :]
        lines = self.findline(
            bottomeyelid,
            canny_low=30,
            canny_high=100,
            hough_thresh=50
        )
        mask = np.zeros(imsz, dtype=bool)
        if lines.size > 0:
            xl, yl = self.linecoords(lines, bottomeyelid.shape)

            if len(yl) == 0:
                return mask

            # Map from crop coordinates back to original image coordinates
            yl = np.round(yl + rowp + rp - 3).astype(int)
            xl = np.round(xl + icl - 2).astype(int)

            # Fill a band below the detected eyelid line
            yla = np.min(yl)
            max_extend = 20  
            y2 = np.arange(yla-1, min(yla-1 + max_extend, imsz[0]))

            yy, xx = np.meshgrid(y2, xl, indexing='ij')
            mask[yy, xx] = True

        if ret_bot is not None:
            ret_bot[0] = mask

        return mask
    
    def findline(
        self,
        img,
        canny_low=30,
        canny_high=100,
        hough_thresh=30,
        min_line_length=30,
        max_line_gap=10
    ):
        """
        Detect line segments in an image using Canny edge detection followed by
        probabilistic Hough transform.

        Parameters
        ----------
        img : ndarray
            Input grayscale image.
        canny_low : int, default=30
            Lower threshold for Canny edge detection.
        canny_high : int, default=100
            Upper threshold for Canny edge detection.
        hough_thresh : int, default=30
            Accumulator threshold for HoughLinesP.
        min_line_length : int, default=30
            Minimum accepted line length.
        max_line_gap : int, default=10
            Maximum allowed gap between points on the same line.

        Returns
        -------
        lines : ndarray of shape (N, 4)
            Each row is a line segment [x1, y1, x2, y2].
            Returns an empty array if no line is found.
        """
                
        if img.size == 0:
            return np.empty((0, 4), dtype=np.int32)

        img = cv2.GaussianBlur(img, (5, 5), 0)
        edges = cv2.Canny(img, canny_low, canny_high)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=hough_thresh,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )

        if lines is None:
            return np.empty((0, 4), dtype=np.int32)

        return lines[:, 0, :]  # shape: (N, 4), each = [x1, y1, x2, y2]
    
    def linecoords(self, lines, imsize):
        """
        Convert the selected Hough line segment into dense (x, y) coordinates
        across the image width.

        Strategy
        --------
        Among all detected line segments, choose the longest near-horizontal one,
        then evaluate its y-coordinate over the full x-range of the image.

        Parameters
        ----------
        lines : ndarray of shape (N, 4)
            Hough line segments, each represented as [x1, y1, x2, y2].
        imsize : tuple
            Image size as (H, W).

        Returns
        -------
        xd : ndarray
            Dense x coordinates across the image width.
        yd : ndarray
            Corresponding y coordinates on the selected line.
            Returns empty arrays if no suitable line is found.
        """

        if lines is None or len(lines) == 0:
            return np.array([]), np.array([])

        H, W = imsize

        best = None
        best_len = 0

        # Select the longest near-horizontal line
        for x1, y1, x2, y2 in lines:

            dx = x2 - x1
            dy = y2 - y1

            if abs(dx) < 1e-6:  # skip vertical lines
                continue

            slope = abs(dy / dx)
            length = np.hypot(dx, dy)

            if slope < 0.5 and length > best_len:
                best = (x1, y1, x2, y2)
                best_len = length

        if best is None:
            return np.array([]), np.array([])

        x1, y1, x2, y2 = best

        # ---- generate coordinates across width ----
        xd = np.arange(W)

        slope = (y2 - y1) / (x2 - x1)
        yd = y1 + slope * (xd - x1)

        yd = np.clip(yd, 0, H - 1)

        return xd.astype(int), yd.astype(int)
    

    def ContourIntegralCircular(self, im, y0, x0, r, angs):
        """
        Compute discrete circular contour integrals for multiple centers/radii.

        For each angle in `angs`, points are sampled on the corresponding circle
        and their intensities are accumulated.

        Parameters
        ----------
        im : ndarray
            Grayscale image.
        y0, x0, r : ndarray
            Arrays of the same shape describing candidate circle centers and radii.
        angs : ndarray
            Angles (in radians) used for circular sampling.

        Returns
        -------
        hs : ndarray
            Circular contour integral values with the same shape as `y0`.
        """

        hs = np.zeros_like(y0, dtype=float)
        for ang in angs:
            yi = np.round(y0 - np.cos(ang) * r).astype(int)
            xi = np.round(x0 + np.sin(ang) * r).astype(int)

            # Clamp sampled points to valid image coordinates
            np.clip(yi, 0, im.shape[0]-1, out=yi)
            np.clip(xi, 0, im.shape[1]-1, out=xi)

            hs += im[yi, xi]

        return hs

    
    def searchInnerBound(self, img):
        """
        Search for the pupil boundary using a coarse-to-fine
        integro-differential style circular search.

        Parameters
        ----------
        img : ndarray
            Input grayscale eye image.

        Returns
        -------
        final_y : int
            Estimated pupil center row.
        final_x : int
            Estimated pupil center column.
        final_r : int
            Estimated pupil radius.
        """

        Y, X = img.shape

        # Coarse search configuration
        sect = X / 4               # Starting offset
        minrad = 10                # Minimum radius for search
        maxrad = sect * 0.8        # Maximum radius for search
        jump = 4                   # Step size for grid


        sz_y = int((Y - 2 * sect) // jump)
        sz_x = int((X - 2 * sect) // jump)
        sz_r = int((maxrad - minrad) // jump)


        angs = np.arange(0, 2 * np.pi, 1)

        # Candidate centers and radii for coarse search
        yg, xg, rg = np.meshgrid(
            np.arange(sz_y),
            np.arange(sz_x),
            np.arange(sz_r),
            indexing="ij"
        )
        yg = sect + yg * jump
        xg = sect + xg * jump
        rg = minrad + rg * jump


        hs = self.ContourIntegralCircular(img, yg, xg, rg, angs)

        # Approximate derivative along radius dimension
        hspdr = hs - hs[:, :, np.insert(np.arange(hs.shape[2] - 1), 0, 0)]

        # Smooth response volume
        hspdrs = signal.fftconvolve(hspdr, np.ones((3, 3, 3)), mode='same')

        ind = np.argmax(hspdrs)
        y0, x0, r0 = np.unravel_index(ind, hspdrs.shape)

        
        inner_y = int(round(sect + y0 * jump))
        inner_x = int(round(sect + x0 * jump))
        inner_r = int(round(minrad + (r0 - 1) * jump))

        # Fine search around coarse estimate
        angs2 = np.arange(0, 2 * np.pi, 0.1)
        offsets = np.arange(2 * jump)

        x2, y2, r2 = np.meshgrid(offsets, offsets, offsets)
        y2 = inner_y - jump + y2
        x2 = inner_x - jump + x2
        r2 = inner_r - jump + r2


        hs2 = self.ContourIntegralCircular(img, y2, x2, r2, angs2)
        hspdr2 = hs2 - hs2[:, :, np.insert(np.arange(hs2.shape[2] - 1), 0, 0)]
        hspdrs2 = signal.fftconvolve(hspdr2, np.ones((3, 3, 3)), mode='same')


        ind2 = np.argmax(hspdrs2)
        y1, x1, r1 = np.unravel_index(ind2, hspdrs2.shape)


        final_y = inner_y - jump + y1
        final_x = inner_x - jump + x1
        final_r = inner_r - jump + r1 - 1

        return final_y, final_x, final_r
        ...

    def searchOuterBound(self, img, inner_y, inner_x, inner_r):
        """
        Search for the outer iris boundary using an integro-differential style
        circular search centered around the detected pupil.

        Parameters
        ----------
        img : ndarray
            Input grayscale image.
        inner_y : int
            Pupil center row.
        inner_x : int
            Pupil center column.
        inner_r : int
            Pupil radius.

        Returns
        -------
        y_outer : int
            Estimated iris center row.
        x_outer : int
            Estimated iris center column.
        r_outer : int
            Estimated iris radius.
        """


        delta   = int(round(inner_r * 0.15))        # neighborhood around inner center
        min_rad = int(round(inner_r * 3))         # minimum iris radius
        max_rad = int(round(inner_r * 6))         # maximum iris radius

        # Avoid angular sectors likely occluded by eyelids
        angular_ranges = np.array([[2/6, 4/6], [8/6, 10/6]]) * np.pi
        angles1 = np.arange(angular_ranges[0,0], angular_ranges[0,1], 0.05)
        angles2 = np.arange(angular_ranges[1,0], angular_ranges[1,1], 0.05)
        angles = np.concatenate([angles1, angles2])


        offsets = np.arange(0, 2 * delta)

        Yg, Xg, Rg = np.meshgrid(offsets, offsets, np.arange(max_rad - min_rad), indexing='xy')
        Yg = inner_y - delta + Yg
        Xg = inner_x - delta + Xg
        Rg = min_rad + Rg


        H = self.ContourIntegralCircular(img, Yg, Xg, Rg, angles)

        # Approximate derivative along radius dimension
        H_diff = H - H[:, :, np.insert(np.arange(H.shape[2]-1), 0, 0)]

        # Smooth response volume
        smoothing_kernel = np.ones((7, 7, 7))
        H_smooth = signal.fftconvolve(H_diff, smoothing_kernel, mode='same')


        idx = np.argmax(H_smooth)
        y_idx, x_idx, r_idx = np.unravel_index(idx, H_smooth.shape)


        y_outer = inner_y - delta + y_idx + 1
        x_outer = inner_x - delta + x_idx + 1
        r_outer = min_rad + r_idx - 1

        return y_outer, x_outer, r_outer

    def segment(self, img):
        """
        Run full iris segmentation.

        Parameters
        ----------
        img : ndarray
            Input grayscale eye image.

        Returns
        -------
        iris_boundary : tuple
            (row, col, r) for the outer iris boundary.
        pupil_boundary : tuple
            (rowp, colp, rp) for the pupil boundary.
        noise_img : ndarray
            Float image where masked noise pixels are set to `np.nan`.
        """
        # Step 1: detect pupil boundary
        rowp, colp, rp = self.searchInnerBound(img)
        rowp, colp, rp = map(int, map(round, (rowp, colp, rp)))

        # Step 2: detect iris
        row, col, r = self.searchOuterBound(img, rowp, colp, rp)
        row, col, r = map(int, map(round, (row, col, r)))

        # Step 3: crop around detected iris
        irl = max(0, row - r)
        iru = min(img.shape[0]-1, row + r)
        icl = max(0, col - r)
        icu = min(img.shape[1]-1, col + r)
        crop = img[irl:iru+1, icl:icu+1]

        # Step 4: eyelid masks
        mask_top = self.findTopEyelid(img.shape, crop, irl, icl, rowp, rp)
        mask_bot = self.findBottomEyelid(img.shape, crop, irl, icl, rowp, rp)

        # Step 5: initialize noise image
        noise_img = img.astype(float)
        noise_img[mask_top] = np.nan
        noise_img[mask_bot] = np.nan

        # Step 6: eyelash masking via black-hat morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

        mu, sigma = blackhat.mean(), blackhat.std()
        eyelash_mask = blackhat > (mu + 1.5 * sigma)
        noise_img[eyelash_mask] = np.nan

        iris_boundary = (row, col, r)
        pupil_boundary = (rowp, colp, rp)

        return iris_boundary, pupil_boundary, noise_img