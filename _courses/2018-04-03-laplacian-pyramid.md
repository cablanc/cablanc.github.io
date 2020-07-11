---
title: Laplacian Pyramids
tags: [Machine Perception]
layout: post
---

These notes cover the construction and theory of Gaussian and Laplacian pyramids, and the SIFT detector/descriptor.
Multiresolution representations such as image pyramids were introduced primarily to improve the computational costs of pattern analysis
and image matching [@crowley2002fast](https://goo.gl/QGupMh). With a multiresolution representation structures of different scales can be analyzed with 
a filter of the same scale. The image pyramid is often credited with improving the computational efficiency of 
pattern matching since it facilitates a coarse-to-fine search strategy; search begins at the coarsest representation (the top of the pyramid) 
and is refined at subsequent pyramid level.

# The Gaussian pyramid<a id="sec-2" name="sec-2"></a>

The Gaussian pyramid is a multiresolution image representation with images in sequential pyramid levels, \\(p_l(x,y)\\) and \\(p_{l+1}(x,y)\\) 
related by 

$$
\tilde{p}_{l+1} = p_l * g_{\sigma}\\
p_{l+1} = \tilde{p}_{l+1}(2x, 2y)
$$

where \\(g_{\sigma}\\) is a 2-dimensional Gaussian filter.

In the continuous setting a scale equivariant pyramid can be constructed with any filter; however, unless the image is bandlimited and 
the filter is carefully chosen the compuational cost can be prohibitive [@crowley2002fast](https://goo.gl/QGupMh). The Gaussian filter is particularly nice 
for this task since it is separable, and convolving it with itself is akin to scaling.

In the Gaussian pyramid implementation of [@burt1983laplacian](https://goo.gl/U4YjLK), the Gaussian filter is approximated by a normalized and symmetric
equivalent weighting function. An equivalent weighting function is chosen to ensure equal contribution, i.e. all nodes at a given layer
contribute equally to the construction of the subsequent layer [@burt1983laplacian](https://goo.gl/U4YjLK). A length 5 filter is equivalent weighting function
if it has the following structure: 

$$
\begin{bmatrix}
c & b & a & b & c 
\end{bmatrix}
$$

where \\(b=1/4\\) and \\(c = b - a/2\\).

![img](../figures/equi_weight_fcn.png "Adapted from  [@burt1983laplacian](https://goo.gl/U4YjLK)")

The best approximation to the Gaussian is achieved with \\(a=0.4\\). The resulting six level Gaussian pyramid is shown below.

```python
    import matplotlib.pylab as plt
    import numpy as np
    
    from skimage import data
    from scipy.signal import convolve2d
    
    def disp_fmt_pyr(pyr):
        """
        Visualize the Gaussian pyramid
        """
        num_levels = len(pyr)
    
        H, W = pyr[0].shape
    
        img_heights = [ H * 2**(-i) for i in np.arange(num_levels) ]
        H = np.int(np.sum( img_heights ))
    
        out = np.zeros((H, W))
    
        for i in np.arange(num_levels):
            rstart = np.int(np.sum(img_heights[:i]))
            rend = np.int(rstart + img_heights[i])
    
            out[rstart:rend, :np.int(img_heights[i])] = pyr[i]
    
        return out
    
    
    def gauss_pyr(img, levels=6):
        """
        Compute the Gaussian pyramid
    
        Inputs:
        - img: Input image of size (N,M)
        - levels: Number of stages for the Gaussian pyramid
    
        Returns:
        A tuple of levels images 
        """
    
        # approximate length 5 Gaussian filter using binomial filter
        a = 0.4
        b = 1./4
        c = 1./4 - a/2
    
        filt =  np.array([[c, b, a, b, c]])
    
        # approximate 2D Gaussian
        # filt = convolve2d(filt, filt.T)
    
        pyr = [img]
    
        for i in np.arange(levels):
            # zero pad the previous image for convolution
            # boarder of 2 since filter is of length 5
            p_0 = np.pad( pyr[-1], (2,), mode='constant' )
    
            # convolve in the x and y directions to construct p_1
            p_1 = convolve2d( p_0, filt, 'valid' )
            p_1 = convolve2d( p_1, filt.T, 'valid' )
    
            # DoG approximation of LoG
            pyr.append( p_1[::2,::2] )
    
        return pyr
    
    
    camera = data.camera()
    pyr = gauss_pyr(camera)
    img = disp_fmt_pyr(pyr)
    
    plt.imshow(img)
    plt.savefig('figures/gauss_pyr.png', bbox_inches="tight")
    
    return 'figures/gauss_pyr.png'
```   

![img](../figures/gauss_pyr.png "Gaussian pyramid")

## A closer look<a id="sec-2-1" name="sec-2-1"></a>

For simplicity we consider first the pyramid constructed without subsampling, i.e.
the pyramid with adjacent pyramid levels related by the following

$$
\tilde{p}_{l+1} = p_l * g_{\sigma}\\
p_{l+1} = \tilde{p}_{l+1}
$$

We arrive at image \\(p_{l}\\) by convolving the original image with \\(g_{\sigma}\\) \\(l\\) times,

$$
p_l = p_0 * \underbrace{g_\sigma \dots * g_\sigma}_l
$$

### What is the "effective" \\(\sigma\\) at level \\(l\\)?<a id="sec-2-1-1" name="sec-2-1-1"></a>

Convolving a Gaussian with itself \\(l\\) times produces a new Gaussian \\(g_{\sigma'}\\) with 
\\(\sigma' = \sqrt{\underbrace{\sigma^2 + \dots + \sigma^2}_l}\\) (we call \\(\sigma'\\) the "effective" \\(\sigma\\)). 
This is easily shown if we use the Fourier transform,

$$
g_\sigma * \dots * g_\sigma = \mathcal{F}^{-1}\mathcal{F}\left( g_\sigma * \dots * g_\sigma\right) \\
= \mathcal{F}^{-1}\left(\mathcal{F} g_\sigma \dots \mathcal{F} g_\sigma\right).
$$

Since 

$$
\mathcal{F} g_\sigma = e^{-\sigma^2\omega^2/2},
$$

(see the proof in [@derpanis2005fourier](https://goo.gl/TvJTof), its a one page document)

$$
\mathcal{F}^{-1}\left(\mathcal{F} g_\sigma \dots \mathcal{F} g_\sigma\right) = 
\mathcal{F}^{-1}\left( e^{-\sigma^2\omega^2/2} \dots e^{-\sigma^2\omega^2/2}\right)\\
= \mathcal{F}^{-1}\left( e^{-l\sigma^2\omega^2/2} \right)\\
= g_\sigma'
$$

with \\(\sigma' = \sqrt{l}\sigma\\).

### What is the relationship between pyramid levels?<a id="sec-2-1-2" name="sec-2-1-2"></a>

The blur/width/\\(\sigma\\) associated with \\(p_l\\) is \\(\sqrt{l}\sigma\\), and that of \\(p_{l+1}\\) is \\(\sqrt{l+1}\sigma\\). The utility of this relationship
is not particularly obvious; however, it is useful in informing our approach to a pyramidal algorithm. 
Through an understanding of this relationship the Laplacian pyramid can be obtained from a Gaussian pyramid in which sequential pyramid levels
have scales/"effective" \\(\sigma\\) which differ by a factor of \\(\sqrt{2}\\).

# The Laplacian pyramid<a id="sec-3" name="sec-3"></a>

The Laplacian pyramid is a multiresolution representation derived from a Gaussian pyramid by taking the difference of sequential Gaussian pyramid levels.
The approach is seen as an improvement on the Gaussian pyramid since pyramid levels are largely decorrelated [@burt1983laplacian](https://goo.gl/U4YjLK).
The original approach, introduced in 1983 by Burt and Adelson [@burt1983laplacian](https://goo.gl/U4YjLK), was later improved upon by [@crowley2002fast](https://goo.gl/QGupMh) where a different 
"Gaussian" filter is used and only the difference of select pyramid levels is taken.

In [@crowley2002fast](https://goo.gl/QGupMh) the Gaussian pyramid is constructed using a binomial filter which approximates a Gaussian with \\(\sigma=1\\). Difference between 
pyramid levels are taken such that the ratio of scale to sample rate is constant [@crowley2002fast](https://goo.gl/QGupMh).

## A Gaussian pyramid with constant ratio of scales<a id="sec-3-1" name="sec-3-1"></a>

In the previous section, we followed the work of Burt and Adelson [@burt1983laplacian](https://goo.gl/U4YjLK), to construct a Gaussian pyramid 
with a relationship of \\(\frac{\sqrt{l+1}}{\sqrt{l}}\sigma\\) between sequential pyramid levels. The work of [@crowley2002fast](https://goo.gl/QGupMh) introduces a Gaussian 
pyramid with a fixed relationship between sequential pyramid levels by exploing the fact that the ratio of "effective" \\(\sigma\\) between 
pyramid levels \\(l\\) and \\(2l\\) is consistently \\(\sqrt{2}\sigma\\) (take note, this approach is also used in SIFT [@lowe2004distinctive](https://goo.gl/A6tVe9)).

The Gaussian pyramid of [@crowley2002fast](https://goo.gl/QGupMh) introduces stages each of which incorporates a sequence of pyramid levels (3) of the same size. 
We will write \\(p^s_l\\) to denote level \\(l\\) of stage \\(s\\). The algorithm for constructing this Gaussian pyramid is as follows:

$$
p^s_0 = 
\begin{cases}
s=0 & \text{I} * g_{\sigma}\\
s>0 & p^{s-1}_2(2x,2y)
\end{cases}\\
p^s_1 = p^s_0 * g_{\sigma}\\
p^s_2 = p^s_1 * g_{\sigma} * g_{\sigma}
$$

where \\(I\\) is the input image, and \\(g_\sigma\\) is fixed.

### A closer look<a id="sec-3-1-1" name="sec-3-1-1"></a>

To clarify why this works we examine stage 0 and stage 1, you'll have to convince yourself for other stages. For stage 0, \\(s=0\\), we have the following

<table id="tab:stage0" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 1:</span> Stage 0: levels and effective \\(\sigma\\)</caption>

<colgroup>
<col  class="right" />

<col  class="right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="right">level</th>
<th scope="col" class="right">"&sigma;"</th>
</tr>
</thead>

<tbody>
<tr>
<td class="right">0</td>
<td class="right">1</td>
</tr>


<tr>
<td class="right">1</td>
<td class="right">&radic;<span style="text-decoration:overline;">&nbsp;2&nbsp;</span></td>
</tr>


<tr>
<td class="right">2</td>
<td class="right">2</td>
</tr>
</tbody>
</table>

And for stage 1, \\(s=1\\)

<table id="tab:stage1" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 2:</span> Stage 1: levels and effective &sigma</caption>

<colgroup>
<col  class="right" />

<col  class="right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="right">level</th>
<th scope="col" class="right">"&sigma;"</th>
</tr>
</thead>

<tbody>
<tr>
<td class="right">0</td>
<td class="right">2</td>
</tr>


<tr>
<td class="right">1</td>
<td class="right">&radic;<span style="text-decoration:overline;">&nbsp;2&nbsp;</span></td>
</tr>


<tr>
<td class="right">2</td>
<td class="right">4</td>
</tr>
</tbody>
</table>

The downsampling between stages allows us to maintain a fixed "effective" \\(\sigma\\) ratio between levels while keeping the filter fixed. Without 
downsampling, the "effective" \\(\sigma\\) level relationship in stage 1 would be,

<table id="tab:stage1_nodown" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 3:</span> Stage 1: levels and effective \\(\sigma\\) without downsampling</caption>

<colgroup>
<col  class="right" />

<col  class="left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="right">level</th>
<th scope="col" class="left">"&sigma;"</th>
</tr>
</thead>

<tbody>
<tr>
<td class="right">0</td>
<td class="left">2</td>
</tr>


<tr>
<td class="right">1</td>
<td class="left">&radic;<span style="text-decoration:overline;">&nbsp;5&nbsp;</span></td>
</tr>


<tr>
<td class="right">2</td>
<td class="left">&radic;<span style="text-decoration:overline;">&nbsp;7&nbsp;</span></td>
</tr>
</tbody>
</table>

Downsampling effectively doubles the \\(\sigma\\) of our filter allowing us to maintain both a fixed "effective" \\(\sigma\\) ratio and fixed procedure
for generating Gaussian pyramid levels

![img](../figures/laplacian_gauss_pyr.png "Gaussian pyramid adapted from [@crowley2002fast](https://goo.gl/QGupMh)")

## Difference of Gaussian as an approximation for the Laplacian of Gaussian<a id="sec-3-2" name="sec-3-2"></a>

Normalizing the Laplacian with a factor of \\(\sigma^2\\) is required for true scale invariance [@lindeberg1994scale](https://goo.gl/hMH9ZE).
Maxima and minima of the Laplacian are more stable than those of the Hessian, gradient or Harris corner [@mikolajczyk2002detection](https://goo.gl/GiiV8P).

The Laplacian pyramid is constructed by taking the difference of Gaussian pyramid levels. 
The derivative of the Gaussian, \\(g_{\sigma}\\), with respect to \\(\sigma\\) can be expressed

$$
\frac{\partial}{\partial \sigma} g_{\sigma} = \frac{\partial}{\partial \sigma}\left( \frac{1}{\sigma^2} e^{-(x^2 + y^2)/2\sigma^2}\right) \\
= \left(\frac{\partial}{\partial \sigma} \frac{1}{\sigma^2}\right) e^{-(x^2 + y^2)/2\sigma^2} + \frac{1}{\sigma^2} \left(\frac{\partial}{\partial \sigma} e^{-(x^2 + y^2)/2\sigma^2}\right)\\
= - \frac{2}{\sigma^3} e^{-(x^2 + y^2)/2\sigma^2} + \frac{1}{\sigma^2} \left( \frac{x^2 + y^2}{\sigma^3} e^{-(x^2 + y^2)/2\sigma^2} \right)\\
= \left( \frac{ (x^2 + y^2) }{\sigma^5} - \frac{2}{\sigma^3} \right) e^{-(x^2 + y^2)/2\sigma^2}.
$$

Laplacian of Gaussian with respect to \\(r^2 = x^2 + y^2\\) is

$$
\frac{\partial^2}{\partial r^2} g_{\sigma} = \frac{\partial^2}{\partial r^2}\left( \frac{1}{\sigma^2} e^{-r^2/2\sigma^2}\right) \\
= -\frac{\partial}{\partial r}\left(\frac{r}{\sigma^2}\frac{1}{\sigma^2} e^{-r^2/2\sigma^2}\right) \\
= -\left(\frac{\partial}{\partial r}\frac{r}{\sigma^4}\right) e^{-r^2/2\sigma^2} + \frac{r}{\sigma^4}\left(\frac{\partial}{\partial r} e^{-r^2/2\sigma^2} \right)\\
= -\left(\frac{1}{\sigma^4}\right) e^{-r^2/2\sigma^2} + \frac{r}{\sigma^4}\left(-\frac{r}{\sigma^2} e^{-r^2/2\sigma^2} \right)\\
= -\left(\frac{1}{\sigma^4} - \frac{r^2}{\sigma^6}\right) e^{-r^2/2\sigma^2}.
$$

Relating the two we have,

$$
\frac{\partial}{\partial \sigma} g_{\sigma} = \sigma \frac{\partial^2}{\partial r^2} g_{\sigma}.
$$

Since the derivative can be approximated by a difference over the magnitude in change of the variable, we can write

$$
\frac{\partial}{\partial \sigma} g_{\sigma} \approx \frac{g_{\sigma + \Delta\sigma} - g_{\sigma}}{\Delta\sigma}\\
\Delta\sigma \left(\sigma \frac{\partial^2}{\partial r^2} g_{\sigma}\right) \approx g_{\sigma + \Delta\sigma} - g_{\sigma}.
$$

```python
    import matplotlib.pylab as plt
    import numpy as np
    
    from skimage import data
    from scipy.signal import convolve2d
    
    def disp_fmt_pyr(pyr, laplace=True):
        """
        Visualize the Laplacian pyramid
        """
        num_levels = len(pyr)
        num_stages = num_levels/2
    
        H, W = pyr[0].shape
    
        img_heights = [ H * 2.**(-i) for i in np.arange(num_stages) ]
        H = np.int(np.sum( img_heights ))
    
        out = np.zeros((H, W*2))
    
        for i in np.arange(num_stages):
            rstart = np.int(np.sum(img_heights[:i]))
            rend = np.int(rstart + img_heights[i])
    
            out[rstart:rend, :np.int(img_heights[i]*2)] = np.hstack((pyr[i*2], pyr[i*2+1]))
    
        return out
    
    
    def laplace_pyr(img, stages=3):
        """
        Compute the Laplacian pyramid
    
        Inputs:
        - img: Input image of size (N,M)
        - stages: Number of stages for the Laplacian pyramid
    
        Returns:
        A tuple of stages*2 images 
        """
    
        # approximate length 5 Gaussian filter using binomial filter
        filt = 1./16 * np.array([[1, 4, 6, 4, 1]])
        filt2 = np.pad( filt, ((0,0),(2,2)), mode='constant' )
        filt2 = convolve2d( filt2, filt, 'valid')
    
        # approximate 2D Gaussian
        # filt = convolve2d(filt, filt.T)
    
        pyr = []
    
        old_img = img
        for i in np.arange(stages):
            # zero pad the previous image for convolution
            # boarder of 2 since filter is of length 5
            p_0 = np.pad( old_img, (2,), mode='constant' )
    
            # convolve in the x and y directions to construct p_1
            p_1 = convolve2d( p_0, filt, 'valid' )
            p_1 = convolve2d( p_1, filt.T, 'valid' )
    
            # DoG approximation of LoG
            pyr.append( p_1 - p_0[2:-2,2:-2] )
    
            # convolve with scaled gaussian \sigma_2 = \sqrt(2)\sigma_1
            # this is implemented by cascaded convolution
            p_1 = np.pad( p_1, (2,), mode='constant' )
            p_2 = convolve2d( p_1, filt2, 'valid' )
            p_2 = convolve2d( p_2, filt2.T, 'valid' )
    
            # DoG approximation of LoG
            pyr.append( p_2 - p_1[2:-2,2:-2] )
    
            # subsample p_2 for next stage
            old_img = p_2[::2,::2]
    
        return pyr
    
    
    camera = data.camera()
    pyr = laplace_pyr(camera)
    img = disp_fmt_pyr(pyr)
    
    plt.imshow(img)
    plt.savefig('figures/laplace_pyr.png', bbox_inches="tight")
    
    return 'figures/laplace_pyr.png'
```	

![img](../figures/laplace_pyr.png "Laplacian pyramid")

# SIFT: Scale Invariant Feature Transform<a id="sec-4" name="sec-4"></a>

SIFT is an approach for identifying and describing image regions useful for tasks such as image recognition and retrieval.
The approach is two stage, the first stage is detection which uses ideas from automatic scale selection [@lindeberg1998edge](https://goo.gl/RRxkQf)
and Harris Corners [@harris1988combined](https://goo.gl/GSrmxK) to identify stable scale invariant features.
The second stage is description, a representation of the feature is constructed using a histogram of oriented gradients [@lowe2004distinctive](https://goo.gl/A6tVe9).
This discription method gained significant popularity after [@lowe2004distinctive](https://goo.gl/A6tVe9) through the work of 
[@dalal2005histograms](https://goo.gl/L16SYM) and [@felzenszwalb2010object](https://goo.gl/kW5HJj).

## Scale invariant detection<a id="sec-4-1" name="sec-4-1"></a>

In [@lowe2004distinctive](https://goo.gl/A6tVe9) scale invariance is achieved by identifying distinctive locations \\(p = (x, y, \sigma)\\) in the
scale-space representation of the image (i.e. the Laplacian pyramid). 
These locations are characterized as being maximal in absolute value within their local neighborhood
(i.e. eight neighbors in the same scale, and nine neighbors in both the scale above and below for a total of 26 neighbors).
The detection is scale invariant since a feature can be detected and matched across images of different scales as long as the "inherent" 
scale of the feature is represented in the scale-space representations of both images.

## Rotation invariant descriptor<a id="sec-4-2" name="sec-4-2"></a>

Although the name does not reveal the orientation invariance of the descriptor this characteristic is important for robust and repeatable
detection. The SIFT descriptor is a histogram of oriented gradients. At the detection scale the orientation and magnitude
of each point in the feature neighborhood is computed. These orientations are bined into a histogram where their contribution to 
the bin count is weighted by the magnitude of the gradient and the distance of the neighbor from the detected feature.
The dominant orientation (the bin with the highest count) is refered to as the keypoint orientation. The coordinates 
of descriptor (histogram) are shfited relative to this orientation making the descriptor rotation invariant.

## A closer look<a id="sec-4-3" name="sec-4-3"></a>

### Pyramid construction<a id="sec-4-3-1" name="sec-4-3-1"></a>

1.  Frequency of sampling in scale

2.  Frequency of spatial sampling

### Accurate keypoint localization and thresholding<a id="sec-4-3-2" name="sec-4-3-2"></a>

A candidate feature is found by nonmaximum suppression. A 3D quadratic function is fit to the candidate feature and its neighbors
where the candidate feature is assumed to be the centroid. The location of the extremum is determined by taking the derivative of 
the quadratic and setting it to zero. If the solution computed here is more than \\(0.5\\) from the candidate feature the extremum is 
closer to one of the other sample points and the interpolation is computed about that point instead. If the function value at the 
extrema is small (less than 0.03 in the paper) the point is discarded as unstable due to low contrast.

A candidate feature will also result on edges where localization of the extrema is poorly determined. These cases are eliminated by 
the metric introduced in [@harris1988combined](https://goo.gl/GSrmxK). The autocorrelation matrix is formed the eigenvalues of which illucidate the 
cornerness of the region.

### Other invariances<a id="sec-4-3-3" name="sec-4-3-3"></a>

linear and non-linear illumination

![img](../figures/sift_increasing_robustness.png "Adapted from  [@lowe2004distinctive](https://goo.gl/A6tVe9)")

