# Diffusion models and single-image 3D generation: do they keep translucency?

A learning repository with one experiment. The first half implements diffusion models from scratch
(DDPM, classifier-free guidance, DDIM). The second half takes two pretrained single-image-to-3D models and asks
a question from material perception: **when the input image shows a translucent object, what do these models
put on the side they cannot see?** The answer is checked against physically based renders of the true object
from the same viewpoints, and against an opaque control with the same shape and lighting.

Everything was run on a Quadro M4000 (8 GB, Maxwell generation).

## At a glance

- **Diffusion from scratch.** DDPM, class conditioning with classifier-free guidance, and a DDIM sampler, in
  about 300 lines of PyTorch. Trained on a 2D toy distribution and on MNIST.
- **Zero-1-to-3 sampling written by hand.** The conditioning (CLIP embedding + relative camera pose, input
  latent concatenated to the noisy latent) and the guidance loop are written directly on the pretrained
  weights, using the same pieces as the MNIST code.
- **An experiment with ground truth.** 6 shape-and-lighting conditions x 3 scattering densities. For each, the
  true novel views and an opaque control are path traced; Zero-1-to-3 (3 seeds) and TripoSR are run on the same
  input image.

| Result (unseen views: 90, 180, 270 degrees; 18 cells) | Zero-1-to-3 | TripoSR |
|---|---|---|
| Higher contrast than the true translucent object | 18 / 18 | 11 / 18 |
| Darker than the true translucent object | 16 / 18 | 15 / 18 |
| Front lighting: luminance / contrast distance to truth | 0.116 / 0.278 | 0.210 / 0.184 |
| Front lighting: luminance / contrast distance to opaque control | 0.116 / 0.275 | 0.062 / 0.445 |

**What holds:** on the unseen side, Zero-1-to-3 always produces stronger shading contrast than the real
translucent object, and usually a darker one. **What does not hold as stated:** the outputs do not simply match
an opaque object. That description fits the front-lit studio condition well, but over all conditions the
outputs sit between the two references. Given matched translucent and opaque inputs, the far side
changes a little, and the change follows the front view of the input rather than the true far side. Yet the
models' image encoders predict human translucency ratings of real photographs well, including the difference
between two photos of the same soap (Part 3). Used as a translucency measure, that readout finds that
Zero-1-to-3 gives the far side the same value as the input view, while the real far side scores as more
translucent than its front. Observers' fixations go to the bright parts of the object, not to where the
readout finds translucency. Details and limits are below.

![Front-lit studio, low density](image_to_3d/figures/views_stim01_studio_rot000_coef016.png)

*Front-lit studio, lowest scattering density. Columns are viewpoints; the leftmost is the input view. Row 1 is
the path-traced truth, row 2 the opaque control, rows 3 and 4 the models. The true object stays bright from
behind because light passes through it. Zero-1-to-3 turns it into a dark, metal-like object.*

## Part 1: diffusion from scratch

| File | Contents |
|---|---|
| `diffusion/ddpm.py` | Noise schedule, forward process, epsilon-prediction loss, ancestral sampler, DDIM sampler, classifier-free guidance |
| `diffusion/unet.py` | Small U-Net (0.94 M parameters) with time embedding and optional class embedding |
| `diffusion/train_2d.py` | DDPM on a 2D spiral, with figures of the forward and reverse processes |
| `diffusion/train_mnist.py` | Unconditional DDPM on MNIST |
| `diffusion/train_mnist_cfg.py` | Class-conditional MNIST with label dropout; compares guidance scales and step counts |

| ![guidance](diffusion/figures/mnist_cfg_scales.png) | ![steps](diffusion/figures/mnist_ddim_steps.png) |
|---|---|
| Rows: guidance scale 0, 1, 3, 7. Columns: requested digit 0-9. Same initial noise in every row. | Rows: DDIM with 5, 10, 20, 50 steps, then DDPM with 1000 steps. Guidance scale 3. |

- With guidance scale 1 several digits are malformed; with 3 and 7 all ten are readable.
- DDIM is close to converged at 10 steps (0.1 s for 10 images, against 9.0 s for 1000 DDPM steps).
- Training: 15 epochs, about 16 minutes each for the unconditional and the conditional model.

## Part 2: translucency in single-image 3D generation

### Question

A translucent object lets light through, so its far side can be bright, and its thin parts glow when lit from
behind. A model that sees one image has to invent the far side. Does it invent the far side of a translucent
object, or of an opaque one?

### Setup

- **Inputs.** Existing path-traced stimuli from a perception experiment: two blob shapes with a glass surface
  and a homogeneous scattering volume. Scattering density takes three levels (16, 128, 512; higher is more
  turbid). Lighting is a studio environment map (two rotations) or a sunlit outdoor map (front, side and back
  lighting). 6 shape-and-lighting conditions x 3 densities = 18 cells.
- **Models.** Zero-1-to-3 XL (a view-conditioned diffusion model; 3 seeds, 50 DDIM steps, guidance 3) and
  TripoSR (a feed-forward reconstruction model, rendered from its own radiance field).
- **Truth.** The camera in the original Blender scene is orbited around the object by 90, 180 and 270 degrees.
  Object and lighting stay fixed. Same Cycles settings as the stimuli (256 bounces, no denoising), at 512 x 512
  and 512 samples. The 0-degree render reproduces the original stimulus (mean luminance within 0.03 in all 18
  cells).
- **Opaque control.** Same shape, lighting, surface roughness and index of refraction, but no light enters
  the object. Its reflectance is chosen per cell so that the front view has the same mean luminance as the
  translucent stimulus: "the opaque object that explains the front view".
- **Measure.** Mean luminance and contrast (standard deviation / mean of linear luminance) inside the object,
  averaged over the three unseen views. Pixelwise comparison is not meaningful because the models guess the
  far-side shape wrongly. Averaging over 90, 180 and 270 degrees also makes the result independent of which
  rotation direction a model calls positive.

### Results

Distance to each reference (mean absolute difference over cells, luminance / contrast):

| Lighting | Cells | Zero-1-to-3 to truth | Zero-1-to-3 to opaque | TripoSR to truth | TripoSR to opaque |
|---|---|---|---|---|---|
| Front | 9 | 0.116 / 0.278 | 0.116 / 0.275 | 0.210 / 0.184 | 0.062 / 0.445 |
| Side | 3 | 0.228 / 0.405 | 0.252 / 0.282 | 0.186 / 0.188 | 0.210 / 0.242 |
| Back | 6 | 0.132 / 0.191 | 0.267 / 0.134 | 0.120 / 0.162 | 0.196 / 0.165 |
| All | 18 | 0.140 / 0.270 | 0.189 / 0.229 | 0.176 / 0.177 | 0.131 / 0.318 |

For scale, the two references differ from each other by 0.174 / 0.394 on average.

![summary](image_to_3d/figures/summary_scatter.png)

*Each point is one cell. Points on the diagonal agree with the reference on the horizontal axis. Error bars
on Zero-1-to-3 are the standard deviation over 3 seeds.*

1. **Zero-1-to-3 over-shades the unseen side.** Its contrast is above the truth in all 18 cells and its mean
   luminance is below the truth in 16. The variation across seeds is small (0.022 / 0.046), so this is not
   sampling noise.
2. **In the front-lit studio condition the output matches the opaque control.** For the first shape the unseen
   views give 0.32 / 0.74, 0.33 / 0.63, 0.35 / 0.69 for the three densities, against 0.25 / 0.79, 0.29 / 0.72,
   0.35 / 0.65 for the opaque control and 0.56 / 0.38, 0.52 / 0.28, 0.45 / 0.42 for the truth.
3. **This does not generalise as "the model assumes an opaque object".** Over all cells Zero-1-to-3 is closer
   to the truth in luminance (0.140 against 0.189) and only slightly closer to the opaque control in contrast
   (0.229 against 0.270). Under sunlight the opaque control has hard cast shadows (contrast above 1.2) that
   the model does not produce.
4. **Scattering density barely changes what the model puts on the far side** in the front-lit studio condition
   (luminance 0.32 to 0.35 across densities, where the truth goes from 0.56 to 0.45).
5. **TripoSR is darker than the truth in 15 of 18 cells** but its contrast is mixed (higher in 11). Its renders
   are blurry, which lowers contrast regardless of material.

### Back lighting

With the light behind the object, the front of the translucent object is bright because light comes through
it. No opaque reflectance reproduces that: in 8 of the 18 cells (all back-lit cells and two side-lit ones) even
a reflectance of 0.9 leaves the opaque front view darker than the stimulus. In those cells the control uses
0.9 and its front view does not match. So the back-lit input images contain direct evidence that the object
transmits light. Zero-1-to-3 is still darker than the truth on the far side in 4 of the 6 back-lit cells and
higher in contrast in all 6.

![Back-lit sunlight, low density](image_to_3d/figures/views_stim01_rustig_koppie_rot135_coef016.png)

### Does the model tell the two materials apart?

The comparison above shows that the output is wrong, not whether the model responds to the material at all. To
test that, Zero-1-to-3 is given pairs of inputs that differ only in material: the translucent stimulus, and the
opaque control with the same shape, lighting and front-view mean luminance. The measure was fixed before
running: regress, across cells, the difference between the two model outputs on the difference between the two
true far sides. A slope of 0 means the model ignores the material; 1 means it tracks it. The prediction was a
slope near 0. Only the 10 cells where the front views could be matched are a fair pair.

| Unseen views, 10 matched cells | Luminance | Contrast |
|---|---|---|
| Slope (model difference on true difference) | +0.21 | +0.35 |
| Mean absolute true difference | 0.215 | 0.541 |
| Mean absolute model difference | 0.057 | 0.199 |
| Difference expected from seeds alone | 0.026 | 0.060 |
| Cells where the model difference has the right sign | 7 / 10 | 9 / 10 |

![pairs](image_to_3d/figures/pairs_scatter.png)

**The prediction was wrong: the two inputs do not give the same far side.** The difference is above seed noise
and mostly has the sign of the true difference, but it is a fifth to a third of its size.

![pairs example](image_to_3d/figures/pairs_example.png)

**The difference follows the front view of the input, not the true far side.** Across the 10 cells, the model's
far-side contrast difference correlates 0.79 with the contrast difference between the two input images, and
0.48 with the true far-side difference; the input difference and the true far-side difference are themselves
unrelated (0.05). So the model carries the shading of the visible side over to the hidden side, and the
agreement in sign with the truth is largely a by-product. Mean luminance is not explained this way: the two
inputs were matched in front luminance (largest difference 0.019), yet the far side is 0.057 brighter for the
translucent input. Ten cells, one of them with a much larger contrast difference than the rest.

### Is the far side drawn as a glossy object?

In the example above the translucent input becomes a metal-like object. Highlight cues from a separate project
on gloss perception (highlight region = where a multi-scale centre-surround response exceeds a threshold fixed
on a rated gloss dataset) were applied to the same images.

| Unseen views, 10 matched cells | Highlight coverage | Highlight contrast | Highlight edge sharpness |
|---|---|---|---|
| Truth, translucent | 0.084 | 0.40 | 0.135 |
| Truth, opaque | 0.099 | 0.68 | 0.076 |
| Model, translucent input | 0.183 | 0.54 | 0.076 |
| Model, opaque input | 0.194 | 0.60 | 0.058 |

- The model's far side has about twice the highlight coverage of the true translucent object (larger in 9 of
  10 cells) and higher highlight contrast (10 of 10).
- This is not specific to translucent inputs: the opaque input gives the same coverage and contrast. The far
  side is drawn highlight-rich whatever the material.
- The input changes only the edge sharpness (sharper for the translucent input in 10 of 10), which again
  matches the front views (0.134 against 0.086).
- These cues also respond to shading edges and fine relief, and the generated shapes are more convoluted than
  the true ones. Perceived gloss was not measured.

### Limits

- Two shapes, two environment maps, one surface finish. Both shapes are irregular blobs.
- The stimuli were rendered with a 5-degree field of view. Zero-1-to-3 was trained at about 49 degrees, so
  part of the error may come from this mismatch rather than from the material.
- The models get the far-side shape wrong, and shape changes shading. The opaque control shows what an opaque
  object of the true shape looks like; it does not remove the shape difference in the model output.
- The object region of generated images is found by thresholding the white background, so highlights that
  touch the outline can be cut. Blender images use the alpha channel.
- Luminance statistics are a proxy. Whether observers see the generated views as translucent has not been
  measured.
- TripoSR assumes the input is at zero elevation; the stimuli are at 7.2 degrees.

### Files

| File | Contents |
|---|---|
| `image_to_3d/common.py` | Conditions, input preparation, luminance statistics |
| `image_to_3d/render_gt_views.py` | Blender script: orbit the camera, render the translucent truth or the opaque control |
| `image_to_3d/render_all_gt.py` | Runs the Blender script for every condition and picks the control reflectance |
| `image_to_3d/run_zero123.py` | Zero-1-to-3 sampling written on the pretrained weights |
| `image_to_3d/run_triposr.py` | TripoSR inference and novel-view rendering |
| `image_to_3d/analyze_all.py` | Tables, `figures/summary.json`, and the comparison figures |
| `image_to_3d/analyze_pairs.py` | The matched-pair test: translucent input against opaque input |
| `image_to_3d/analyze_gloss.py` | Highlight cues of true and generated far sides (uses a separate gloss project) |
| `soap/segment.py`, `soap/finalize.py` | Object regions for the soap photographs |
| `soap/features.py`, `soap/predict.py` | Model features and image statistics; prediction of the ratings |
| `soap/ratings.py`, `soap/within_pair.py` | Rating loader and cross-validation; the test on two photos of the same soap |
| `soap/patch_features.py`, `soap/gaze_model.py` | Patch features of the photos; gaze against the model features |
| `image_to_3d/translucency_readout.py` | The soap-trained readout applied to the Part 2 images |

Peak GPU memory: 6.5 GB for Zero-1-to-3 (4 views with guidance), 2.2 GB for TripoSR.

## Part 3: do the models' features carry perceived translucency?

Part 2 looks at outputs. This part asks whether the information is present inside the models at all.

- **Data.** 122 photographs of 61 soaps (two photos each) from the Translucent Image Dataset (Liao, Sawayama &
  Xiao, 2023, CC BY 4.0), with translucency ratings on a 7-point scale from 6 observers collected in our lab.
  Photographs and ratings are not in this repository.
- **Target.** Ratings standardised per observer and averaged per photo. Split-half correlation between
  observers is 0.95, so a perfect predictor would reach about r = 0.99.
- **Object regions.** A background-removal network, checked against 61 hand-drawn contours (mean IoU 0.936);
  four photos where it included the cast shadow were corrected by hand or with a second network, one is left.
- **Features.** Zero-1-to-3: its CLIP image embedding, and U-Net activations (mid block and two up blocks,
  averaged inside the object) when the photo is the conditioning image and its own noised latent is the input,
  at three noise levels. TripoSR: its DINO image tokens. Baselines: 14 image statistics used in the lab
  (luminance, contrast, highlight and spectral measures) and mean colour.
- **Prediction.** Ridge regression, leaving one soap out at a time so that both photos of a soap are always
  held out together. Confidence intervals resample soaps; chance is from 200 permutations of ratings over soaps.

| Features | Dimensions | r, held-out soaps | 95% CI |
|---|---|---|---|
| Mean colour | 6 | 0.57 | 0.42 to 0.69 |
| 14 image statistics | 14 | 0.69 | 0.61 to 0.75 |
| Image statistics + colour | 20 | 0.66 | 0.57 to 0.75 |
| Zero-1-to-3 CLIP embedding (photo / object on white) | 768 | 0.92 / 0.93 | 0.90 to 0.95 |
| Zero-1-to-3 U-Net, low / medium / high noise (object on white) | 3200 | 0.87 / 0.88 / 0.89 | 0.83 to 0.92 |
| Zero-1-to-3 U-Net, low / medium / high noise (photo) | 3200 | 0.84 / 0.81 / 0.85 | 0.74 to 0.89 |
| TripoSR DINO tokens (photo / object on white) | 768 | 0.86 / 0.82 | 0.74 to 0.90 |

Chance level (95th percentile of permutations) is 0.14 to 0.23 for every row.

![prediction](soap/figures/predict_r.png)

- **Perceived translucency can be read out linearly from both models' features**, far better than from the
  image statistics (+0.26 for the best model features over statistics + colour, 95% CI +0.18 to +0.35).
- Removing the background changes little, so the prediction does not rest on shadows or light cast on the table.
- Together with Part 2: the information that separates translucent from opaque is available to the model, but
  the generated far side does not use it to change how light behaves. The failure is in what is rendered, not
  in what is encoded. The two sections below check the first half of this (is the readout more than soap-type
  recognition?) and the second half (what does the readout say about the generated far side?).

### Same soap, two photos

Only 5% of the variance of the ratings lies within soaps, so r = 0.93 mostly measures how well the features
separate soaps. Recognising the kind of soap (clear glycerine bar, milky bar) would go a long way. The two
photos of a soap show the same object under different light: one diffuse, the other a strong side or back
light that also throws coloured, transmitted light onto the table. A readout that also follows the rating
difference between the two photos responds to how the object looks, not only to which kind of soap it is.

- **Test.** The readouts above (whole soaps held out) are applied to both photos of the held-out soap. The
  difference between the two predictions is compared with the difference between the two ratings, using a
  correlation about zero so that the order of the photos does not matter. No readout was trained on
  differences within a soap. Chance comes from flipping the signs of the rating differences.
- **Ceiling.** The differences are small (mean 0.31, against a standard deviation of 0.94 over photos) but
  observers agree on them: split-half r = 0.70, so a perfect predictor would reach about r = 0.91.

| Features | r across soaps (soap means) | r within soaps | 95% CI | p | Right direction, 29 large differences |
|---|---|---|---|---|---|
| Image statistics + colour | 0.72 | 0.01 | -0.23 to 0.26 | 0.47 | 14 |
| 14 image statistics | 0.78 | -0.04 | -0.25 to 0.17 | 0.65 | 13 |
| Zero-1-to-3 CLIP, object on white | 0.95 | 0.60 | 0.36 to 0.76 | 0.001 | 24 |
| Zero-1-to-3 CLIP, photo | 0.95 | 0.47 | 0.17 to 0.67 | 0.005 | 19 |
| Zero-1-to-3 U-Net, high noise, object on white | 0.92 | 0.55 | 0.31 to 0.73 | 0.001 | 24 |
| TripoSR DINO, photo | 0.90 | 0.48 | 0.22 to 0.68 | 0.001 | 19 |
| TripoSR DINO, object on white | 0.88 | 0.23 | 0.00 to 0.44 | 0.04 | 19 |

Chance (95th percentile of sign flips) is 0.17 to 0.39. It is high for the model features because a few pairs
with large predicted differences dominate the correlation.

![within soap](soap/figures/within_pair.png)

- **The model readouts follow differences within a soap; the image statistics do not.** The CLIP readout reaches
  r = 0.60, two thirds of the ceiling, although it was trained only across soaps. The image-statistics readout
  is at zero (difference +0.59, 95% CI +0.23 to +0.91). So the model features carry more than the kind of soap.
- **This rests on the pairs where observers clearly see a difference** (split at the median rating difference,
  0.21; this split was added after seeing the correlation). Among the 29 larger differences the CLIP readout
  gives the right direction in 24; two halves of the observer panel agree on the direction in 90% of them.
  Among the 32 smaller ones the two halves agree in only 41%, so there is no reliable direction to predict,
  and the readout is at chance there (9 of 32).
- Removing the background does not hurt the Zero-1-to-3 features (0.60 without, 0.47 with; the intervals
  overlap widely), so the light cast on the table is not what they rely on. For DINO it goes the other way
  (0.23 without, 0.48 with).
- Fitting a readout directly to the 61 differences is weaker (CLIP 0.46, chance 0.22): too few examples.

### The readout as a translucency measure for the Part 2 images

The CLIP readout (object on white) is fitted to all 122 soap photos and applied to the Part 2 images, placed
the same way (the object fills 85% of a white square). Its value is in units of the soap ratings. This puts
the question of Part 2 in the model's own feature space: by the measure that predicts human ratings, does the
far side that Zero-1-to-3 generates look translucent? A readout from TripoSR's DINO tokens gives a second
measure. The check and the questions were written into the script before it was run; the comparison with the
input view was added after seeing the first figure.

- **Check.** On the far side the true translucent object scores above its opaque control in 18 of 18 cells
  with both measures. In the front view, where the two were matched in mean luminance, it scores higher in 13
  (CLIP) and 16 (DINO) of 18. So the measure carries over from photographs of soap to the rendered blobs, at
  least where the true difference is large.

| Mean over 18 cells | CLIP readout | DINO readout |
|---|---|---|
| Input (true front view), translucent | -0.06 | -0.67 |
| True far side, translucent | +0.11 | -0.47 |
| True far side, opaque control | -0.31 | -1.12 |
| Zero-1-to-3, same view as the input | -0.03 | -0.95 |
| Zero-1-to-3, far side | -0.02 | -1.22 |
| TripoSR, same view as the input | -0.58 | -0.63 |
| TripoSR, far side | -0.38 | -0.90 |

![readout](image_to_3d/figures/readout_scatter.png)

*Each point is one cell. Horizontal: readout of the input image. Vertical: readout of the far side (mean of
90, 180 and 270 degrees). On the diagonal the far side scores the same as the front.*

1. **The real translucent object scores as more translucent from the far side than from the front**: +0.17
   with CLIP (higher in 15 of 18 cells), +0.20 with DINO.
2. **Zero-1-to-3's far side keeps the value of the input view.** With CLIP its far side differs from its own
   0-degree output by +0.00 and is below the true far side in 13 of 18 cells (seed SD 0.08). Across cells it
   follows the input (partial r = 0.66 with the truth held fixed) more than the true far side (0.24 with the
   input held fixed). It is not pulled down to the opaque control (-0.31); as in Part 2 it lies between the
   two references.
3. The DINO readout agrees for Zero-1-to-3: the far side is 0.26 below the model's own 0-degree output and below
   the truth in 18 of 18 cells. But DINO also scores the 0-degree output 0.28 below the input, so it rates
   generated images as less translucent overall.
4. **Matched pairs** (10 cells). The far-side difference between translucent and opaque input is +0.20 (CLIP),
   the same as the difference between the two inputs (+0.21) and half the true far-side difference (+0.39);
   the sign agrees with the truth in 6 of 10. DINO gives +0.38, +0.34 and +0.67, and there the model's
   difference correlates more with the input difference (0.86) than with the true one (0.50). In CLIP the
   input and true differences are themselves correlated (0.93), so CLIP cannot separate them.
5. **TripoSR: the two measures disagree.** Its blurry renders score low on CLIP already at 0 degrees and rise
   towards the far side. On DINO, TripoSR's own encoder, the 0-degree render keeps the input value and the far
   side falls by 0.28. No conclusion is drawn for TripoSR.

So Zero-1-to-3 keeps the translucency that its conditioning embedding encodes for the visible side and copies
it, unchanged, to the hidden side. It misses the increase that the real object shows away from the input view.
Because the output is conditioned on this embedding, staying close to the input in CLIP space is expected for
any property, not only translucency; the DINO measure, which the model never sees, shows the same pattern.

### Where observers look

The ratings were collected with eye tracking. 97% of fixations fall on the object, which is why earlier summaries
of how widely people look (entropy, dispersion, share inside the outline) showed no relation to the ratings. Here
the question is where on the object people look, compared with where the model features carry the information.

- **Observers.** One observer (P003) fixated the centre of the object far more than the others: 81% of fixations
  within a quarter of the object size from its centroid, against 30 to 58%. P003 is left out, leaving 5
  observers and a median of 36 fixations per photo. The rating target is unchanged (all 6 observers).
- **Features.** Patch tokens of the photos as shown: TripoSR's DINO (32 x 32) and the last layer of Zero-1-to-3's
  CLIP (16 x 16).
- **Controls.** A "typical gaze layout" maps other soaps' fixations onto this object's bounding box, with the
  same number of fixations as the photo has. It keeps what all photos share (looking near the centre, for
  example) and drops what is specific to the photo. The questions and decisions were written into the script
  before it was run.

**(a) Does pooling by gaze predict the ratings better?** The patches inside the object are averaged with
different weights, and ratings are predicted as in the table above.

| Weights | DINO patches | CLIP patches |
|---|---|---|
| Uniform over the object | 0.90 | 0.92 |
| Gaze density of that photo | 0.87 | 0.91 |
| Typical gaze layout, 20 draws (median, range) | 0.85 (0.83 to 0.88) | 0.91 (0.90 to 0.92) |
| Where observers did not look | 0.90 | 0.92 |

- Gaze adds nothing beyond the typical layout: 95% CI of the difference -0.01 to +0.04 (DINO) and -0.01 to
  +0.01 (CLIP). Gaze is below uniform pooling for DINO (-0.03, CI -0.05 to -0.01): concentrating the weights
  only discards patches. The parts nobody looked at carry as much as the whole object.
- Averaging DINO patches inside the object gives 0.90, above the 0.86 of the first table, which averaged all
  tokens including the background.

**(b) Do observers look where the readout is high?** The uniform readout (soap held out) is applied to each patch,
which splits the photo's prediction into a map. The measure is the map value at fixations, standardised within
the object, minus the value the typical layout would give.

| Map | DINO | CLIP | Photos above the typical layout (DINO / CLIP) |
|---|---|---|---|
| Luminance | +0.26 (0.21 to 0.31) | +0.25 (0.20 to 0.30) | 101 / 96 of 122 |
| Readout | +0.04 (-0.01 to +0.09) | +0.05 (+0.01 to +0.09) | 71 / 73 |
| Readout with luminance regressed out | +0.01 (-0.04 to +0.06) | +0.04 (+0.01 to +0.08) | 63 / 67 |

![gaze examples](soap/figures/gaze_examples.png)

*Top: the two highest- and two lowest-rated photos with the fixations of 5 observers. Bottom: the DINO readout
per patch on one colour scale. The translucency signal covers the whole object; fixations do not pick out
particular patches.*

- **Observers look at the bright parts of the object**, a quarter of a standard deviation above the typical
  layout, in about 4 of 5 photos. How strongly they do so is unrelated to the rating (r = -0.01).
- **They hardly look where the readout is high.** The effect is a fifth of the luminance effect, absent for DINO,
  and only just above zero for CLIP among 6 comparisons. The readout maps are not luminance maps (correlation
  within the object 0.09 for DINO, 0.03 for CLIP).
- So, in this task, gaze follows brightness. The translucency information that the models read out is spread
  over the object, not concentrated where people look. Where people look does not change with translucency,
  which fits the earlier null results.

### Limits of Part 3

- 61 objects and 6 observers. The embedding is a general-purpose image encoder used as is, so this says nothing
  about what training for 3D added. The U-Net receives the image itself, so high decodability there is
  expected.
- The within-soap test shows that the readout is not only soap-type recognition, but within-soap differences
  are mostly a change of lighting, and the agreement comes from about 30 pairs.
- The readout is trained on photographs of soap and applied to renders of blobs with a glass surface. The
  front-view check passes in only 13 of 18 cells with CLIP. Readout values are not ratings of these images:
  whether observers see the generated far side as less translucent than the true one is still unmeasured.
- Gaze: 5 observers and about 36 fixations per photo, recorded during a rating task. The order and timing of
  fixations are not used, and the 30-pixel smoothing is close to one DINO patch.
- Decodable does not mean used.

## Related work and what is still open

From a short literature search (summaries of the papers, except TRELLIS.2, which was read in full):

- Recent image-to-3D models output albedo, roughness and metallic maps (for example Hunyuan3D 2.1). There is
  no channel for light that passes through the object, so translucency has nowhere to go even if it is detected.
- TRELLIS.2 (Xiang et al., 2025) adds a per-voxel opacity and states that it can handle translucent surfaces.
  The paper defines opacity only as a value in [0, 1], models no scattering, transmission or refraction, gives
  no evaluation specific to translucent objects, and lists no limitation about materials. It needs far more
  GPU memory than was available here, so it was not run.
- Subsurface scattering is handled on the reconstruction side, from many views of one object (SSS-3DGS, GTSR).
  DIAMOND-SSS fine-tunes view-synthesis and relighting diffusion models on translucent captures, which suggests
  that the missing ingredient is training data rather than architecture.
- Marlow, Kim & Anderson (2017) argue that perceived opacity depends on how intensity covaries with surface
  orientation, computed where 3D shape is explicit. A model that maps image to image without an explicit shape
  or light has no place to do this for the unseen side.

No study was found that measures what a single-image 3D model puts on the far side of a translucent object
against a physically based reference, or that relates it to human judgements. The search was not exhaustive.

## What did not work

A multi-view set of a wax statue and an opaque copy (rendered earlier with another renderer) looked ideal: same
shape and light, true novel views already available. Zero-1-to-3 did not produce coherent side views of the
tall, thin statue (duplicated or fragmented figures at 90 and 270 degrees, with either material), and the
higher-resolution wax images were not denoised. The scripts are kept (`image_to_3d/statue.py`); no result
from this set is reported.

## Setup

```
python -m venv .venv
.venv\Scripts\pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple
.venv\Scripts\pip install -r requirements.txt
```

Part 2 also needs the TripoSR code in `data/third_party/TripoSR`, the model weights from Hugging Face
(`stabilityai/TripoSR`, `bennyguo/zero123-xl-diffusers`), Blender 4.5, and the stimulus scenes, which are not
part of this repository.
