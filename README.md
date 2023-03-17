# stochastic-interpolants


<img src="interp_images.png"  width="675" height="300">

<img src="http://malbergo.me/docs/papers/ode_v_sde.png"  width="675" height="200">

This repo provides a very simple implementation of the stochastic interpolant method of [Building Normalizing Flows with Stochastic Interpolants](https://arxiv.org/abs/2209.15571) and [Stochastic Interpolants: A Unifying Framework for Flows and Diffusions](https://arxiv.org/abs/2303.08797).

The intent of this repo is to provide the reader with an interactive tool to understand the mechanisms of the framework, as well as to reproduce any simple figures in [2].

A demonstration script of defining an interpolant $x_t = I_t(x_0, x_1) + \gamma(t) z$ and learning the associated velocity fields $v_t(x)$ and $s_t(x)$ (for the score function) is given in the notebooks folder in 'checker.ipynb'. 


ODE and SDE intregrators, as well as the class for the interpolant, are provided in `interflow/stochastic_interpolant.py`









### If you use this code for some purpose, please cite:

```
@inproceedings{
albergo2023building,
title={Building Normalizing Flows with Stochastic Interpolants},
author={Michael Samuel Albergo and Eric Vanden-Eijnden},
url={https://arxiv.org/abs/2209.15571},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
}
```

```
@misc{albergo2023stochastic,
  doi = {10.48550/ARXIV.2303.08797},
  url = {https://arxiv.org/abs/2303.08797},
  author = {Albergo, Michael S. and Boffi, Nicholas M. and Vanden-Eijnden, Eric},
  title = {Stochastic Interpolants: A Unifying Framework for Flows and Diffusions},
  publisher = {arXiv},
  year = {2023},
}

```

#### Please also consider citing these two related works by Liu et al 2022 and Lipman et al 2022:

```
@inproceedings{
liu2022,
title={Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow},
author={Xingchao Liu and Chengyue Gong and Qiang Liu},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=XVjTT1nw5z}
}
```

```
@inproceedings{
lipman2022,
title={Flow Matching for Generative Modeling},
author={Yaron Lipman and Ricky T. Q. Chen and Heli Ben-Hamu and Maximilian Nickel and Matthew Le},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=PqvMRDCJT9t}
}
```

