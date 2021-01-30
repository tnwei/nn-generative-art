# nn-generative-art

Generative art with neural networks in Pytorch. Read this [Towards Data Science article](https://towardsdatascience.com/making-deep-neural-networks-paint-to-understand-how-they-work-4be0901582ee) and [this post on otoro.net](https://blog.otoro.net/2015/06/19/neural-network-generative-art/), mesmerized by the wild splotches of colour, and decided to dive into this further. WIP.

Replicating work in the blogpost produced this: ![grid of generated abstract images using neural networks](output/grid.png)

Expanded upon it further to generate smoothly varying frames over time, by sampling trajectories in a latent space. Added image upscaling to get this to run on my Thinkpad X220. Example in `cli.py`, libraries used in `environment.yml`.

![Animated latent walk](output/animated-latent-walk.gif)

``` bash
(nn-gen-art) tnwei@rama:~/projects/nn-generative-art$ python cli.py -h
usage: cli.py [-h] [--XDIM XDIM] [--YDIM YDIM] [--scale SCALE] [--noframelimit]

optional arguments:
  -h, --help      show this help message and exit
  --XDIM XDIM     Dimension of X, defaults to 160
  --YDIM YDIM     Dimension of Y, defaults to 90
  --scale SCALE   Scale factor to resize the image, defaults to 4
  --noframelimit  Lifts cap on max frames per second, not recommended
```


