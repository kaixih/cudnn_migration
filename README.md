# cuDNN Migration Samples 

The guide is for users of cuDNN v7 legacy APIs to migrate to cuDNN v8 frontend
APIs, where one can benefit from faster computational kernels, lower CPU
overhead, more flexible controls, etc. More in [here](https://github.com/NVIDIA/cudnn-frontend).

Differing from the offical samples, this repo contains self-contained code
samples to demonstrate the one-to-one mapping of v7 legacy APIs and its
corresponding v8 frontend API. To make the code easy to read, I restrain the use
of function nesting, template, branches, command arguments, etc.
