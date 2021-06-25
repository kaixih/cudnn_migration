# cuDNN Migration Samples 

The guide is for users of cuDNN v7 legacy APIs to migrate to cuDNN v8 frontend
APIs, where one can benefit from faster computational kernels, lower CPU
overhead, more flexible controls, etc. More in [here](https://github.com/NVIDIA/cudnn-frontend).

Differing from the offical samples, this repo contains self-contained code
samples to demonstrate the one-to-one mapping of v7 legacy APIs and its
corresponding v8 frontend API. To make the code easy to read, I restrain the use
of function nesting, template, branches, command arguments, etc.

# Usage
In general, every `xxx_v7_xxx.cpp` corresponds to a `xxx_v8_xxx.cpp` and the
`Makefile` shows how to compile them. When running the executable, we can
specify which algorithm (v7) or which engine (v8) for the convolution. They are
supposed to generate same results when the inputs are equal. Note, the "same"
outputs don't mean bitwise same but the difference between two numbers is under
a small acceptable tolerance. For example,

- Compile the code with v8 frontend APIs and use the 0th engine.
```bash
$ make cudnn_v8_conv2d_fwd_float.out
$ ./cudnn_v8_conv2d_fwd_float.out 0
```

- Compile the code with v7 legacy APIs and use the 0th algorithm.

```bash
$ make cudnn_v7_conv2d_fwd_float.out
$ ./cudnn_v7_conv2d_fwd_float.out 0
```
