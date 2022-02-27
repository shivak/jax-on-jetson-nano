# JAX wheels for NVIDIA Jetson Nano / JetPack 4.6

If you're running <a href="https://developer.nvidia.com/embedded/jetpack#collapseJetsonNano">Jetpack 4.6</a> on your Jetson Nano, you can install JAX 0.2.4 with the following commands:

```
python3 -m pip install --upgrade scipy numpy
python3 -m pip install https://github.com/shivak/jax-on-jetson-nano/releases/download/jax-v0.2.4/jaxlib-0.1.57+cuda102-py3-none-any.whl
python3 -m pip install https://github.com/shivak/jax-on-jetson-nano/releases/download/jax-v0.2.4/jax-0.2.4-py3-none-any.whl
```

**Background**. The Jetson Nano is a useful little testbed for TinyML. But it's stuck
on CUDA 10.2 and Python 3.6. Wheels for <a href="https://qengineering.eu/install-pytorch-on-jetson-nano.html">recent PyTorch</a> are available.
However, JAX doesn't have old aarch64 + CUDA 10.2 builds.

# How to build them 
If you're building on the Nano itself, you'll want to reduce wear on the poor little SD card: mount a USB drive to `/d`, and 
point Bazel's cache towards it: `ln -s ~/.cache/bazel /d/bazel-cache`. Pull a version of JAX that's compatible with Python 3.6, along with a contemporaneous version of Bazel.
```
git clone -b jax-v0.2.4 https://github.com/google/jax.git /d/jax
curl https://releases.bazel.build/3.4.0/release/bazel-3.4.0-linux-arm64 -o /d/bazel
```
Build jaxlib and JAX. These settings will make the build take approximately forever, but at least your Nano won't blow up. 
```
cd /d/jax && python3 build/build.py --bazel_path /d/bazel --enable_cuda --bazel_options='--jobs=2'
```                                              
Finally, build the wheels.
```
python3 setup.py bdist_wheel
cd build && JAX_CUDA_VERSION=10.2 python3 setup.py bdist_wheel
```
