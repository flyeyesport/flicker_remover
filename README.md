# flicker_remover
Source code for "Realtime flicker removal for fast video streaming" paper.

We use it in quite a complex system as one of the first filters in online stream processing. The system processes 150-200fps and the filter alone is much faster.
We detect movement in streams by calculating differential images between consecutive frames. Streams are captured in places with 
artificial lighting causing flickering, and many false positives, and very poor results in detecting movement from differential images. Adding this filter 
allows us to remove flickering before calculating differential images. It is implemented for CPU and GPU (OpenCL).

We tested it only on Linux. You should install OpenCV before compilation.

We provide a simple main.cxx to test the code.

```
mkdir build && cd build && cmake .. && make && ./flicker_remover
```
