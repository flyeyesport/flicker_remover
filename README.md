# flicker_remover
Source code for "Realtime flicker removal for fast video streaming" paper (to be published soon).

We use it in quite a complex system as one of the first filters in online stream processing. The system processes 150-200fps and the filter alone is much faster.
We detect movement in streams by calculating differential images between consecutive frames. Streams are captured in places with 
artificial lighting causing flickering, and many false positives, and very poor results in detecting movement from differential images. Adding this filter 
allows us to remove flickering before calculating differential images. It is implemented for CPU and GPU (OpenCL).

We tested it only on Linux. You should install OpenCV (version 4.0 or higher) before compilation.

We provide a simple main.cxx to test the code.

## Building instructions:
```
mkdir build && cd build && cmake .. && make
```

## Running:
To run and test you can use your own movies or sets of frames or our sets of frames and a movie used in our paper. You can download our frames and a movie (examples 1-3 from our paper) from here: [example 1](https://www.google.com), 
[example 2](https://www.google.com), [example 3](https://www.google.com).


## Examples:
Unzip downloaded files in the directory where you built the program (where `flicker_executable` is).

For example 1 run:
```
./flicker_remover example_1/source_frames 4 190
```

For example 2 run:
```
./flicker_remover example_2/source_frames 4 150
```

For example 3 run:
```
./flicker_remover example_3/source_movie/source_movie.avi 4 190
```
For example 3 change value of `second_neighbours_limit` constant in the code, recompile, run and see the difference.

