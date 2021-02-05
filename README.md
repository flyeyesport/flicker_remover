# flicker_remover
Source code for "[Realtime flicker removal for fast video streaming](https://link.springer.com/article/10.1007/s11042-020-10385-8)" paper.

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
To run and test you can use your own movies or sets of frames or our sets of frames and a movie used in our paper. **Important:** all movies and frames must be in 1-channel, 800x600 format. You can download our frames and a movie (examples 1-3 from our paper) from here: [example 1](https://1drv.ms/u/s!ApYchjX9LRlxjxyaUNrckiq6Orn4?e=VeD1Tc),
[example 2](https://1drv.ms/u/s!ApYchjX9LRlxjx30jepAl6u24O78?e=qFgtY5),
[example 3](https://1drv.ms/u/s!ApYchjX9LRlxjx7jSC62KXkUqvpe?e=pPVpqS).


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
./flicker_remover example_3/source_movie/source_movie.avi 4 150
```
For example 3 change value of `second_neighbours_limit` constant in the code, recompile, run and see the difference.

## Commandline parameters
Usage: 
```
./flicker_remover <path to directory with jpeg images | movie filename> <execution mode> <fps>
```
where:
* `<path to directory with jpeg images | movie filename>` is a directory with frames from the movie (it can be jpeg, png or other format that can be read by opencv) or a path to the movie in format that can be read by opencv.
* `<execution mode>` is a number from 1 to 4:
  + 1 - the program will not use flicker removal algorithm it will output only a differential images calculated for pairs of consecutive frames,
  + 2 - the same as 1, but all values of pixels of the differential images not equal to 0 will be set to 255,
  + 3 - flicker removal algorithm run on CPU,
  + 4 - flicker removal algorithm run on GPU (OpenCL).
* `<fps>` is a speed (frames per second) at which a movie or frames were recorded.

## Output
The program genarates and saves 4 movies in the current directory:
* orig.avi - original movie without any changes
* diff.avi - movie with only differential frames, like with `<execution mode>` set to 2
* flicker_free.avi - movie with flicker removal applied
* combined.avi - movie with 4 combined movies:
  + original, unchanged
  + with flicker removal applied
  + differential images from frames with removed flickering
  + differential images from frames with removed flickering with second level filter applied
