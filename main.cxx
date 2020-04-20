#include <filesystem>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include "circular_buffer.hpp"
#include "open_cl_kernels.hpp"
#include "flicker_remover.hpp"
#include "flicker_remover_cpu.hpp"

using namespace cv;
using namespace std::filesystem;
using namespace std;

double wallTime()
{
    struct timeval time;
    if(gettimeofday(&time, NULL) != 0) {
        //  Handle error
        return 0;
    }
    return (double) time.tv_sec + (double) time.tv_usec * .000001;
}

double norm(const Mat &frame_a, const Mat &frame_b, const Mat &mask)
{
    double diff = 0;
    unsigned int sum = 0;
    for(int row = 0; row < mask.rows; ++row) {
        for(int col = 0; col < mask.cols; ++col) {
            if(mask.at<unsigned char>(row, col) != 0) {
                sum++;
                if(frame_a.type() == CV_8UC1 || frame_a.type() == CV_8U) {
                    diff += pow(frame_a.at<unsigned char>(row, col) - frame_b.at<unsigned char>(row, col), 2);
                } else if(frame_a.type() == CV_32S || frame_a.type() == CV_32SC1) {
                    diff += pow(frame_a.at<int>(row, col) - frame_b.at<int>(row, col), 2);
                }
            }
        }
    }
    if(sum == 0) {
        return 0;
    } else {
        return diff / sum;
    }
}

unsigned int numberOfWhiteNeighbours(const Mat &image, int row, int col, int radius, int threshold)
{
    unsigned int count = 0;
    const int min_x = max(col - radius, 0);
    const int max_x = min(col + radius, image.cols - 1);
    const int min_y = max(row - radius, 0);
    const int max_y = min(row + radius, image.rows - 1);
    for(int x = min_x; x <= max_x; x++) {
        for(int y = min_y; y <= max_y; y++) {
            if(image.at<unsigned char>(y, x) > threshold) {
                count++;
            }
        }
    }
    if(image.at<unsigned char>(row, col) > threshold) {
        count--;
    }
    return count;
}

bool similar(int a, int b, int delta)
{
    return (abs(a - b) <= delta);
}

void readFilenames(const string &directory, vector<path> &filenames)
{
    auto directory_path = path(directory);
    cout << "path: " << directory_path << endl;
    auto input_images = directory_iterator(directory_path);
    auto end = directory_iterator();

    while(input_images != end) {
        const path &p = input_images->path();
        if(exists(p) && is_regular_file(p) && p.extension() == ".jpg") {
            filenames.emplace_back(p);
        }
        ++input_images;
    }
    //sort the filenames (filesystem iterator does not have this functionality)
    sort(filenames.begin(), filenames.end());
}

bool readImage(const vector<path> &filenames, unsigned int frame_number, Mat &image_buffer)
{
    image_buffer = imread(filenames[frame_number].string(), IMREAD_GRAYSCALE);
    if(image_buffer.data == nullptr) {
        cerr << "Cannot read image:" << filenames[frame_number].string() << endl;
        return false;
    }
    return true;
}

bool readVideoFrame(VideoCapture &video_capture, Mat &image_buffer)
{
    if(!video_capture.isOpened()) {
        cerr << "Video stream is not opened for reading." << endl;
        return false;
    } else {
        video_capture >> image_buffer;
        if(!image_buffer.empty() && image_buffer.channels() > 1) {
            cvtColor(image_buffer, image_buffer, COLOR_BGR2GRAY);
        }
    }
    return true;
}

bool openVideo(VideoCapture &video_capture, const string &movie_path)
{
    if(!video_capture.open(movie_path)) {
        video_capture.release();
        cerr << "Can't open movie: " + movie_path << endl;
        return false;
    }
    return true;
}

bool simpleDiff(const Mat &prev_frame, const Mat &actual_frame)
{
    Mat diff;
    absdiff(prev_frame, actual_frame, diff);
    imshow("diff", diff);
    waitKey(1);
    return true;
}

bool simpleDiffMaxed(const Mat &prev_frame, const Mat &actual_frame)
{
    const unsigned char BLACK = 0;
    const unsigned char WHITE = 255;

    Mat diff;
    absdiff(prev_frame, actual_frame, diff);
    diff.forEach<unsigned char>([](unsigned char &value, const int *position) {
        if(value != BLACK) {
            value = WHITE;
        }
    });
    imshow("diff", diff);
    waitKey(1);
    return true;
}


bool iterateImages(const vector<path> &filenames, const function<bool(const Mat &, const Mat &)> &callback_for_pair)
{
    unsigned int frame_number = 0;
    Mat frames[2];
    Mat *prev_frame = &(frames[1]);
    Mat *actual_frame = &(frames[0]);
    while(frame_number < filenames.size()) {
        if(!readImage(filenames, frame_number, *actual_frame)) {
            return false;
        }
        if(!prev_frame->empty()) {
            if(!callback_for_pair(*prev_frame, *actual_frame)) {
                return false;
            }

        }
        Mat *tmp;
        tmp = prev_frame;
        prev_frame = actual_frame;
        actual_frame = tmp;
        frame_number++;
    }
    return true;
}

bool iterateVideoFrames(VideoCapture &video_capture, const function<bool(const Mat &, const Mat &)> &callback_for_pair)
{
    Mat frames[2];
    Mat *prev_frame = &(frames[1]);
    Mat *actual_frame = &(frames[0]);
    while(true) {
        if(!readVideoFrame(video_capture, *actual_frame)) {
            return false;
        }
        if(actual_frame->empty()) {
            return true;
        }
        if(!prev_frame->empty()) {
            if(!callback_for_pair(*prev_frame, *actual_frame)) {
                return false;
            }
        }
        Mat *tmp;
        tmp = prev_frame;
        prev_frame = actual_frame;
        actual_frame = tmp;
    }
}

int flickerRemoverOnCPU(bool images_from_dir, VideoCapture &video_capture, const vector<path> &filenames,
                        unsigned int fps, int rows, int cols)
{
    FlickerRemoverCPU flicker_remover(fps, 5, 3, rows, cols);
    auto skip_frames = flicker_remover.getWarmUpDuration();

    VideoWriter video_orig("orig.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(cols, rows), false);
    VideoWriter video_flicker_free("flicker_free.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(cols, rows),
                                   false);
    VideoWriter video_diff("diff.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(cols, rows), false);
    VideoWriter video_combined("combined.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(2 * cols, 2 * rows),
                               false);

    const unsigned int low_threshold = 10;
    const unsigned char WHITE = 255;
    const unsigned char BLACK = 0;
    //you may change it from 8 to 6 or even 3 for example 3
    const unsigned int second_neighbours_limit = 8;
    const unsigned int second_radius = 1;

    const double timestamps_delta = 1000. / fps;
    double fake_timestamp = 34.0;
    unsigned int frame_number = 0;
    Mat prev_orig;
    Mat *prev_frame = nullptr;
    CircularBuffer<Mat *> to_delete_in_future(flicker_remover.getNumberOfStoredFrames());
    double total_time = 0;
    bool was_error = false;
    double norm_sum = 0;
    double orig_norm_sum = 0;
    unsigned int norm_count = 0;
    while(!images_from_dir || frame_number < filenames.size()) {
        Mat orig_frame;
        if(images_from_dir) {
            if(!readImage(filenames, frame_number, orig_frame)) {
                was_error = true;
                break;
            }
        } else {
            if(!readVideoFrame(video_capture, orig_frame)) {
                was_error = true;
                break;
            }
            if(orig_frame.empty()) {
                break;
            }
        }

        string error;
        auto start = wallTime();
        Mat *frame_without_flickering = flicker_remover.removeFlickering(orig_frame, fake_timestamp, error);
        auto end = wallTime();
        total_time += (end - start);
        if(frame_without_flickering == nullptr) {
            cout << "Flicker remover reported an error: " << error << endl;
            was_error = true;
            break;
        }

        Mat frame_without_flickering_8u;
        frame_without_flickering->convertTo(frame_without_flickering_8u, CV_8UC1);
        imshow("image with removed flickering", frame_without_flickering_8u);
        imshow("original image", orig_frame);
        waitKey(1);

        video_orig.write(orig_frame);
        video_flicker_free.write(frame_without_flickering_8u);

        if(prev_frame != nullptr) {
            if(skip_frames < frame_number) {
                Mat mask;
                if(flicker_remover.getMaskOfStaticPixelsOfLastPairOfFrames(mask, error)) {
                    norm_sum += norm(*prev_frame, *frame_without_flickering, mask);
                    orig_norm_sum += norm(prev_orig, orig_frame, mask);
                    norm_count++;
                } else {
                    cout << error << endl;
                    was_error = true;
                    break;
                }
            }
            Mat prev_frame_8u;
            prev_frame->convertTo(prev_frame_8u, CV_8UC1);
            Mat diff;
            absdiff(prev_frame_8u, frame_without_flickering_8u, diff);
            Mat filtered_diff(diff.rows, diff.cols, diff.type());

            filtered_diff.forEach<unsigned char>([&diff](unsigned char &value, const int *position) {
                int row = position[0];
                int col = position[1];
                if(diff.at<unsigned char>(row, col) > low_threshold &&
                   numberOfWhiteNeighbours(diff, row, col, second_radius, low_threshold) >= second_neighbours_limit) {
                    //pixel belongs to moving object
                    value = WHITE;
                } else {
                    //pixel belongs to background
                    value = BLACK;
                }
            });

            imshow("diff after flickering remove", filtered_diff);
            waitKey(1);
            video_diff.write(filtered_diff);
            Mat v1, v2, v3;
            hconcat(orig_frame, frame_without_flickering_8u, v1);
            Mat diff_orig;
            absdiff(prev_orig, orig_frame, diff_orig);
            diff_orig.forEach<unsigned char>([](unsigned char &value, const int *position) {
                if(value > low_threshold) {
                    value = WHITE;
                }
            });
            hconcat(diff_orig, filtered_diff, v2);
            vconcat(v1, v2, v3);
            video_combined.write(v3);
        }
        orig_frame.copyTo(prev_orig);
        auto to_delete = to_delete_in_future.push(prev_frame);
        delete to_delete;
        prev_frame = frame_without_flickering;
        fake_timestamp += timestamps_delta;
        frame_number++;
    }
    delete prev_frame;
    auto to_delete = to_delete_in_future.pop();
    while(to_delete != nullptr) {
        delete to_delete;
        to_delete = to_delete_in_future.pop();
    }
    video_orig.release();
    video_flicker_free.release();
    video_diff.release();
    video_combined.release();

    if(was_error) {
        return -1;
    } else {
        cout << "TOTAL TIME: " << total_time << " for: " << frame_number << " frames.";
        if(frame_number > 0) {
            cout << " Average: " << (total_time / frame_number);
            if(frame_number > skip_frames) {
                cout << " Norm with flicker removal: " << (norm_sum / norm_count);
                cout << " Norm without flicker removal: " << (orig_norm_sum / norm_count);
            }
        }
        cout << endl;
        return 0;
    }
}


int flickerRemoverOnGPU(bool images_from_dir, VideoCapture &video_capture, const vector<path> &filenames,
                        unsigned int fps, int rows, int cols)
{
    OpenCLKernels opencl_kernels;
    FlickerRemover flicker_remover(opencl_kernels, fps, 5, 3, rows, cols);
    auto skip_frames = flicker_remover.getWarmUpDuration();

    VideoWriter video_orig("orig.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(cols, rows), false);
    VideoWriter video_flicker_free("flicker_free.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(cols, rows),
                                   false);
    VideoWriter video_diff("diff.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(cols, rows), false);
    VideoWriter video_combined("combined.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(2 * cols, 2 * rows),
                               false);

    const unsigned int low_threshold = 10;
    const unsigned char WHITE = 255;

    //you may change it from 8 to 6 or even 3 for example 3
    const unsigned int second_neighbours_limit = 8;
    const unsigned int second_radius = 1;

    const double timestamps_delta = 1000. / fps;
    double fake_timestamp = 34.0;
    unsigned int frame_number = 0;
    Mat prev_orig;
    UMat *prev_frame = nullptr;
    CircularBuffer<UMat *> to_delete_in_future(flicker_remover.getNumberOfStoredFrames());
    double total_time = 0;
    bool was_error = false;
    double norm_sum = 0;
    double orig_norm_sum = 0;
    unsigned int norm_count = 0;
    while(!images_from_dir || frame_number < filenames.size()) {
        Mat orig_frame;
        if(images_from_dir) {
            if(!readImage(filenames, frame_number, orig_frame)) {
                was_error = true;
                break;
            }
        } else {
            if(!readVideoFrame(video_capture, orig_frame)) {
                was_error = true;
                break;
            }
            if(orig_frame.empty()) {
                break;
            }
        }

        string error;
        auto start = wallTime();
        UMat *frame_without_flickering = flicker_remover.removeFlickering(orig_frame.getUMat(ACCESS_READ),
                                                                          fake_timestamp, error);
        auto end = wallTime();
        total_time += (end - start);
        if(frame_without_flickering == nullptr) {
            cout << "Flicker remover reported an error: " << error << endl;
            was_error = true;
            break;
        }

        imshow("image with removed flickering", *frame_without_flickering);
        imshow("original image", orig_frame);
        waitKey(1);

        video_orig.write(orig_frame);
        video_flicker_free.write(*frame_without_flickering);

        if(prev_frame != nullptr) {
            if(skip_frames < frame_number) {
                Mat mask;
                if(flicker_remover.getMaskOfStaticPixelsOfLastPairOfFrames(mask, error)) {
                    norm_sum += norm(prev_frame->getMat(ACCESS_READ), frame_without_flickering->getMat(ACCESS_READ),
                                     mask);
                    orig_norm_sum += norm(prev_orig, orig_frame, mask);
                    norm_count++;
                } else {
                    cout << error << endl;
                    was_error = true;
                    break;
                }
            }

            UMat diff;
            absdiff(*prev_frame, *frame_without_flickering, diff);
            UMat filtered_diff(diff.rows, diff.cols, diff.type());
            if(!opencl_kernels.runKernelCalculateFilteredDiff(diff, low_threshold, second_neighbours_limit,
                                                              filtered_diff, error)) {
                cout << "OpenCL kernel reported an error: " << error << endl;
                was_error = true;
                break;
            }

            imshow("diff after flickering remove", filtered_diff);
            waitKey(1);
            video_diff.write(filtered_diff);
            Mat v1, v2, v3;
            hconcat(orig_frame, *frame_without_flickering, v1);
            Mat diff_orig;
            absdiff(prev_orig, orig_frame, diff_orig);
            diff_orig.forEach<unsigned char>([](unsigned char &value, const int *position) {
                if(value > low_threshold) {
                    value = WHITE;
                }
            });

            hconcat(diff_orig, filtered_diff, v2);
            vconcat(v1, v2, v3);
            video_combined.write(v3);
        }
        orig_frame.copyTo(prev_orig);
        auto to_delete = to_delete_in_future.push(prev_frame);
        delete to_delete;
        prev_frame = frame_without_flickering;
        fake_timestamp += timestamps_delta;
        frame_number++;
    }
    delete prev_frame;
    auto to_delete = to_delete_in_future.pop();
    while(to_delete != nullptr) {
        delete to_delete;
        to_delete = to_delete_in_future.pop();
    }
    video_orig.release();
    video_flicker_free.release();
    video_diff.release();
    video_combined.release();

    if(was_error) {
        return -1;
    } else {
        cout << "TOTAL TIME: " << total_time << " for: " << frame_number << " frames.";
        if(frame_number > 0) {
            cout << " Average: " << (total_time / frame_number);
            if(norm_count > 0) {
                cout << " Norm with flicker removal: " << (norm_sum / norm_count);
                cout << " Norm without flicker removal: " << (orig_norm_sum / norm_count);
            }
        }
        cout << endl;
        return 0;
    }
}

int main(int argc, char *argv[])
{
    vector<path> filenames;
    VideoCapture video_capture;
    unsigned int fps;
    const int rows = 600;
    const int cols = 800;

    if(argc != 4) {
        cout << "Expected 3 arguments. Usage: " << argv[0]
             << " <path to directory with jpeg images | movie filename> <execution mode> <fps>" << endl
             << "Available execution modes: " << endl
             << "1 - simple diff" << endl
             << "2 - simple diff with all pixels with values different than 0 (black) set to 255 (white)" << endl
             << "3 - flicker remover on CPU" << endl
             << "4 - flicker remover on GPU" << endl
             << "IMPORTANT: all images and videos should be in << " << cols << "x" << rows << " pixel format." << endl;
        return -1;
    }
    string filename = argv[1];

    string fps_string = argv[3];
    try {
        size_t pos;
        int tmp_fps = stoi(fps_string, &pos);
        if(pos < fps_string.size()) {
            cerr << "Error in <fps> command line parameter. Trailing characters after number: " << fps_string << endl;
            return -1;
        } else if(tmp_fps < 0) {
            cerr << "Error in <fps> command line parameter. Frames per second must have a positive value." << endl;
            return -1;
        } else {
            fps = (unsigned int) tmp_fps;
        }
    } catch(invalid_argument const &ex) {
        cerr << "Error in <fps> command line parameter. Invalid number: " << fps_string << endl;
        return -1;
    } catch(out_of_range const &ex) {
        cerr << "Error in <fps> command line parameter. Number out of range: " << fps_string << endl;
        return -1;
    }

    auto directory_path = path(filename);
    bool images_from_dir = false;
    if(is_directory(directory_path)) {
        images_from_dir = true;
        readFilenames(filename, filenames);
        if(filenames.empty()) {
            cerr << "Could not find any jpeg files in directory: "<< filename << "." << endl;
            return -1;
        }
    } else {
        if(!openVideo(video_capture, directory_path.string())) {
            return -1;
        }
    }

    int execution_mode = 0;
    string execution_mode_string = argv[2];
    try {
        size_t pos;
        execution_mode = stoi(execution_mode_string, &pos);
        if(pos < execution_mode_string.size()) {
            cerr << "Error in <execution mode> command line parameter. Trailing characters after number: "
                 << execution_mode_string << endl;
            return -1;
        }
    } catch(invalid_argument const &ex) {
        cerr << "Error in <execution mode> command line parameter. Invalid number: " << execution_mode_string << endl;
        return -1;
    } catch(out_of_range const &ex) {
        cerr << "Error in <execution mode> command line parameter. Number out of range: "
             << execution_mode_string << endl;
        return -1;
    }

    switch(execution_mode) {
        case 1:
            cout << "Simple absolute diff of 2 frames." << endl;
            if(images_from_dir) {
                if(iterateImages(filenames, simpleDiff)) {
                    return 0;
                } else {
                    cerr << "Error occurred while iterating over images to produce and display diffs." << endl;
                    return -1;
                }
            } else {
                if(iterateVideoFrames(video_capture, simpleDiff)) {
                    return 0;
                } else {
                    cerr << "Error occurred while iterating over frames from video file to produce and display diffs."
                         << endl;
                    return -1;
                }
            }
        case 2:
            cout << "Simple absolute diff of 2 frames with all pixels with values "
                 << "different than 0 (black) set to 255 (white)." << endl;
            if(images_from_dir) {
                if(iterateImages(filenames, simpleDiffMaxed)) {
                    return 0;
                } else {
                    cerr << "Error occurred while iterating over images to produce and display diffs with all pixels "
                         << "with values different than 0 (black) set to 255 (white)." << endl;
                    return -1;
                }
            } else {
                if(iterateVideoFrames(video_capture, simpleDiffMaxed)) {
                    return 0;
                } else {
                    cerr << "Error occurred while iterating over frames from video file to produce and display diffs "
                         << "with all pixels with values different than 0 (black) set to 255 (white)." << endl;
                    return -1;
                }
            }
        case 3:
            cout << "Flicker remover on CPU." << endl;
            return flickerRemoverOnCPU(images_from_dir, video_capture, filenames, fps, rows, cols);
        case 4:
            cout << "Flicker remover on GPU." << endl;
            return flickerRemoverOnGPU(images_from_dir, video_capture, filenames, fps, rows, cols);
        default:
            cout << "Unknown execution mode: " << execution_mode_string
                 << ". It should be an integral value from range: 1 - 6." << endl;
            return -1;
    }
}