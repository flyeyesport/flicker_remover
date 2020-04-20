//
// Created by jarek on 05.04.19.
//

#include "flicker_remover.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;

using std::to_string;


const double FlickerRemover::FIRST_TIMESTAMP = -1;


FlickerRemover::FlickerRemover(OpenCLKernels &opencl_kernels, unsigned int camera_fps, int flickering_threshold,
                               int max_allowed_flicker_duration, int frame_rows, int frame_cols)
        : opencl_kernels(opencl_kernels), frame_rows(frame_rows), frame_cols(frame_cols),
          expected_timestamp(FIRST_TIMESTAMP), timestamps_delta(1000.0 / camera_fps),
          accepted_timestamp_difference(timestamps_delta / 3),
          flicker_counter(frame_rows, frame_cols, CV_8U, Scalar(0)),
          flickering_threshold(flickering_threshold), max_allowed_flicker_duration(max_allowed_flicker_duration),
          corresponding_frames_similarity_sum(frame_rows, frame_cols, CV_8U, Scalar(0)), frames_block(0),
          corresponding_frames_similarity_levels(0),
          adjacent_frames_similarity_sum(frame_rows, frame_cols, CV_8U, Scalar(0)), adjacent_frames_similarity_levels(0)
{
    //calculate number of masks
    const unsigned int current_frequency = 50;
    if(camera_fps <= current_frequency) {
        throw std::runtime_error("Camera fps cannot be equal or smaller than power line frequency (50Hz) for flicker remover to work properly.");
    }

    const unsigned int max_number_of_masks = current_frequency;
    unsigned int i = 1;
    unsigned int count = 1;
    while(i <= max_number_of_masks) {
        count = i * camera_fps / current_frequency;
        if(count == 1.0 * i * camera_fps / current_frequency) {
            break;
        }
        i++;
    }
    number_of_masks = count - 1;
    block_size = count;
    frames_block.setMaxSize(block_size);
    masks.reserve(number_of_masks);
    for(unsigned int j = 0; j < number_of_masks; j++) {
        masks.emplace_back(frame_rows, frame_cols, CV_16S, Scalar(0));
    }
    actual_mask = number_of_masks;
    corresponding_frames_similarity_levels.setMaxSize(block_size);
    adjacent_frames_similarity_levels.setMaxSize(block_size - 1);
    for(unsigned int j = 0; j < block_size; j++) {
        corresponding_frames_similarity_levels.push(new UMat(frame_rows, frame_cols, CV_8UC1, Scalar(0)));
    }
    for(unsigned int j = 0; j < block_size - 1; j++) {
        adjacent_frames_similarity_levels.push(new UMat(frame_rows, frame_cols, CV_8UC1, Scalar(0)));
    }
}

FlickerRemover::~FlickerRemover()
{
    clear();
}

UMat *FlickerRemover::removeFlickering(const UMat &frame, double timestamp, string &error)
{
    if(frame.rows != frame_rows || frame.cols != frame_cols) {
        error = "Flickering cannot be removed. Size of the frame: " + to_string(frame.cols) + "x" +
                to_string(frame.rows) + " is different than expected: " + to_string(frame_cols) + "x" +
                to_string(frame_rows) + ".";
        return nullptr;
    }
    if(!timestampIsCloseToExpectedTimestamp(timestamp)) {
        //very unlikely. Should not happen...
        if(timestamp < expected_timestamp) {
            error = "Received unexpected timestamp: " + to_string(timestamp) + " Expected value close to: " +
                    to_string(expected_timestamp);
            return nullptr;
        }
        //calculate number of frames that were dropped
        auto number_of_dropped = (unsigned int) ((timestamp - expected_timestamp + accepted_timestamp_difference) /
                                                 timestamps_delta);
        actual_mask = (actual_mask + number_of_dropped) % (number_of_masks + 1);
        expected_timestamp = timestamp;
    }
    calculateNextExpectedTimestamp(timestamp);

    auto frame_copy = new UMat(frame_rows, frame_cols, CV_8UC1);

    if(actual_mask == number_of_masks) {
        actual_mask = 0;
        frame.convertTo(*frame_copy, CV_8UC1);
    } else {
        subtract(frame, masks[actual_mask], *frame_copy, noArray(), CV_8UC1);
        actual_mask++;
    }

    auto last_frame = frames_block.last();
    if(last_frame != nullptr) {
        auto new_adjacent_similarity = new UMat(frame_rows, frame_cols, CV_8UC1, Scalar(0));
        auto old_adjacent_similarity = adjacent_frames_similarity_levels.push(new_adjacent_similarity);

        auto ret = opencl_kernels.runKernelUpdateSimilarityLevels(*last_frame, *frame_copy, *old_adjacent_similarity,
                                                                  flickering_threshold, *new_adjacent_similarity,
                                                                  adjacent_frames_similarity_sum, error);
        delete old_adjacent_similarity;
        if(!ret) {
            delete frame_copy;
            return nullptr;
        }
    }

    //push returns pointer to the allocated earlier matrix, but do not delete, since we already returned this pointer
    //outside of this method, and it is the responsibility of the caller to delete this pointer.
    auto prev_frame = frames_block.push(frame_copy);

    if(prev_frame != nullptr) {
        auto new_similarity_levels = new UMat(frame_rows, frame_cols, CV_8UC1, Scalar(0));
        auto old_similarity_levels = corresponding_frames_similarity_levels.push(new_similarity_levels);

        auto ret = opencl_kernels.runKernelUpdateSimilarityLevels(*prev_frame, *frame_copy, *old_similarity_levels,
                                                                  flickering_threshold, *new_similarity_levels,
                                                                  corresponding_frames_similarity_sum, error);
        delete old_similarity_levels;
        if(!ret) {
            return nullptr;
        }
    }

    if(actual_mask == number_of_masks && frames_block.isFull()) {
        auto ret = opencl_kernels.runKernelUpdateFlickerCounter(adjacent_frames_similarity_sum, number_of_masks,
                                                                corresponding_frames_similarity_sum,
                                                                0.7f * (float) block_size, flicker_counter, error);
        if(!ret) {
            return nullptr;
        }
        for(int i = 0; i < (int) number_of_masks; i++) {
            ret = opencl_kernels.runKernelUpdateMasks(*(frames_block[0]), *(frames_block[i + 1]), flicker_counter,
                                                      max_allowed_flicker_duration, masks[i], error);
            if(!ret) {
                return nullptr;
            }
        }
        ret = opencl_kernels.runKernelZeroFlickerCounter(max_allowed_flicker_duration, flicker_counter, error);
        if(!ret) {
            return nullptr;
        }
    }
    return frame_copy;
}

bool FlickerRemover::timestampIsCloseToExpectedTimestamp(double timestamp) const
{
    return (expected_timestamp == FIRST_TIMESTAMP ||
            abs(expected_timestamp - timestamp) < accepted_timestamp_difference);
}

void FlickerRemover::calculateNextExpectedTimestamp(double timestamp)
{
    expected_timestamp = timestamp + timestamps_delta;
}

unsigned int FlickerRemover::getNumberOfStoredFrames() const
{
    return block_size;
}

void FlickerRemover::reset()
{
    clear();
    corresponding_frames_similarity_levels.clear();
    adjacent_frames_similarity_levels.clear();
    flicker_counter.setTo(Scalar(0));
    corresponding_frames_similarity_sum.setTo(Scalar(0));
    adjacent_frames_similarity_sum.setTo(Scalar(0));
    masks.clear();
    masks.reserve(number_of_masks);
    for(unsigned int j = 0; j < number_of_masks; j++) {
        masks.emplace_back(frame_rows, frame_cols, CV_16S, Scalar(0));
    }
    actual_mask = number_of_masks;
    for(unsigned int j = 0; j < block_size; j++) {
        corresponding_frames_similarity_levels.push(new UMat(frame_rows, frame_cols, CV_8UC1, Scalar(0)));
    }
    for(unsigned int j = 0; j < block_size - 1; j++) {
        adjacent_frames_similarity_levels.push(new UMat(frame_rows, frame_cols, CV_8UC1, Scalar(0)));
    }
    frames_block.clear();
}

void FlickerRemover::clear() {
    auto to_delete_1 = corresponding_frames_similarity_levels.pop();
    while(to_delete_1 != nullptr) {
        delete to_delete_1;
        to_delete_1 = corresponding_frames_similarity_levels.pop();
    }
    auto to_delete_2 = adjacent_frames_similarity_levels.pop();
    while(to_delete_2 != nullptr) {
        delete to_delete_2;
        to_delete_2 = adjacent_frames_similarity_levels.pop();
    }
}

bool FlickerRemover::getMaskOfStaticPixelsOfLastPairOfFrames(Mat &mask, string &error) const
{
    if(adjacent_frames_similarity_levels.isEmpty()) {
        error = "Flicker remover has to process at least 2 frames to be able to compare 2 consecutive frames and "
                "distinguish pixels that belong to moving objects from pixels that belong to static objects. "
                "Flicker remover processed less than 2 frames and cannot generate a requested mask.";
        return false;
    } else {
        adjacent_frames_similarity_levels.last()->copyTo(mask);
//        int x = 0;
//        for(unsigned int row = 0; row < mask.rows; ++row) {
//            for(unsigned int col = 0; col < mask.cols; ++col) {
//                if(mask.at<unsigned char>(row, col) != 0) {
//                    x++;
//                }
//            }
//        }
//        auto source = frames_block.last()->getMat(ACCESS_READ);
//        auto source_prev = frames_block[-2]->getMat(ACCESS_READ);
//
//        int y = 0;
//        for(unsigned int row = 0; row < mask.rows; ++row) {
//            for(unsigned int col = 0; col < mask.cols; ++col) {
//                if(abs(source.at<unsigned char>(row, col) - source_prev.at<unsigned char>(row, col)) <= flickering_threshold) {
//                    y++;
//                }
//            }
//        }
//        std::cout << "mask sum: " << x << " " << y << " " << (x != y ? "X" : ".") << std::endl;
//        std::cout << x << std::endl;
        return true;
    }
}

unsigned int FlickerRemover::getWarmUpDuration() const
{
    return block_size * (max_allowed_flicker_duration + 2);
}
