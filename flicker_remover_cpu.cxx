//
// Created by jarek on 03.11.2019.
//

#include "flicker_remover_cpu.hpp"

using namespace cv;

using std::to_string;


const double FlickerRemoverCPU::FIRST_TIMESTAMP = -1;


FlickerRemoverCPU::FlickerRemoverCPU(unsigned int camera_fps, int flickering_threshold,
                                     int max_allowed_flicker_duration, int frame_rows, int frame_cols)
        : frame_rows(frame_rows), frame_cols(frame_cols),
          expected_timestamp(FIRST_TIMESTAMP), timestamps_delta(1000.0 / camera_fps),
          accepted_timestamp_difference(timestamps_delta / 3), frames_block(0),
          flicker_counter(frame_rows, frame_cols, CV_8U, Scalar(0)), flickering_threshold(flickering_threshold),
          max_allowed_flicker_duration(max_allowed_flicker_duration), corresponding_frames_similarity_levels(0),
          corresponding_frames_similarity_sum(frame_rows, frame_cols, CV_8U, Scalar(0)),
          adjacent_frames_similarity_levels(0), adjacent_frames_similarity_sum(frame_rows, frame_cols, CV_8U, Scalar(0))
{
    //calculate number of masks
    const unsigned int current_frequency = 50;
    if(camera_fps <= current_frequency) {
        throw std::runtime_error(
                "Camera fps cannot be equal or smaller than power line frequency (50Hz) for flicker remover to work properly.");
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
        masks.emplace_back(frame_rows, frame_cols, CV_32S, Scalar(0));
    }
    actual_mask = number_of_masks;
    corresponding_frames_similarity_levels.setMaxSize(block_size);
    adjacent_frames_similarity_levels.setMaxSize(block_size - 1);
    for(unsigned int j = 0; j < block_size; j++) {
        corresponding_frames_similarity_levels.push(
                new BooleanArray2D((unsigned int) frame_rows, (unsigned int) frame_cols));
    }
    for(unsigned int j = 0; j < block_size - 1; j++) {
        adjacent_frames_similarity_levels.push(
                new BooleanArray2D((unsigned int) frame_rows, (unsigned int) frame_cols));
    }
}

FlickerRemoverCPU::~FlickerRemoverCPU()
{
    clear();
}

Mat *FlickerRemoverCPU::removeFlickering(const Mat &frame, double timestamp, string &error)
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

    auto frame_copy = new Mat();

    if(actual_mask == number_of_masks) {
        actual_mask = 0;
        frame.convertTo(*frame_copy, CV_32S);
    } else {
        subtract(frame, masks[actual_mask], *frame_copy, noArray(), CV_32S);
        actual_mask++;
    }

    auto last_frame = frames_block.last();
    if(last_frame != nullptr) {
        auto new_similarity_levels = new BooleanArray2D((unsigned int) frame_rows, (unsigned int) frame_cols);
        auto old_similarity_levels = adjacent_frames_similarity_levels.push(new_similarity_levels);

        last_frame->forEach<int>(
                [this, &frame_copy, new_similarity_levels, old_similarity_levels](int &value, const int *position) {
                    int row = position[0];
                    int col = position[1];
                    bool pixels_are_similar = similar(value, frame_copy->at<int>(row, col));
                    new_similarity_levels->set((unsigned int) row, (unsigned int) col, pixels_are_similar);
                    adjacent_frames_similarity_sum.at<unsigned char>(row, col) +=
                            (unsigned char) pixels_are_similar -
                            (unsigned char) old_similarity_levels->at((unsigned int) row, (unsigned int) col);
                });
        delete old_similarity_levels;
    }

    //push returns pointer to the allocated earlier matrix, but do not delete, since we already returned this pointer
    //outside of this method, and it is the responsibility of the caller to delete this pointer.
    auto prev_frame = frames_block.push(frame_copy);

    if(prev_frame != nullptr) {
        auto new_similarity_levels = new BooleanArray2D((unsigned int) frame_rows, (unsigned int) frame_cols);
        auto old_similarity_levels = corresponding_frames_similarity_levels.push(new_similarity_levels);
        prev_frame->forEach<int>(
                [this, &frame_copy, new_similarity_levels, old_similarity_levels](int &value, const int *position) {
                    int row = position[0];
                    int col = position[1];
                    bool pixels_are_similar = similar(value, frame_copy->at<int>(row, col));
                    new_similarity_levels->set((unsigned int) row, (unsigned int) col, pixels_are_similar);
                    corresponding_frames_similarity_sum.at<unsigned char>(row, col) +=
                            (unsigned char) pixels_are_similar -
                            (unsigned char) old_similarity_levels->at((unsigned int) row, (unsigned int) col);
                });
        delete old_similarity_levels;
    }

    if(actual_mask == number_of_masks && frames_block.isFull()) {
        flicker_counter.forEach<unsigned char>([this, &frame_copy](unsigned char &value, const int *position) {
            int row = position[0];
            int col = position[1];
            if(corresponding_frames_similarity_sum.at<unsigned char>(row, col) > 0.7 * block_size) {
                bool values_similar = true;
                unsigned int block_number = 0;
                while(values_similar && block_number + 1 < frames_block.maxSize()) {
                    values_similar = similar(frames_block[(int) block_number]->at<int>(row, col),
                                             frames_block[(int) block_number + 1]->at<int>(row, col));
                    block_number++;
                }
                if(!values_similar) {
                    value++;
                } else {
                    value = 0;
                }
            } else {
                value = 0;
            }
            if(value > max_allowed_flicker_duration) {
                for(int i = 0; i < (int) number_of_masks; i++) {
                    masks[i].at<int>(row, col) +=
                            frames_block[i + 1]->at<int>(row, col) - frames_block[0]->at<int>(row, col);
                }
                value = 0;
                //subtract mask from frame copy, but be sure that result is between 0-255
                auto mask_val = masks[number_of_masks - 1].at<int>(row, col);
                auto &frame_copy_val = frame_copy->at<int>(row, col);
                if(mask_val >= 0) {
                    if(frame_copy_val >= mask_val) {
                        frame_copy_val -= mask_val;
                    } else {
                        frame_copy_val = 0;
                    }
                } else {
                    if((int) frame_copy_val - mask_val > 255) {
                        frame_copy_val = 255;
                    } else {
                        frame_copy_val -= mask_val;
                    }
                }
            }
        });
    }
    return frame_copy;
}

bool FlickerRemoverCPU::similar(int a, int b) const
{
    return (abs(a - b) <= flickering_threshold);
}

bool FlickerRemoverCPU::timestampIsCloseToExpectedTimestamp(double timestamp) const
{
    return (expected_timestamp == FIRST_TIMESTAMP ||
            abs(expected_timestamp - timestamp) < accepted_timestamp_difference);
}

void FlickerRemoverCPU::calculateNextExpectedTimestamp(double timestamp)
{
    expected_timestamp = timestamp + timestamps_delta;
}

unsigned int FlickerRemoverCPU::getNumberOfStoredFrames() const
{
    return frames_block.maxSize();
}

void FlickerRemoverCPU::reset()
{
    clear();
    frames_block.clear();
    corresponding_frames_similarity_levels.clear();
    adjacent_frames_similarity_levels.clear();
    flicker_counter.setTo(Scalar(0));
    corresponding_frames_similarity_sum.setTo(Scalar(0));
    masks.clear();
    masks.reserve(number_of_masks);
    for(unsigned int j = 0; j < number_of_masks; j++) {
        masks.emplace_back(frame_rows, frame_cols, CV_32S, Scalar(0));
    }
    actual_mask = number_of_masks;
    for(unsigned int j = 0; j < block_size; j++) {
        corresponding_frames_similarity_levels.push(
                new BooleanArray2D((unsigned int) frame_rows, (unsigned int) frame_cols));
    }
    for(unsigned int j = 0; j < block_size - 1; j++) {
        adjacent_frames_similarity_levels.push(
                new BooleanArray2D((unsigned int) frame_rows, (unsigned int) frame_cols));
    }
}

void FlickerRemoverCPU::clear()
{
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

bool FlickerRemoverCPU::getMaskOfStaticPixelsOfLastPairOfFrames(Mat &mask, string &error) const
{
    if(frames_block.size() < 2) {
        error = "Flicker remover has to process at least 2 frames to be able to compare 2 consecutive frames and "
                "distinguish pixels that belong to moving objects from pixels that belong to static objects. "
                "Flicker remover processed less than 2 frames and cannot generate a requested mask.";
        return false;
    } else {
        auto source = frames_block.last();
        auto source_prev = frames_block[-2];
        mask = Mat::zeros(source->rows, source->cols, CV_8UC1);
//        int y = 0;
        for(unsigned int row = 0; row < source->rows; ++row) {
            for(unsigned int col = 0; col < source->cols; ++col) {
                if(similar(source->at<int>(row, col), source_prev->at<int>(row, col))) {
                    mask.at<unsigned char>(row, col) = 1;
//                    y++;
                }
            }
        }

//        auto source_2 = adjacent_frames_similarity_levels.last();
//        int x = 0;
//        for(unsigned int row = 0; row < source->rows; ++row) {
//            for(unsigned int col = 0; col < source->cols; ++col) {
//                if(source_2->at(row, col) != 0) {
//                    x++;
//                }
//            }
//        }

//        std::cout << x << " " << y << " " << (x != y ? "X" : ".") << std::endl;

        return true;
    }

//    if(adjacent_frames_similarity_levels.isEmpty()) {
//        error = "Flicker remover has to process at least 2 frames to be able to compare 2 consecutive frames and "
//                "distinguish pixels that belong to moving objects from pixels that belong to static objects. "
//                "Flicker remover processed less than 2 frames and cannot generate a requested mask.";
//        return false;
//    } else {
//        auto source = adjacent_frames_similarity_levels.last();
//        mask = Mat::zeros(source->rows, source->cols, CV_8UC1);
//        for(unsigned int row = 0; row < source->rows; ++row) {
//            for(unsigned int col = 0; col < source->cols; ++col) {
//                if(source->at(row, col) != 0) {
//                    mask.at<unsigned char>(row, col) = 1;
//                }
//            }
//        }
//        return true;
//    }
}

unsigned int FlickerRemoverCPU::getWarmUpDuration() const
{
    return block_size * (max_allowed_flicker_duration + 2);
}
