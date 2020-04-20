//
// Created by jarek on 03.11.2019.
//

#ifndef FLICKER_REMOVER_CPU_HPP
#define FLICKER_REMOVER_CPU_HPP

#include <vector>
#include <opencv2/opencv.hpp>
#include <string>
#include "circular_buffer.hpp"
#include "boolean_array_2_d.hpp"

using cv::Mat;
using std::vector;
using std::string;

/**
 * @brief Class implementing simple algorithm to remove flickering in the consecutive frames captured by camera.
 * Flickering is caused by changes in the current. The artificial light usually is turned on and off 50 times per second.
 * It causes flickering which is very undesirable for our algorithms which detect shuttles.
 */
class FlickerRemoverCPU {
protected:
    /**
     * @brief Constant indicating that we expect first timestamp, so we do not know what to expect and it can be any
     * value.
     */
    static const double FIRST_TIMESTAMP;

    /**
     * @brief Expected, usual difference between timestamps of the consecutive frames. Calculated from camera's fps.
     * This value is stored in milliseconds.
     */
    const double timestamps_delta;

    /**
     * @brief Constant used to decide if given timestamp is similar to expected timestamp or they differ too much.
     * This value is stored in milliseconds.
     */
    const double accepted_timestamp_difference;

    /**
     * @brief Number of needed masks to be used to remove flickering. This number is calculated based on fps of the
     * camera and constant frequency of the current and light intensity changes (50Hz). For example for 150fps
     * every 3 frames we have the same lightning conditions, so we need 2 masks. For frame 0 we do not need mask because
     * it is our "ground level" frame, then for frame 1 we need mask to change lightning conditions to "ground level",
     * then for frame 2 we need different mask to also change lightning conditions to "ground level". Then comes frame 3
     * which has the same lightning conditions as frame 0 so we do not need mask, then frame 4 - the same lightning
     * conditions as frame 1 - we use first mask, then frame 5 - same as frame 2 - we use mask 2, then frame 6 - no
     * mask as with frames 0 and 3, and so on.
     */
    unsigned int number_of_masks;

    /**
     * @brief Number of frames stored in one block. It is equal to the number of masks + 1.
     */
    unsigned int block_size;

    /**
     * @brief identifier of the mask to be used in the next <b>removeFlickering()</b> method call. After each call it is
     * incremented. If it is bigger than number of masks, then the mask is not applied, but this number is decremented
     * back to 0, to start from the beginning. This way we apply consecutive masks to consecutive frames, but there
     * are frames that do not get masks applied - these frames we treat as "ground level" frames.
     */
    unsigned int actual_mask;

    /**
     * @brief Calculated masks which are used to remove flickering. They are applied to the consecutive frames.
     * Every time next mask is applied. When last mask is applied, then we make a brake for one frame, and then we
     * start over with first mask to be applied next.
     */
    vector<Mat> masks;

    /**
     * @brief Circular buffer of pointers to copies of historical frames. Number of frames is double the number of
     * frames per block. Number of frames per block is equal to number of masks plus 1.
     * These frames are used to detect flickering patterns for every pixel.
     */
    CircularBuffer<Mat *> frames_block;

    /**
     * @brief Circular buffer of pointers to special arrays with infos about similarities of corresponding frames from
     * different blocks. In each array there are as many boolean flags as there are pixels in the frame. For every
     * pixel position there is an information if pixels from corresponding frames are similar or not. Number of arrays
     * is equal number of masks plus 1.
     */
    CircularBuffer<BooleanArray2D *> corresponding_frames_similarity_levels;

    /**
     * @brief Circular buffer of pointers to special arrays with infos about similarities of adjacent frames from
     * last block. In each array there are as many boolean flags as there are pixels in the frame. For every pixel
     * position there is an information if pixels from adjacent frames are similar or not. Number of arrays is equal
     * number of masks.
     */
    CircularBuffer<BooleanArray2D *> adjacent_frames_similarity_levels;

    /**
     * @brief Special array with infos about levels of similarities of different blocks. The array is the sum of values
     * from all <b>corresponding_frames_similarity_levels</b>. It is used to speed up processing.
     */
    Mat corresponding_frames_similarity_sum;

    /**
     * @brief Special array with infos about levels of similarities of different blocks. The array is the sum of values
     * from all <b>adjacent_frames_similarity_levels</b>. It is used to speed up processing.
     */
    Mat adjacent_frames_similarity_sum;

    /**
     * @brief Internal matrix storing counts of last flickers for each pixel. Every similar block for given pixel is
     * counted as 1.
     */
    Mat flicker_counter;

    /**
     * @brief Flickering threshold. When we compare 2 values of the same pixel from 2 consecutive frames, this is the
     * threshold that is used to distinguish flickering pixels from not flickering ones.
     */
    const int flickering_threshold;

    /**
     * @brief Maximum number of consecutive blocks for which given pixel can have the same flickering pattern. If this
     * value is reached, values of the masks for this pixel are changed to remove flickering in next frames.
     */
    const int max_allowed_flicker_duration;

    /**
     * @brief Height of frames that we expect to calculate masks and then height of the frames that we expect during
     * flickering removal.
     */
    const int frame_rows;

    /**
     * @brief Width of frames that we expect to calculate masks and then width of the frames that we expect during
     * flickering removal.
     */
    const int frame_cols;

    /**
     * @brief Expected value of the timestamp of the next frame to be processed. It is calculated based on camera's fps
     * and last timestamp. It may be a value in milliseconds or -1 (FIRST_TIMESTAMP) to indicate that we accept any
     * timestamp.
     */
    double expected_timestamp;

    /**
     * @brief Tests if 2 values are close enough to each other. It is used to compare values of the same pixel from 2
     * different frames. It uses flickering_threshold.
     * @param a First value to compare.
     * @param b Second value to compare.
     * @return True if values are similar (close enough to each other), false otherwise.
     */
    [[nodiscard]] bool similar(int a, int b) const;

    /**
     * @brief Checks if passed in parameter timestamp is similar to the expected timestamp of the next frame.
     * @param timestamp Timestamp to be checked.
     * @return True if expected timestamp and received timestamp are similar, false otherwise.
     */
    [[nodiscard]] bool timestampIsCloseToExpectedTimestamp(double timestamp) const;

    /**
     * @brief Calculates new value of timestamp of expected next frame.
     * @param timestamp Timestamp of actual frame, based on this value timestamp for the next frame is calculated.
     */
    void calculateNextExpectedTimestamp(double timestamp);

    /**
     * @brief Removes allocated earlier data in corresponding_frames_similarity_levels and
     * adjacent_frames_similarity_levels.
     */
    void clear();

public:
    /**
     * @brief Constructor. Based on fps of the camera calculates number of masks.
     * @param camera_fps Frames per second of the camera from which frames will be used to first calculate masks, and
     * then remove flickering effect using these masks.
     * @param flickering_threshold When we compare 2 values of the same pixel from 2 consecutive frames, this is the
     * threshold that is used to distinguish flickering pixels from not flickering ones.
     * @param max_allowed_flicker_duration Maximum number of consecutive blocks for which given pixel can have the same
     * flickering pattern. If this value is reached, values of the masks for this pixel are changed to remove flickering
     * in next frames. Value of at least 2 is needed. The bigger the number the longer singular pixels will flicker
     * before being removed.
     * @param frame_rows Height of the frames that can be processed by this flickering remover.
     * @param frame_cols Width of the frames that can be processed by this flickering remover.
     */
    FlickerRemoverCPU(unsigned int camera_fps, int flickering_threshold, int max_allowed_flicker_duration,
                      int frame_rows, int frame_cols);

    /**
     * @brief Default destructor.
     */
    virtual ~FlickerRemoverCPU();

    /**
     * @brief Creates and returns pointer to the new frame which is constructed from passed in parameter frame
     * by applying one of the masks calculated earlier. It also refines mask used to remove flickering.
     * @param frame Frame from which copy is made and from this copy flickering is removed. Pointer to the copy is
     * returned.
     * @param timestamp Timestamp of the frame used to control if we remove flickering from consecutive frames.
     * The algorithm of this class uses set of masks that have to be applied in accurate order. If we dropped one or
     * more frames we have to detect such situations and adapt the order of applying masks. We use timestamps to detect
     * frame drops.
     * @param error Returned description of the problem if an error occurs.
     * @return Pointer to the newly allocated frame or nullptr in case of an error.
     */
    Mat *removeFlickering(const Mat &frame, double timestamp, string &error);

    /**
     * @brief Getter for calculated number of elements stored in blocks buffer.
     * @return Maximum number of frames stored in internal structures for processing, masks improvement and flicker
     * removal.
     */
    [[nodiscard]] unsigned int getNumberOfStoredFrames() const;

    /**
     * @brief Resets internal state, so the processing can start over.
     */
    void reset();

    /**
     * @brief Returns a matrix with pixels set to 1 where there was no movement detected and 0 for pixels that were
     * detected as movement.
     * @param mask Returned mask with elements of type unsigned char.
     * @param error Description of the problem if an error occurs.
     * @return True if operation was successful, false otherwise.
     */
    bool getMaskOfStaticPixelsOfLastPairOfFrames(Mat &mask, string &error) const;

    /**
     * @brief Calculates number of frames after which flicker remover starts to remove flickering from frames.
     * @return Number of first frames processed without removing flickering.
     */
    [[nodiscard]] unsigned int getWarmUpDuration() const;

};


#endif //FLICKER_REMOVER_CPU_HPP
