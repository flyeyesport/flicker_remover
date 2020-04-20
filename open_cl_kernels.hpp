//
// Created by jarek on 24.10.2019.
//

#ifndef OPEN_CL_KERNELS_HPP
#define OPEN_CL_KERNELS_HPP

#include <string>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include <mutex>

using cv::UMat;

/**
 * @brief Helper class containing all OpenCL kernels. It makes it easier to use this logic in the program.
 */
class OpenCLKernels {
protected:
    /**
     * @brief String with OPENCL kernels - functions that are compiled and run on GPU.
     */
    static const char *kernels_src;

    /**
     * @brief String with descriptions of problems when an error occurs. It is set together with <b>opencl_available</b>
     * boolean flag.
     */
    std::string availability_error;

    /**
     * @brief Boolean flag indicating if all kernels were compiled on GPU and are ready to be used. If not, then
     * <b>availability_error</b> is filled with the description of the problem.
     */
    bool opencl_available;

    /**
     * @brief Kernels for which definitions are in <b>kernels_src</b> string. Convention: here we have
     * kernel_<kernel name> names and in kernels_src we have corresponding <kernel name>. See: <b>initOpenCL</b>
     * method for more info of how they are initialized and related to <b>kernels_src</b>.
     */
    cv::ocl::Kernel kernel_update_similarity_levels;
    cv::ocl::Kernel kernel_update_flicker_counter;
    cv::ocl::Kernel kernel_update_masks;
    cv::ocl::Kernel kernel_zero_flicker_counter;
    cv::ocl::Kernel kernel_calculate_filtered_diff;
    cv::ocl::Kernel kernel_calculate_accumulated_diff;

    /**
     * @brief Mutexes guarding access to corresponding kernels which can be accessed from different threads.
     */
    mutable std::mutex kernel_update_similarity_levels_guard;
    mutable std::mutex kernel_update_flicker_counter_guard;
    mutable std::mutex kernel_update_masks_guard;
    mutable std::mutex kernel_zero_flicker_counter_guard;
    mutable std::mutex kernel_calculate_filtered_diff_guard;
    mutable std::mutex kernel_calculate_accumulated_diff_guard;

    /**
     * @brief Initializes OpenCL context, device, program and in the end compiles kernels to be ready to be used and run.
     */
    void initOpenCL();

public:
    /**
     * @brief Constructor. Initializes OpenCL and kernels. After creating the object call <b>isAvailable<b> method to
     * check if kernels can be run.
     */
    OpenCLKernels();

    /**
     * @brief Default destructor.
     */
    ~OpenCLKernels() = default;

    /**
     * @brief Getter for status of availability of OpenCL and possibility to run kernels on GPU.
     * @param error In case of no availabilty the description of the cause and problem is returned in this parameter.
     * @return True if there were no errors and kernels are ready to be run on GPU, false in case of errors.
     */
    [[nodiscard]] bool isAvailable(std::string &error) const;

    /**
     * @brief Used by FlickerRemover to run part of its algorithm on a GPU. It is synchronous (it waits for the
     * processing on GPU to finish).
     * @param src_1
     * @param src_2
     * @param old_levels
     * @param threshold
     * @param new_levels
     * @param dst_levels
     * @param error Returned description of the problem in case of an error.
     * @return True if call was successful, false otherwise.
     */
    bool runKernelUpdateSimilarityLevels(const UMat &src_1, const UMat &src_2, const UMat &old_levels, int threshold,
                                         UMat &new_levels, UMat &dst_levels, std::string &error);

    /**
     * @brief Used by FlickerRemover to run part of its algorithm on a GPU. It is synchronous (it waits for the
     * processing on GPU to finish).
     * @param adjacent_frames_similarity_sum
     * @param similarity_max
     * @param corresponding_frames_similarity_sum
     * @param threshold
     * @param flicker_counter
     * @param error Returned description of the problem in case of an error.
     * @return True if call was successful, false otherwise.
     */
    bool runKernelUpdateFlickerCounter(const UMat &adjacent_frames_similarity_sum, unsigned int similarity_max,
                                       const UMat &corresponding_frames_similarity_sum, float threshold,
                                       UMat &flicker_counter, std::string &error);

    /**
     * @brief Used by FlickerRemover to run part of its algorithm on a GPU. It is synchronous (it waits for the
     * processing on GPU to finish).
     * @param src_1
     * @param src_2
     * @param flicker_counter
     * @param max_duration
     * @param mask
     * @param error Returned description of the problem in case of an error.
     * @return True if call was successful, false otherwise.
     */
    bool runKernelUpdateMasks(const UMat &src_1, const UMat &src_2, const UMat &flicker_counter, int max_duration,
                              UMat &mask, std::string error);

    /**
     * @brief Used by FlickerRemover to run part of its algorithm on a GPU. It is synchronous (it waits for the
     * processing on GPU to finish).
     * @param max_duration
     * @param flicker_counter
     * @param error Returned description of the problem in case of an error.
     * @return True if call was successful, false otherwise.
     */
    bool runKernelZeroFlickerCounter(int max_duration, UMat &flicker_counter, std::string &error);

    /**
     * @brief Used by BlobFinder to run calculating of a filter on a diff image on a GPU. It checks close neighbours of
     * the pixel and also uses 2 thresholds. It is synchronous (it waits for the processing on GPU to finish).
     * Returned pixels get 255 value if and only if number of "white" neighbour pixels is equal or more than
     * threshold_2)/ "White" pixels are those with color equal or more than threshold_1.
     * @param src_diff Source image with 1 channel unsigned char pixels.
     * @param threshold_1 Value of minimum color of the checked pixels to be treated as "white".
     * @param threshold_2 Minimum number of the neighbours of the checked pixel to be treated as "white".
     * @param filtered_diff Returned black and white image.
     * @param error Returned description of the problem in case of an error.
     * @return True if call was successful, false otherwise.
     */
    bool runKernelCalculateFilteredDiff(const UMat &src_diff, unsigned int threshold_1, unsigned int threshold_2,
                                        UMat &filtered_diff, std::string &error);

    /**
     * @param error Returned description of the problem in case of an error.
     * @return True if call was successful, false otherwise.
     */
    bool runKernelCalculateAccumulatedDiff(const UMat &diff_image, const UMat &previous_accumulated_diff_image,
                                           const UMat &mask, const UMat &previous_accumulated_diff_mask,
                                           unsigned char number_of_frames_in_accumulated_diff,
                                           unsigned char accumulation_delta, UMat &accumulated_diff_mask,
                                           UMat &accumulated_diff_image, std::string &error);
};


#endif //OPEN_CL_KERNELS_HPP
