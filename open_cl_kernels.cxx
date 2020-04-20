//
// Created by jarek on 24.10.2019.
//

#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "open_cl_kernels.hpp"
#include <iostream>
#include <chrono>

using namespace std;

const char *OpenCLKernels::kernels_src =
        "unsigned int number_of_white_neighbours(\n"
        "       __global const uchar* image,\n"
        "       int image_step,\n"
        "       int image_offset,\n"
        "       int image_rows,\n"
        "       int image_cols,\n"
        "       int row,\n"
        "       int col,\n"
        "       int radius,\n"
        "       int threshold)\n"
        "{\n"
        "   unsigned int count = 0;\n"
        "   const int min_x = max(col - radius, 0);\n"
        "   const int max_x = min(col + radius, image_cols - 1);\n"
        "   const int min_y = max(row - radius, 0);\n"
        "   const int max_y = min(row + radius, image_rows - 1);\n"
        "   for(int x = min_x; x < col; x++) {\n"
        "       for(int y = min_y; y <= max_y; y++) {\n"
        "           int image_idx = y * image_step + x + image_offset;\n"
        "           if(image[image_idx] > threshold) {\n"
        "                count++;\n"
        "           }\n"
        "       }\n"
        "   }\n"
        "   for(int y = min_y; y < row; y++) {\n"
        "       int image_idx = y * image_step + col + image_offset;\n"
        "       if(image[image_idx] > threshold) {\n"
        "           count++;\n"
        "       }\n"
        "   }\n"
        "   for(int y = row + 1; y <= max_y; y++) {\n"
        "       int image_idx = y * image_step + col + image_offset;\n"
        "       if(image[image_idx] > threshold) {\n"
        "           count++;\n"
        "       }\n"
        "   }\n"
        "   for(int x = col + 1; x <= max_x; x++) {\n"
        "       for(int y = min_y; y <= max_y; y++) {\n"
        "           int image_idx = y * image_step + x + image_offset;\n"
        "           if(image[image_idx] > threshold) {\n"
        "               count++;\n"
        "           }\n"
        "       }\n"
        "   }\n"
        "   return count;\n"
        "}\n"
        "__kernel void update_similarity_levels(\n"
        "       __global const uchar* src_frame_1, int src_frame_1_step, int src_frame_1_offset,\n"
        "       __global const uchar* src_frame_2, int src_frame_2_step, int src_frame_2_offset,\n"
        "       __global uchar* new_levels, int new_levels_step, int new_levels_offset,\n"
        "       __global const uchar* old_levels, int old_levels_step, int old_levels_offset,\n"
        "       __global uchar* dst_levels, int dst_levels_step, int dst_levels_offset,\n"
        "       int similarity_threshold)\n"
        "{\n"
        "   int x = get_global_id(0);\n"
        "   int y = get_global_id(1);\n"
        "   int src_frame_1_idx = y * src_frame_1_step + x + src_frame_1_offset;\n"
        "   int src_frame_2_idx = y * src_frame_2_step + x + src_frame_2_offset;\n"
        "   int new_levels_idx = y * new_levels_step + x + new_levels_offset;\n"
        "   int old_levels_idx = y * old_levels_step + x + old_levels_offset;\n"
        "   int dst_levels_idx = y * dst_levels_step + x + dst_levels_offset;\n"
        "   if(abs_diff(src_frame_1[src_frame_1_idx], src_frame_2[src_frame_2_idx]) <= similarity_threshold) {\n"
        "       new_levels[new_levels_idx] = 1;\n"
        "       dst_levels[dst_levels_idx] += 1;\n"
        "       dst_levels[dst_levels_idx] -= old_levels[old_levels_idx];\n"
        "   } else {\n"
        "       new_levels[new_levels_idx] = 0;\n"
        "       dst_levels[dst_levels_idx] -= old_levels[old_levels_idx];\n"
        "   }\n"
        "}\n"
        "\n"
        "__kernel void update_flicker_counter(\n"
        "       __global const uchar* src, int src_step, int src_offset,\n"
        "       uint src_max,\n"
        "       __global const uchar* src_sim, int src_sim_step, int src_sim_offset,\n"
        "       float threshold,\n"
        "       __global uchar* dst, int dst_step, int dst_offset)\n"
        "{\n"
        "   int x = get_global_id(0);\n"
        "   int y = get_global_id(1);\n"
        "   int src_idx = y * src_step + x + src_offset;\n"
        "   int src_sim_idx = y * src_sim_step + x + src_sim_offset;\n"
        "   int dst_idx = y * dst_step + x + dst_offset;\n"
        "   if(src_sim[src_sim_idx] > threshold && src[src_idx] < src_max) {\n"
        "       dst[dst_idx]++;\n"
        "   } else {\n"
        "       dst[dst_idx] = 0;\n"
        "   }\n"
        "}\n"
        "\n"
        "__kernel void update_masks(\n"
        "       __global const uchar* src_frame_1, int src_frame_1_step, int src_frame_1_offset,\n"
        "       __global const uchar* src_frame_2, int src_frame_2_step, int src_frame_2_offset,\n"
        "       __global const uchar* flicker, int flicker_step, int flicker_offset,\n"
        "       int max_duration,\n"
        "       __global short* dst_mask, int dst_mask_step, int dst_mask_offset)\n"
        "{\n"
        "   int x = get_global_id(0);\n"
        "   int y = get_global_id(1);\n"
        "   int flicker_idx = y * flicker_step + x + flicker_offset;\n"
        "   if(flicker[flicker_idx] > max_duration) {\n"
        "       int src_frame_1_idx = y * src_frame_1_step + x + src_frame_1_offset;\n"
        "       int src_frame_2_idx = y * src_frame_2_step + x + src_frame_2_offset;\n"
        "       int dst_mask_idx = y * dst_mask_step / 2 + x + dst_mask_offset / 2;\n"
        "       dst_mask[dst_mask_idx] += src_frame_2[src_frame_2_idx];\n"
        "       dst_mask[dst_mask_idx] -= src_frame_1[src_frame_1_idx];\n"
        "   }\n"
        "}\n"
        "\n"
        "__kernel void zero_flicker_counter(\n"
        "       int max_duration,\n"
        "       __global uchar* flicker, int flicker_step, int flicker_offset)\n"
        "{\n"
        "   int x = get_global_id(0);\n"
        "   int y = get_global_id(1);\n"
        "   int flicker_idx = y * flicker_step + x + flicker_offset;\n"
        "   if(flicker[flicker_idx] > max_duration) {\n"
        "       flicker[flicker_idx] = 0;\n"
        "   }\n"
        "}\n"
        "\n"
        "__kernel void calculate_filtered_diff(\n"
        "       __global const uchar* src_diff, int src_diff_step, int src_diff_offset, int src_diff_rows, int src_diff_cols,\n"
        "       unsigned int threshold_1,\n"
        "       unsigned int threshold_2,\n"
        "       __global uchar* filtered_diff, int filtered_diff_step, int filtered_diff_offset)\n"
        "{\n"
        "   int x = get_global_id(0);\n"
        "   int y = get_global_id(1);\n"
        "   int src_diff_idx = y * src_diff_step + x + src_diff_offset;\n"
        "   int filtered_diff_idx = y * filtered_diff_step + x + filtered_diff_offset;\n"
        "   if(src_diff[src_diff_idx] > threshold_1 && number_of_white_neighbours(src_diff, src_diff_step, src_diff_offset, src_diff_rows, src_diff_cols, y, x, 1, threshold_1) >= threshold_2) {\n"
        "       filtered_diff[filtered_diff_idx] = 255;\n"
        "   } else {\n"
        "       filtered_diff[filtered_diff_idx] = 0;\n"
        "   }\n"
        "}\n"
        "\n"
        "__kernel void calculate_accumulated_diff(\n"
        "       __global const uchar* diff_image, int diff_image_step, int diff_image_offset, int diff_image_rows, int diff_image_cols,\n"
        "       __global const uchar* previous_accumulated_diff_image, int previous_accumulated_diff_image_step, int previous_accumulated_diff_image_offset,\n"
        "       __global const uchar* mask, int mask_step, int mask_offset,\n"
        "       __global const uchar* previous_accumulated_diff_mask, int previous_accumulated_diff_mask_step, int previous_accumulated_diff_mask_offset,\n"
        "       uchar number_of_frames_in_accumulated_diff,\n"
        "       uchar accumulation_delta,\n"
        "       int radius,\n"
        "       __global uchar* accumulated_diff_mask, int accumulated_diff_mask_step, int accumulated_diff_mask_offset,\n"
        "       __global uchar* accumulated_diff_image, int accumulated_diff_image_step, int accumulated_diff_image_offset)\n"
        "{\n"
        "   int x = get_global_id(0);\n"
        "   int y = get_global_id(1);\n"
        "   int mask_idx = y * mask_step + x + mask_offset;\n"
        "   int previous_accumulated_diff_mask_idx = y * previous_accumulated_diff_mask_step + x + previous_accumulated_diff_mask_offset;\n"
        "   int diff_image_idx = y * diff_image_step + x + diff_image_offset;\n"
        "   int accumulated_diff_image_idx = y * accumulated_diff_image_step + x + accumulated_diff_image_offset;\n"
        "   if((mask[mask_idx] > 0 || previous_accumulated_diff_mask[previous_accumulated_diff_mask_idx] > 0) && diff_image[diff_image_idx] > 0) {\n"
        "       accumulated_diff_image[accumulated_diff_image_idx] = number_of_frames_in_accumulated_diff + accumulation_delta;\n"
        "       int row_min, row_max, col_min, col_max;\n"
        "       if(x - radius < 0) {\n"
        "           col_min = 0;\n"
        "       } else {\n"
        "           col_min = x - radius;\n"
        "       }\n"
        "       if(x + radius >= diff_image_cols) {\n"
        "           col_max = diff_image_cols - 1;\n"
        "       } else {\n"
        "           col_max = x + radius;\n"
        "       }\n"
        "       if(y - radius < 0) {\n"
        "           row_min = 0;\n"
        "       } else {\n"
        "           row_min = y - radius;\n"
        "       }\n"
        "       if(y + radius >= diff_image_rows) {\n"
        "           row_max = diff_image_rows - 1;\n"
        "       } else {\n"
        "           row_max = y + radius;\n"
        "       }\n"
        "       for(int r = row_min; r <= row_max; r++) {\n"
        "           for(int c = col_min; c <= col_max; c++) {\n"
        "               int idx = r * accumulated_diff_mask_step + c + accumulated_diff_mask_offset;\n"
        "               accumulated_diff_mask[idx] = 255;\n"
        "           }\n"
        "       }\n"
        "   } else {\n"
        "       int previous_accumulated_diff_image_idx = y * previous_accumulated_diff_image_step + x + previous_accumulated_diff_image_offset;\n"
        "       uchar previous_accumulated_value = previous_accumulated_diff_image[previous_accumulated_diff_image_idx];\n"
        "       if(previous_accumulated_value > accumulation_delta) {\n"
        "           accumulated_diff_image[accumulated_diff_image_idx] = previous_accumulated_value - (uchar) 1;\n"
        "       } else {\n"
        "           accumulated_diff_image[accumulated_diff_image_idx] = 0;\n"
        "       }\n"
        "   }\n"
        "}\n";

OpenCLKernels::OpenCLKernels()
        : opencl_available(false)
{
    initOpenCL();
}

bool OpenCLKernels::isAvailable(std::string &error) const
{
    if(opencl_available) {
        return true;
    } else {
        error = availability_error;
        return false;
    }
}

void OpenCLKernels::initOpenCL()
{
    if(!cv::ocl::haveOpenCL()) {
        availability_error = "No OpenCL available.";
        opencl_available = false;
        return;
    }

//    const cv::ocl::Context &ctx = cv::ocl::Context::getDefault();
//    if(!ctx.ptr()) {
//        availability_error = "Could not get default context for OpenCL.";
//        opencl_available = false;
//        return;
//    }
//
//    const cv::ocl::Device &device = cv::ocl::Device::getDefault();
//    if(!device.compilerAvailable()) {
//        availability_error = "Could not get compiler for default device for OpenCL.";
//        opencl_available = false;
//        return;
//    }

    cv::ocl::Context context;
    if (!context.create(cv::ocl::Device::TYPE_GPU)) {
        availability_error = "Could not get default context for OpenCL.";
        opencl_available = false;
        return;
    }

    if(context.ndevices() < 1) {
        availability_error = "No GPU devices available for OpenCL.";
        opencl_available = false;
        return;
    }

    const cv::ocl::Device &device = context.device(0);
    if(!device.compilerAvailable()) {
        availability_error = "Could not get compiler for default device for OpenCL.";
        opencl_available = false;
        return;
    }

    //make sure that we have different module names for different objects of this class.
    ostringstream address;
    address << (void const *)this;
    string module_name = address.str();

    cv::ocl::ProgramSource source(module_name, "simple", kernels_src, "");

    cv::String errmsg;
    cv::ocl::Program program = context.getProg(source, "", errmsg);
    if(program.ptr() == NULL) {
        availability_error = "Could not compile program: " + errmsg;
        opencl_available = false;
        return;
    }

    if(!errmsg.empty()) {
        std::cout << "OpenCL program build log:" << std::endl << errmsg << std::endl;
    }

    kernel_update_similarity_levels = cv::ocl::Kernel("update_similarity_levels", program);
    if(kernel_update_similarity_levels.empty()) {
        availability_error = "Could not get kernel: update_similarity_levels.";
        opencl_available = false;
        return;
    }

    kernel_update_flicker_counter = cv::ocl::Kernel("update_flicker_counter", program);
    if(kernel_update_flicker_counter.empty()) {
        availability_error = "Could not get kernel: update_flicker_counter.";
        opencl_available = false;
        return;
    }

    kernel_update_masks = cv::ocl::Kernel("update_masks", program);
    if(kernel_update_masks.empty()) {
        availability_error = "Could not get kernel: update_masks.";
        opencl_available = false;
        return;
    }

    kernel_zero_flicker_counter = cv::ocl::Kernel("zero_flicker_counter", program);
    if(kernel_zero_flicker_counter.empty()) {
        availability_error = "Could not get kernel: zero_flicker_counter.";
        opencl_available = false;
        return;
    }

    kernel_calculate_filtered_diff = cv::ocl::Kernel("calculate_filtered_diff", program);
    if(kernel_calculate_filtered_diff.empty()) {
        availability_error = "Could not get kernel: calculate_filtered_diff.";
        opencl_available = false;
        return;
    }

    kernel_calculate_accumulated_diff = cv::ocl::Kernel("calculate_accumulated_diff", program);
    if(kernel_calculate_accumulated_diff.empty()) {
        availability_error = "Could not get kernel: calculate_accumulated_diff.";
        opencl_available = false;
        return;
    }

    opencl_available = true;
}


bool OpenCLKernels::runKernelUpdateSimilarityLevels(const UMat &src_1, const UMat &src_2, const UMat &old_levels,
                                                    int similarity_threshold, UMat &new_levels, UMat &dst_levels,
                                                    std::string &error)
{
    if(!isAvailable(error)) {
        return false;
    }

    size_t global_size[2] = {(size_t) src_1.cols, (size_t) src_1.rows};
    size_t local_size[2] = {16, 16};
    bool execution_result;
    {
        scoped_lock<mutex> lock(kernel_update_similarity_levels_guard);
        execution_result = kernel_update_similarity_levels.args(
                        cv::ocl::KernelArg::ReadOnlyNoSize(src_1),
                        cv::ocl::KernelArg::ReadOnlyNoSize(src_2),
                        cv::ocl::KernelArg::ReadWriteNoSize(new_levels),
                        cv::ocl::KernelArg::ReadOnlyNoSize(old_levels),
                        cv::ocl::KernelArg::WriteOnlyNoSize(dst_levels),
                        similarity_threshold
                ).run(2, global_size, local_size, true);
    }
    if(!execution_result) {
        error = "OpenCL kernel: kernel_update_similarity_levels launch failed.";
        return false;
    }

    return true;
}

bool OpenCLKernels::runKernelUpdateFlickerCounter(const UMat &adjacent_frames_similarity_sum,
                                                  unsigned int similarity_max,
                                                  const UMat &corresponding_frames_similarity_sum, float threshold,
                                                  UMat &flicker_counter, std::string &error)
{
    if(!isAvailable(error)) {
        return false;
    }

    size_t global_size[2] = {(size_t) adjacent_frames_similarity_sum.cols,
                             (size_t) adjacent_frames_similarity_sum.rows};
    size_t local_size[2] = {16, 16};
    bool execution_result;
    {
        scoped_lock<mutex> lock(kernel_update_flicker_counter_guard);
        execution_result = kernel_update_flicker_counter.args(
                        cv::ocl::KernelArg::ReadOnlyNoSize(adjacent_frames_similarity_sum),
                        similarity_max,
                        cv::ocl::KernelArg::ReadOnlyNoSize(corresponding_frames_similarity_sum),
                        threshold,
                        cv::ocl::KernelArg::ReadWriteNoSize(flicker_counter)
                ).run(2, global_size, local_size, true);
    }
    if(!execution_result) {
        error = "OpenCL kernel: kernel_update_flicker_counter launch failed.";
        return false;
    }

    return true;
}

bool OpenCLKernels::runKernelUpdateMasks(const UMat &src_1, const UMat &src_2, const UMat &flicker_counter,
                                         int max_duration, UMat &mask, std::string error)
{
    if(!isAvailable(error)) {
        return false;
    }

    size_t global_size[2] = {(size_t) src_1.cols, (size_t) src_1.rows};
    size_t local_size[2] = {16, 16};
    bool execution_result;
    {
        scoped_lock<mutex> lock(kernel_update_masks_guard);
        execution_result = kernel_update_masks.args(
                cv::ocl::KernelArg::ReadOnlyNoSize(src_1),
                cv::ocl::KernelArg::ReadOnlyNoSize(src_2),
                cv::ocl::KernelArg::ReadWriteNoSize(flicker_counter),
                max_duration,
                cv::ocl::KernelArg::ReadWriteNoSize(mask)
        ).run(2, global_size, local_size, true);
    }
    if(!execution_result) {
        error = "OpenCL kernel: kernel_update_masks launch failed.";
        return false;
    }

    return true;
}

bool OpenCLKernels::runKernelZeroFlickerCounter(int max_duration, UMat &flicker_counter, std::string &error)
{
    if(!isAvailable(error)) {
        return false;
    }

    size_t global_size[2] = {(size_t) flicker_counter.cols, (size_t) flicker_counter.rows};
    size_t local_size[2] = {16, 16};
    bool execution_result;
    {
        scoped_lock<mutex> lock(kernel_zero_flicker_counter_guard);
        execution_result = kernel_zero_flicker_counter.args(
                        max_duration,
                        cv::ocl::KernelArg::ReadWriteNoSize(flicker_counter)
                ).run(2, global_size, local_size, true);
    }
    if(!execution_result) {
        error = "OpenCL kernel: kernel_zero_flicker_counter launch failed.";
        return false;
    }

    return true;
}

bool
OpenCLKernels::runKernelCalculateFilteredDiff(const UMat &src_diff, unsigned int threshold_1, unsigned int threshold_2,
                                              UMat &filtered_diff, std::string &error)
{
    if(!isAvailable(error)) {
        return false;
    }

    size_t global_size[2] = {(size_t) src_diff.cols, (size_t) src_diff.rows};
    size_t local_size[2] = {16, 16};
    bool execution_result;
    {
        scoped_lock<mutex> lock(kernel_calculate_filtered_diff_guard);
        execution_result = kernel_calculate_filtered_diff.args(
                cv::ocl::KernelArg::ReadOnly(src_diff),
                threshold_1,
                threshold_2,
                cv::ocl::KernelArg::WriteOnlyNoSize(filtered_diff)
        ).run(2, global_size, local_size, true);
    }
    if(!execution_result) {
        error = "OpenCL kernel: kernel_calculate_filtered_diff launch failed.";
        return false;
    }

    return true;
}

bool
OpenCLKernels::runKernelCalculateAccumulatedDiff(const UMat &diff_image, const UMat &previous_accumulated_diff_image,
                                                 const UMat &mask, const UMat &previous_accumulated_diff_mask,
                                                 unsigned char number_of_frames_in_accumulated_diff,
                                                 unsigned char accumulation_delta, UMat &accumulated_diff_mask,
                                                 UMat &accumulated_diff_image, std::string &error)
{
    if(!isAvailable(error)) {
        return false;
    }

    size_t global_size[2] = {(size_t) diff_image.cols, (size_t) diff_image.rows};
    size_t local_size[2] = {16, 16};
    bool execution_result;
    {
        scoped_lock<mutex> lock(kernel_calculate_accumulated_diff_guard);
        execution_result = kernel_calculate_accumulated_diff.args(
                cv::ocl::KernelArg::ReadOnly(diff_image),
                cv::ocl::KernelArg::ReadOnlyNoSize(previous_accumulated_diff_image),
                cv::ocl::KernelArg::ReadOnlyNoSize(mask),
                cv::ocl::KernelArg::ReadOnlyNoSize(previous_accumulated_diff_mask),
                number_of_frames_in_accumulated_diff,
                accumulation_delta,
                6,
                cv::ocl::KernelArg::ReadWriteNoSize(accumulated_diff_mask),
                cv::ocl::KernelArg::WriteOnlyNoSize(accumulated_diff_image)
        ).run(2, global_size, local_size, true);
    }
    if(!execution_result) {
        error = "OpenCL kernel: kernel_calculate_accumulated_diff launch failed.";
        return false;
    }

    return true;
}
