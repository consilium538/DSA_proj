#ifndef MOTION_HPP
#define MOTION_HPP

#include "globals.hpp"
#include "metric.hpp"

using mv_t = std::tuple<int, int, int, int, int, int, double>;
using bma_arg_t = int;

using bma_f = std::function<mv_t( cv::Mat,
                                  cv::Mat,
                                  const cv::Rect,
                                  obj_f,
                                  const obj_arg_t&,
                                  const bma_arg_t& )>;

std::vector<mv_t>
bma( cv::Mat ancher_img,
     cv::Mat tracked_img,
     const int block_size,
     obj_f objective,
     const obj_arg_t& obj_args,
     bma_f matcher,
     const bma_arg_t& bma_args );

mv_t
ebma_f( cv::Mat ancher_img,
        cv::Mat tracked_img,
        const cv::Rect ancher_rect,
        obj_f objective,
        const obj_arg_t& obj_args,
        const bma_arg_t& bma_args );

mv_t
tss_f( cv::Mat ancher_img,
       cv::Mat tracked_img,
       const cv::Rect ancher_rect,
       obj_f objective,
       const obj_arg_t& obj_args,
       const bma_arg_t& bma_args );

mv_t
tdls_f( cv::Mat ancher_img,
        cv::Mat tracked_img,
        const cv::Rect ancher_rect,
        obj_f objective,
        const obj_arg_t& obj_args,
        const bma_arg_t& bma_args );

#endif  // MOTION_HPP
