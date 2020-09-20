#ifndef METRIC_HPP
#define METRIC_HPP

#include "globals.hpp"

using obj_arg_t = double;
using obj_f = std::function<double( const cv::Mat,
                                    const cv::Mat,
                                    const double,
                                    const double,
                                    const obj_arg_t& )>;

double
mad_patch( const cv::Mat ref,
           const cv::Mat comp,
           const double dx,
           const double dy,
           const obj_arg_t& args = obj_arg_t() );

double
mse_patch( const cv::Mat ref,
           const cv::Mat comp,
           const double dx,
           const double dy,
           const obj_arg_t& args = obj_arg_t() );

double
mad_dist( const cv::Mat ref,
          const cv::Mat comp,
          const double dx,
          const double dy,
          const obj_arg_t& args );

#endif  // METRIC_HPP