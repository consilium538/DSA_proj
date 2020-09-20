#include "metric.hpp"

double
mad_patch( const cv::Mat ref,
           const cv::Mat comp,
           const double dx,
           const double dy,
           const obj_arg_t& args )
{
    return cv::sum( cv::abs( ref - comp ) )[0];
}

double
mse_patch( const cv::Mat ref,
           const cv::Mat comp,
           const double dx,
           const double dy,
           const obj_arg_t& args )
{
    return cv::sum( ( ref - comp ) * ( ref - comp ) )[0];
}

double
mad_dist( const cv::Mat ref,
          const cv::Mat comp,
          const double dx,
          const double dy,
          const obj_arg_t& args )
{
    return cv::sum( cv::abs( ref - comp ) )[0] + ( dx * dx + dy * dy ) * args;
}