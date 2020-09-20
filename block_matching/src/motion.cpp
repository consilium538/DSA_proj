#include "motion.hpp"

using namespace std::placeholders;

bool
isInsideRect( cv::Rect img, cv::Rect rect )
{
    return ( rect & img ) == rect;
}

std::optional<cv::Mat>
safecrop( cv::Mat img, cv::Rect rect )
{
    if ( !isInsideRect( cv::Rect( cv::Point( 0, 0 ), img.size() ), rect ) )
        return std::nullopt;
    else
        return std::make_optional( img( rect ) );
}

std::vector<mv_t>
bma( cv::Mat ancher_img,
     cv::Mat tracked_img,
     const int block_size,
     obj_f objective,
     const obj_arg_t& obj_args,
     bma_f matcher,
     const bma_arg_t& bma_args )
{
    const int nrow = ancher_img.rows;
    const int ncol = ancher_img.cols;
    const int nrow_block = 1 + ( ( nrow - 1 ) / block_size );
    const int ncol_block = 1 + ( ( ncol - 1 ) / block_size );

    std::vector<mv_t> motion_vec;

    for ( int i = 0; i < nrow_block; i++ )
    {
        const int i_idx = i * block_size;
        const int i_size =
            i_idx + block_size > nrow ? nrow - i_idx : block_size;

        for ( int j = 0; j < ncol_block; j++ )
        {
            const int j_idx = j * block_size;
            const int j_size =
                j_idx + block_size > ncol ? ncol - j_idx : block_size;

            cv::Rect ancher_rect( cv::Point( j_idx, i_idx ),
                                  cv::Size( j_size, i_size ) );

            mv_t least_pos = matcher( ancher_img, tracked_img, ancher_rect,
                                      objective, obj_args, bma_args );

            motion_vec.push_back( least_pos );
        }
    }

    return motion_vec;
}

mv_t
ebma_f( cv::Mat ancher_img,
        cv::Mat tracked_img,
        const cv::Rect ancher_rect,
        obj_f objective,
        const obj_arg_t& obj_args,
        const bma_arg_t& bma_args )
{
    std::vector<std::tuple<double, int, int>> valid_error;
    cv::Mat ancher_cut = ancher_img( ancher_rect );

    for ( int y = -bma_args; y <= bma_args; y++ )
    {
        for ( int x = -bma_args; x <= bma_args; x++ )
        {
            cv::Rect tracked_rect = ancher_rect + cv::Point( x, y );
            if ( !isInsideRect(
                     cv::Rect( cv::Point( 0, 0 ), ancher_img.size() ),
                     tracked_rect ) )
                continue;
            cv::Mat tracked_cut = tracked_img( tracked_rect );

            valid_error.push_back( std::make_tuple(
                objective( ancher_cut, tracked_cut, y, x, obj_args ),
                y, x ) );
        }
    }

    auto& [error, x_min, y_min] =
        *std::min_element( valid_error.begin(), valid_error.end() );

    return std::make_tuple( ancher_rect.tl().y, ancher_rect.br().y,
                            ancher_rect.tl().x, ancher_rect.br().x, x_min,
                            y_min, (double)error );
}

mv_t
tss_f( cv::Mat ancher_img,
       cv::Mat tracked_img,
       const cv::Rect ancher_rect,
       obj_f objective,
       const obj_arg_t& obj_args,
       const bma_arg_t& bma_args )
{
    std::vector<std::tuple<double, int, int>> valid_error;
    cv::Mat ancher_cut = ancher_img( ancher_rect );
    std::tuple<double, int, int> cur_pos = {
        objective( ancher_cut, tracked_img( ancher_rect ), 0, 0, obj_args ), 0,
        0};
    constexpr std::array<std::tuple<int, int>, 8> xypos{
        std::make_tuple( -1, -1 ), std::make_tuple( -1, 0 ),
        std::make_tuple( -1, 1 ),  std::make_tuple( 0, -1 ),
        std::make_tuple( 0, 1 ),   std::make_tuple( 1, -1 ),
        std::make_tuple( 1, 0 ),   std::make_tuple( 1, +1 )};

    for ( int search_step = bma_args / 2; search_step != 0; search_step /= 2 )
    {
        valid_error.push_back( cur_pos );
        auto& [cur_error, cur_y, cur_x] = cur_pos;
        for ( auto& [x, y] : xypos )
        {
            cv::Rect tracked_rect =
                ancher_rect +
                cv::Point( cur_x + x * search_step, cur_y + y * search_step );
            if ( !isInsideRect(
                     cv::Rect( cv::Point( 0, 0 ), ancher_img.size() ),
                     tracked_rect ) )
                continue;

            valid_error.push_back( std::make_tuple(
                objective( ancher_cut, tracked_img( tracked_rect ),
                           cur_y + y * search_step, cur_x + x * search_step,
                           obj_args ),
                cur_y + y * search_step, cur_x + x * search_step ) );
        }
        cur_pos = *std::min_element( valid_error.begin(), valid_error.end() );
        valid_error.clear();
    }

    auto& [error, x_min, y_min] = cur_pos;

    return std::make_tuple( ancher_rect.tl().y, ancher_rect.br().y,
                            ancher_rect.tl().x, ancher_rect.br().x, x_min,
                            y_min, (double)error );
}

mv_t
tdls_f( cv::Mat ancher_img,
        cv::Mat tracked_img,
        const cv::Rect ancher_rect,
        obj_f objective,
        const obj_arg_t& obj_args,
        const bma_arg_t& bma_args )
{
    std::vector<std::tuple<double, int, int>> valid_error;
    cv::Mat ancher_cut = ancher_img( ancher_rect );
    std::tuple<double, int, int> cur_pos = {
        objective( ancher_cut, tracked_img( ancher_rect ), 0, 0, obj_args ), 0,
        0};
    constexpr std::array<std::tuple<int, int>, 4> xypos{
        std::make_tuple( -1, 0 ), std::make_tuple( 0, -1 ),
        std::make_tuple( 0, 1 ), std::make_tuple( 1, 0 )};
    int search_step = static_cast<int>( std::max(
        2.0, std::pow( 2, std::floor( std::log2( bma_args ) ) - 1 ) ) );

    while ( search_step != 0 )
    {
        valid_error.push_back( cur_pos );
        auto& [cur_error, cur_y, cur_x] = cur_pos;
        for ( auto& [x, y] : xypos )
        {
            cv::Rect tracked_rect =
                ancher_rect +
                cv::Point( cur_x + x * search_step, cur_y + y * search_step );
            if ( !isInsideRect(
                     cv::Rect( cv::Point( 0, 0 ), ancher_img.size() ),
                     tracked_rect ) )
                continue;

            valid_error.push_back( std::make_tuple(
                objective( ancher_cut, tracked_img( tracked_rect ),
                           cur_y + y * search_step, cur_x + x * search_step,
                           obj_args ),
                cur_y + y * search_step, cur_x + x * search_step ) );
        }
        auto next_pos =
            *std::min_element( valid_error.begin(), valid_error.end() );
        if ( next_pos == cur_pos )
            search_step /= 2;
        else
            cur_pos = next_pos;

        valid_error.clear();
    }

    auto& [error, x_min, y_min] = cur_pos;

    return std::make_tuple( ancher_rect.tl().y, ancher_rect.br().y,
                            ancher_rect.tl().x, ancher_rect.br().x, x_min,
                            y_min, (double)error );
}