#include "main.hpp"

double
psnr( cv::Mat orig, cv::Mat noised )
{
    cv::Mat square_error;
    cv::multiply( ( orig - noised ), ( orig - noised ), square_error );
    return 10.0 *
           std::log10( 255.0 * 255.0 / (double)cv::mean( square_error )[0] );
}

int
main( int argv, char** argc )
{
    using namespace std::placeholders;
    namespace fs = std::filesystem;

    ////////////////////////////////////

    auto config_file = fs::path( "./config.json" );
    auto inputPath = fs::path( "./srcImg/" );
    auto savepath = fs::path( "./dstImg/" );
    auto logpath = fs::path( "./log" );

    if ( !fs::exists( savepath ) )
        fs::create_directory( savepath );
    if ( !fs::exists( logpath ) )
        fs::create_directory( logpath );

    std::vector<std::tuple<cv::Mat, std::string>> ImgArr;

    std::vector test_set = {
        // std::make_tuple( "cubecut", 8, "ebma", 7, "mad_patch", 0.0 ),
        std::make_tuple( "cubecut", 8, "ebma", 7, "mad_patch", 0.0 ),
        std::make_tuple( "cubecut", 8, "tss", 7, "mad_patch", 0.0 ),
        std::make_tuple( "street", 8, "ebma", 7, "mad_patch", 0.0 ),
        std::make_tuple( "street", 8, "tss", 7, "mad_patch", 0.0 ),
        // std::make_tuple( "cubecut", 8, "ebma", 15, "mad_dist", 0.1 ),
        // std::make_tuple( "street", 8, "ebma", 15, "mad_dist", 0.1 ),
    };

    for ( auto& [img_str, block_size, bma_str, bma_args, obj_str, obj_args] :
          test_set )
    {
        std::string bma_arg_str = fmt::format( "{}", bma_args );
        std::string obj_arg_str = fmt::format( "{}", obj_args );

        std::string option_str =
            fmt::format( "{}({})_{}({})_{}({})", img_str, block_size, bma_str,
                         bma_arg_str, obj_str, obj_arg_str );

        auto ancher_path = inputPath / fmt::format( "{}1.tif", img_str );
        auto ancher_img =
            cv::imread( ancher_path.string(), cv::IMREAD_GRAYSCALE );
        if ( ancher_img.empty() )
        {
            std::cout << "image load failed!" << std::endl;
            return -1;
        }
        ImgArr.push_back( std::make_tuple(
            ancher_img, fmt::format( "{}_{}", option_str, "ancher" ) ) );

        auto tracked_path = inputPath / fmt::format( "{}2.tif", img_str );
        auto tracked_img =
            cv::imread( tracked_path.string(), cv::IMREAD_GRAYSCALE );
        if ( tracked_img.empty() )
        {
            std::cout << "image load failed!" << std::endl;
            return -1;
        }
        ImgArr.push_back( std::make_tuple(
            tracked_img, fmt::format( "{}_{}", option_str, "tracked" ) ) );
        
        std::cout << "loaded" << std::endl;

        const int nrow_ref = ancher_img.rows;
        const int ncol_ref = ancher_img.cols;
        const int nrow_anch = tracked_img.rows;
        const int ncol_anch = tracked_img.cols;

        if ( nrow_anch != nrow_ref || ncol_anch != ncol_ref )
        {
            std::cout << "image size is not same!" << std::endl;
            return -1;
        }

        const int nrow_block = 1 + ( ( nrow_ref - 1 ) / block_size );
        const int ncol_block = 1 + ( ( ncol_ref - 1 ) / block_size );

        std::ofstream mv_out;
        mv_out.open( ( logpath / fmt::format( "{}.csv", option_str ) ).string(),
                     std::ios::trunc );
        mv_out << "xpos, ypos, xvec, yvec, cost\n";

        std::vector<mv_t> motion_vec;

        auto starttime = std::chrono::high_resolution_clock::now();
        motion_vec = bma( ancher_img, tracked_img, block_size, obj_map[obj_str],
                          obj_args, bma_map[bma_str], bma_args );
        auto endtime = std::chrono::high_resolution_clock::now();
        double time_taken_ms =
            (double)( std::chrono::duration_cast<std::chrono::nanoseconds>(
                            endtime - starttime )
                            .count() ) *
            1e-6;

        for ( auto& [tlx, brx, tly, bry, vec_x, vex_y, error] : motion_vec )
        {
            mv_out << fmt::format( "{0:f}, {1:f}, {2:f}, {3:f}, {4:f}\n",
                                   ( tly + (double)block_size / 2 ),
                                   ( tlx + (double)block_size / 2 ),
                                   (double)vex_y, (double)vec_x,
                                   (double)error );
        }

        cv::Mat reconst_img = cv::Mat_<uchar>( tracked_img.size(), 0 );

        for ( auto& [tlx, brx, tly, bry, vec_x, vex_y, error] : motion_vec )
        {
            cv::Mat reconst_Patch =
                tracked_img( cv::Range( tlx + vec_x, brx + vec_x ),
                             cv::Range( tly + vex_y, bry + vex_y ) );
            reconst_Patch.copyTo(
                reconst_img( cv::Range( tlx, brx ), cv::Range( tly, bry ) ) );
        }
        ImgArr.push_back( std::make_tuple(
            reconst_img, fmt::format( "{}_{}", option_str, "reconst" ) ) );

        // do psnr calculation

        mv_out << std::endl;
        std::cout
            << fmt::format(
                   "end of {} computation!\npsnr : {:f} processing time : {:f}\t",
                   option_str, psnr( ancher_img, reconst_img ),
                   time_taken_ms)
            << std::endl;
    }

    img_save( ImgArr, savepath.string(), ".png",
              {cv::IMWRITE_PNG_COMPRESSION, 9} );

    ////////////////////////////////////

    return 0;
}
