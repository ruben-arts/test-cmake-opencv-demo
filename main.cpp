#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <curl/curl.h>

// Callback function to write data from curl to memory
struct MemoryStruct {
    std::vector<unsigned char> memory;
    size_t size;
    
    MemoryStruct() : size(0) {}
};

static size_t WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    struct MemoryStruct *mem = (struct MemoryStruct *)userp;
    
    size_t old_size = mem->memory.size();
    mem->memory.resize(old_size + realsize);
    memcpy(mem->memory.data() + old_size, contents, realsize);
    mem->size += realsize;
    
    return realsize;
}

cv::Mat downloadImage(const std::string& url) {
    CURL *curl_handle;
    CURLcode res;
    struct MemoryStruct chunk;
    
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl_handle = curl_easy_init();
    
    if (curl_handle) {
        curl_easy_setopt(curl_handle, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
        curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, (void *)&chunk);
        curl_easy_setopt(curl_handle, CURLOPT_USERAGENT, "opencv-demo/1.0");
        curl_easy_setopt(curl_handle, CURLOPT_FOLLOWLOCATION, 1L);
        
        res = curl_easy_perform(curl_handle);
        curl_easy_cleanup(curl_handle);
    }
    
    curl_global_cleanup();
    
    if (chunk.size > 0) {
        cv::Mat img = cv::imdecode(chunk.memory, cv::IMREAD_COLOR);
        return img;
    }
    
    return cv::Mat();
}

int main(int argc, char** argv) {
    cv::Mat image;
    std::string input_source;
    
    // Check if we're in CI environment or if display should be disabled
    bool disable_display = false;
    if (getenv("CI") || getenv("GITHUB_ACTIONS") || getenv("DISABLE_DISPLAY")) {
        disable_display = true;
    }
    
    // Parse command line arguments
    std::string image_input;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--no-display" || arg == "--ci") {
            disable_display = true;
        } else if (arg.find("--") != 0) {
            // Not a flag, treat as image input
            image_input = arg;
        }
    }
    
    if (!image_input.empty()) {
        // Check if it's a URL
        if (image_input.find("http://") == 0 || image_input.find("https://") == 0) {
            std::cout << "Downloading image from URL: " << image_input << std::endl;
            image = downloadImage(image_input);
            input_source = "URL: " + image_input;
            
            if (!image.empty()) {
                cv::imwrite("downloaded_image.jpg", image);
                std::cout << "Downloaded image saved as 'downloaded_image.jpg'" << std::endl;
            }
        } else {
            // Try to load as local file
            image = cv::imread(image_input);
            input_source = "File: " + image_input;
        }
    }
    
    // If no image loaded, create synthetic one
    if (image.empty()) {
        std::cout << "No valid image found, creating synthetic test image..." << std::endl;
        
        image = cv::Mat::zeros(400, 600, CV_8UC3);
        cv::rectangle(image, cv::Point(50, 50), cv::Point(200, 150), cv::Scalar(0, 255, 0), -1);
        cv::circle(image, cv::Point(400, 200), 80, cv::Scalar(255, 0, 0), -1);
        cv::ellipse(image, cv::Point(300, 300), cv::Size(100, 50), 45, 0, 360, cv::Scalar(0, 0, 255), -1);
        
        cv::imwrite("synthetic_input.jpg", image);
        input_source = "Synthetic image";
    }

    std::cout << "Processing: " << input_source << std::endl;
    std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;

    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // 1. Edge Detection
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);

    // 2. Contour Detection
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(edges.clone(), contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    cv::Mat contour_image = image.clone();
    cv::drawContours(contour_image, contours, -1, cv::Scalar(0, 255, 255), 2);

    // 3. Color-based segmentation (dominant colors)
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    
    // Create mask for blue objects
    cv::Mat blue_mask;
    cv::inRange(hsv, cv::Scalar(100, 50, 50), cv::Scalar(130, 255, 255), blue_mask);
    
    // Create mask for green objects  
    cv::Mat green_mask;
    cv::inRange(hsv, cv::Scalar(40, 50, 50), cv::Scalar(80, 255, 255), green_mask);
    
    // Combine masks
    cv::Mat color_mask = blue_mask | green_mask;
    cv::Mat color_result;
    image.copyTo(color_result, color_mask);

    // 4. Create a multi-panel result
    cv::Mat top_row, bottom_row, final_result;
    
    // Convert single channel images to 3-channel for concatenation
    cv::Mat edges_color, mask_color;
    cv::cvtColor(edges, edges_color, cv::COLOR_GRAY2BGR);
    cv::cvtColor(color_mask, mask_color, cv::COLOR_GRAY2BGR);
    
    cv::hconcat(image, edges_color, top_row);
    cv::hconcat(contour_image, color_result, bottom_row);
    cv::vconcat(top_row, bottom_row, final_result);

    // Add labels
    cv::putText(final_result, "Original", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    cv::putText(final_result, "Edges", cv::Point(image.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    cv::putText(final_result, "Contours", cv::Point(10, image.rows + 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    cv::putText(final_result, "Color Filter", cv::Point(image.cols + 10, image.rows + 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

    // Save results
    cv::imwrite("opencv_demo_result.jpg", final_result);

    std::cout << "\nOpenCV processing completed!" << std::endl;
    std::cout << "Found " << contours.size() << " contours in the image" << std::endl;
    std::cout << "\nResults saved:" << std::endl;
    std::cout << "  - opencv_demo_result.jpg (4-panel comparison)" << std::endl;

    // Try to display if possible and not disabled
    if (!disable_display) {
        try {
            cv::imshow("OpenCV Demo - Multiple Techniques", final_result);
            std::cout << "\nPress any key to close..." << std::endl;
            cv::waitKey(0);
            cv::destroyAllWindows();
        } catch (const cv::Exception& e) {
            std::cout << "\nDisplay not available, but all images saved successfully!" << std::endl;
        }
    } else {
        std::cout << "\nDisplay disabled (running in CI or --no-display flag used)" << std::endl;
        std::cout << "All images saved successfully!" << std::endl;
    }

    return 0;
}