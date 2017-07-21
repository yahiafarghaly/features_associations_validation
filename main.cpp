#include <cstdlib>
#include "system_includes.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "timer.h"

using namespace std;
using namespace cv;

#define CROSS_CHECK 0
#define FLANN_MATCHING 0
#define CROSS_CHECK_PLUS_THRESHOLD 1

void matchTwoFrames_cpu(cv::Mat& in_cpu_descriptor_1,
        cv::Mat& in_cpu_descriptor_2,
        std::vector<cv::DMatch> &out_good_matches) {

    sf::Timer matching_time;
    matching_time.start();

#if(CROSS_CHECK)
    std::vector<cv::DMatch> matches12, matches21;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(in_cpu_descriptor_1, in_cpu_descriptor_2, matches12);
    matcher.match(in_cpu_descriptor_2, in_cpu_descriptor_1, matches21);
    cv::DMatch forward;
    cv::DMatch backward;
    for (size_t i = 0; i < matches12.size(); i++) {
        forward = matches12[i];
        backward = matches21[forward.trainIdx];

        if (backward.trainIdx == forward.queryIdx)
            out_good_matches.push_back(forward);
    }
#elif(FLANN_MATCHING)
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match(in_cpu_descriptor_1, in_cpu_descriptor_2, matches);
    double max_dist = 0;
    double min_dist = 100;
    //-- Quick calculation of max and min distances between keypoints
    for (int i = 0; i < in_cpu_descriptor_1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
    //-- small)
    //-- PS.- radiusMatch can also be used here.
    for (int i = 0; i < in_cpu_descriptor_1.rows; i++) {
        if (matches[i].distance <= 0.02 /*max(2 * min_dist, 0.02)*/) { //(0.02 gives better result)
            out_good_matches.push_back(matches[i]);
        }
    }
#elif(CROSS_CHECK_PLUS_THRESHOLD)
    std::vector<cv::DMatch> matches12, matches21;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(in_cpu_descriptor_1, in_cpu_descriptor_2, matches12);
    matcher.match(in_cpu_descriptor_2, in_cpu_descriptor_1, matches21);
    cv::DMatch forward;
    cv::DMatch backward;
    for (size_t i = 0; i < matches12.size(); i++) {
        forward = matches12[i];
        backward = matches21[forward.trainIdx];

        if ((backward.trainIdx == forward.queryIdx) && forward.distance < 10) { // cross check + threshold (better)
            //       std::cout << "F distance: " << forward.distance << " B distance: " << backward.distance << "\n";
            out_good_matches.push_back(forward);
        }
    }
#else
    // Ratio Test
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector< std::vector< cv::DMatch> > k_matches;
    matcher.knnMatch(in_cpu_descriptor_1, in_cpu_descriptor_2,
            k_matches, 2);
    //best matches
    for (int k = 0; k < k_matches.size(); k++) {
        if ((k_matches[k].size() <= 2 && k_matches[k].size() > 0) &&
                (k_matches[k][0].distance < 0.6 * (k_matches[k][1].distance))) {
            out_good_matches.push_back(k_matches[k][0]);
        }
    }
#endif
    matching_time.end();
    std::cout << "Matching time: " << matching_time.getTime() << " " << matching_time.getTimeUnitString() << "\n";
}

class IDAssignManager {
public:

    // Default constructor  
    IDAssignManager();
    // Parametrized Constructor
    IDAssignManager(unsigned int _id);
    // Assignment operator
    void operator=(const IDAssignManager &D);
    // assign a new ID to kp_index
    void assignAvailableID(unsigned int *kp_index);
    // forced assigned ID to kp_index
    void assignCertainID(unsigned int in_kp_index, unsigned int in_id);
    // retrieve last assigned ID
    unsigned int getCurrentID();
    // retrieve ID of a giving KP's index
    bool getID(unsigned int *in_kp_index, unsigned int *out_id);
    // print stored KP indices alongside its IDs
    void printMap();
    // check if a previous KP's index is available when it's matched to a current KP's index
    bool is_KP_IndexMatchPreviousKP(unsigned int *in_index,
            std::vector<cv::DMatch> *in_matched_features,
            unsigned int *out_trainIdx) const;
private:
    unsigned int __currentID; ///< holds the last assigned ID
    std::map<unsigned int, unsigned int> ID_List; ///< map<key,value> = map<KP index,ID>
};

IDAssignManager __idManager;

void fillkeyPointsList(bool isFirstTime,
        std::vector<cv::KeyPoint>& __cpu_current_keypoints,
        std::vector<cv::DMatch>&__good_matches) {

    unsigned int old_id;
    unsigned int trainIdx;
    IDAssignManager tmp_manager;
    for (size_t current_KP_index = 0; current_KP_index < __cpu_current_keypoints.size(); current_KP_index++) {

        if (isFirstTime) {
            __idManager.assignAvailableID((unsigned int*) &current_KP_index);
            __idManager.getID((unsigned int*) &current_KP_index, &old_id);
            std::cout << "kp[" << current_KP_index << "]: " << " P-id[" << -1 << "]" << "-->" << " N-id[" << old_id << "]\n";
        } else {
            if (__idManager.is_KP_IndexMatchPreviousKP((unsigned int*) &current_KP_index, &__good_matches, (unsigned int*) &trainIdx)) {
                //here we index the previous ID
                if (__idManager.getID(&trainIdx, &old_id)) {
                    tmp_manager.assignCertainID(current_KP_index, old_id);
                } else {
                    tmp_manager.assignAvailableID((unsigned int*) &current_KP_index);
                }
            } else {
                tmp_manager.assignAvailableID((unsigned int*) &current_KP_index);
            }
            unsigned int new_id;
            tmp_manager.getID((unsigned int*) &current_KP_index, &new_id);
            std::cout << "kp[" << current_KP_index << "]: " << " P-id[" << old_id << "]" << "-->" << " N-id[" << new_id << "]\n";
        }
    }
    if (!isFirstTime) {

        __idManager = tmp_manager;
    }
}

int main(int argc, char** argv) {

    Mat image;
    Mat image_previous;
    Mat out_image;
    char image_name[256] = "output_result";
    //cv::namedWindow(image_name, 0);
    // cv::resizeWindow(image_name, 600, 600);
    bool first = true;
    // current
    std::vector<cv::KeyPoint> current_keypoints;
    cv::Mat current_descriptors;
    //previous
    std::vector<cv::KeyPoint> previous_keypoints;
    cv::Mat previous_descriptors;
    // init extractor
#if(!FLANN_MATCHING)
    cv::ORB detector(10);
#else
    SurfFeatureDetector detector(400);
#endif
    std::vector<cv::DMatch> good_matches;
    char key = 0;
    string fileName;
    int num = 1;
    while (key != 'q') {

        fileName = "./i_seq/s_" + std::to_string(num) + ".jpg";

        image = imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);

        detector.detect(image, current_keypoints);
        detector.compute(image, current_keypoints, current_descriptors);

        // matching
        if (!first) {
            matchTwoFrames_cpu(current_descriptors, previous_descriptors, good_matches);
            fillkeyPointsList(first, current_keypoints, good_matches);
            std::cout << "GMKP: " << good_matches.size() << "\n";
            drawMatches(image, current_keypoints, image_previous, previous_keypoints, good_matches, out_image,
                    cv::Scalar(0, 255, 0));
            cv::imshow(image_name, out_image);
            key = cv::waitKey();
            good_matches.clear();
            current_keypoints.clear();
            previous_keypoints.clear();

        } else {
            fillkeyPointsList(first, current_keypoints, good_matches);
            first = false;
        }
        image.copyTo(image_previous);
        detector.detect(image_previous, previous_keypoints);
        detector.compute(image_previous, previous_keypoints, previous_descriptors);


        num += 1;
        if (num > 2) num = 1;
    }



    return 0;
}

/** 
 *   @brief  Initialize __currentID to zero
 */
IDAssignManager::IDAssignManager() {
    __currentID = 0;
};

/** 
 *   @brief  Initialize __currentID to _id   
 *  
 *   @param  _id is an initialized unsigned integer variable 
 */
IDAssignManager::IDAssignManager(unsigned int _id) : __currentID(_id) {
}

/** 
 *   @brief  Overloaded assignment operator   
 *  
 *   @param  D is an initialized IDAssignManager variable 
 *   @return void
 */
void IDAssignManager::operator=(const IDAssignManager &D) {
    this->__currentID = D.__currentID;
    this->ID_List.clear();
    this->ID_List.insert(D.ID_List.begin(), D.ID_List.end());
}

/** 
 *   @brief  Store KP's index with a new ID in ID_List Map  
 *  
 *   @param  kp_index is an initialized unsigned integer variable 
 *   @return void
 */
void IDAssignManager::assignAvailableID(unsigned int *kp_index) {
    if (__currentID == 100000)
        __currentID = 0;
    ID_List[*kp_index] = __currentID;
    ++__currentID;
}

/** 
 *   @brief  retrieve last assigned ID 
 *  
 *   @return unsigned int of __currentID
 */
unsigned int IDAssignManager::getCurrentID() {
    return __currentID;
}

/** 
 *   @brief  forced assigned ID to KP's index in ID_List Map.   
 *  
 *   @param  in_kp_index is an initialized unsigned integer variable 
 *   @param  in_id is an unsigned initialized integer variable 
 *   @return void
 */
void IDAssignManager::assignCertainID(unsigned int in_kp_index, unsigned int in_id) {
    ID_List[in_kp_index] = in_id;
}

/** 
 *   @brief  retrieve ID of a giving KP's index   
 *  
 *   @param  in_kp_index is an initialized unsigned integer variable
 *   @return out_id is an initialized unsigned integer variable with ID associated with in_kp_index
 *   @return boolen (true => if it's existed in the ID_List ,false => if not)
 *   @note   in case of false, you should assign a new id to this KP via assignCertainID()
 *   @note   in case of false, out_id is left uninitialized
 */
bool IDAssignManager::getID(unsigned int *in_kp_index, unsigned int *out_id) {
    auto it = ID_List.find(*in_kp_index);
    if (it == std::end(ID_List)) { // true if index is not in the list 
        return false;
    } else
        *out_id = it->second;
    return true;
}

/*! \brief  check if a certain current KP exist in the previous KPs
 * via trainIdx 
 */

/** 
 *   @brief  Check if a previous KP's index is available when it's matched to a current KP's index.   
 *  
 *   @param  current_kp_index is an initialized unsigned integer variable 
 *   @param  matched_features is an initialized std::vector<cv::DMatch> vector
 *   @return  out_trainIdx is an initialized integer variable with previous KP's index associated to current KP's index. 
 *   @return boolen (true, if Previous KP's index exists and associated with current KP's index)
 */
bool IDAssignManager::is_KP_IndexMatchPreviousKP(unsigned int *current_kp_index,
        std::vector<cv::DMatch> *matched_features,
        unsigned int *out_trainIdx) const {

    //Binary search as Dmatch vector is sorted by default and it's verified against linear search
    size_t mid, left = 0;
    size_t right = matched_features->size();
    while (left < right) {
        mid = left + (right - left) / 2;
        if (*current_kp_index > (*matched_features)[mid].queryIdx) {
            left = mid + 1;
        } else if (*current_kp_index < (*matched_features)[mid].queryIdx) {
            right = mid;
        } else { // equal
            *out_trainIdx = (*matched_features)[mid].trainIdx;

            return true;
        }
    }
    return false;
    /*
    for (auto kp : *matched_features)
        if (*current_kp_index == kp.queryIdx) {
     *out_trainIdx = kp.trainIdx;
            return true;
        }
    return false;
     */
}

/** 
 *   @brief  print stored KP indices alongside its IDs.
 *   @return void
 */
void IDAssignManager::printMap() {
    std::cout << "[kp]=id\n";
    for (auto id_i : ID_List) {
        std::cout << "[" << id_i.first << "]=" << id_i.second << "\n";
    }
}