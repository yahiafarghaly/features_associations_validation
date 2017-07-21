/** 
 *  @file    id_assignment_manager.cpp
 *  @author  Yahia Farghaly
 *  @date    6/26/2016
 *  @version 1.0 
 *  
 *  @brief Implementation of the wrappered class IDAssignManager to manage the IDs assignment to extracted new and matched features.
 *
 */

#include "id_assignment_manager.h"
#include "../config/vision_config.h"

using namespace sf::vision;

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
    if (__currentID == MAX_ID_NUMBER)
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