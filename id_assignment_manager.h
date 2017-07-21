/** 
 *  @file    id_assignment_manager.h
 *  @author  Yahia Farghaly
 *  @date    6/26/2016
 *  @version 1.0 
 *  
 *  @brief Wrappered class to manage the IDs assignment to extracted new and matched features.
 *
 *  @section DESCRIPTION
 *  
 *  This is a small class which manages the IDs assignment of the extracted features. 
 *  If feature is new, it gets a new ID. If it's matched to a previous iteration, 
 *  it get an old ID of a previous feature.
 *  
 *  The class stores Keypoints information in form of a map that has a [key] and a [value]. 
 *  The [key] is designed to express the [index] of the keypoint in the extracted features vector 
 *  and the [value] is expressing the [ID] value of this keypoint.
 *  map<key,value> = map<KP index,ID>
 *  [key] -> [KP's index]
 *  [value] -> [ID]
 *  
 *  From [KP's index] we can know if the assigned KP to the class is a new KP or old KP to a previous map history.
 *  The history is just one past iteration of assiging ID of this class.
 *  
 *  Since [KP's index] by its ownself is not an enough information to make sure of the assigned KP to the class is 
 *  a new or old. We use the matched features vector of 'current extracted KPs' and 'previous extracted KPs' to know
 *  the previous index of a matched KP of current extracted feature hence from the previous index, we can assign 
 *  the current KP index to its old ID of previous KP index. If it's not matched, then it's a new KP hence a new ID
 *  is assigned to.
 */

#pragma once
#include "system_includes.h"

/**
 * The sf namespace associates all Graduation Project codebase.
 * @author Yahia Farghaly
 * @version 1.0
 */
namespace sf {
    /**
     * The vision namespace associates all the tasks required by computer vision
     * @author Yahia Farghaly
     * @version 1.0
     */
    namespace vision {

        /**
         *  @brief Wrappered class to manage the IDs assignment to extracted new and matched features.  
         */
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
    }
}

