
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <cstring>
#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include <map>
#include <string>
#include <vector>
#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/db.hpp"
#include <stdint.h>
#include <string>
#include <utility>
#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

using namespace cv;
using namespace std;
using namespace caffe;
using namespace caffe;  // NOLINT(build/namespaces)

using std::max;
using std::pair;
using boost::scoped_ptr;

DEFINE_string(backend, "lmdb",
        "The backend {leveldb, lmdb} containing the images");
int np = 18;
int np_in_lmdb = 17; //fake it
struct Joints {
    vector<Point2f> joints;
    vector<int> isVisible;
};

struct Bbox {
    float left;
    float top;
    float width;
    float height;
};

struct MetaData {
    string type; //"cpm" or "detect"
    string file_name;
    string dataset;
    Size img_size;
    bool isValidation;
    int numOtherPeople;
    int people_index;
    int annolist_index;
    int write_number;
    int total_write_number;
    int epoch;
    Point2f objpos; //objpos_x(float), objpos_y (float)
    float scale_self;
    Joints joint_self; //(3*16)
    bool has_teeth_mask;
    int image_id;
    int num_keypoints_self;
    float segmentation_area;
    Bbox bbox;

    vector<Point2f> objpos_other; //length is numOtherPeople
    vector<float> scale_other; //length is numOtherPeople
    vector<Joints> joint_others; //length is numOtherPeople
    vector<Bbox> bboxes_other;
    vector<int> num_keypoints_others;
    vector<float> segmentation_area_others;

    // uniquely used by coco data (detect)
    int numAnns;
    int image_id_in_coco;
    int image_id_in_training;
    vector<int> num_keypoints;
    vector<bool> iscrowd;
    vector<Bbox> bboxes;
    vector<Joints> annotations;
};


void TransformJoints(Joints& j) {
    Joints jo = j;
    int COCO_to_ours_1[18] = {1,6, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
    int COCO_to_ours_2[18] = {1,7, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
    jo.joints.resize(18);
    jo.isVisible.resize(18);

    for(int i = 0; i < 18; i++){
        jo.joints[i] = (j.joints[COCO_to_ours_1[i]-1] + j.joints[COCO_to_ours_2[i]-1]) * 0.5;
        if(j.isVisible[COCO_to_ours_1[i]-1]==2 || j.isVisible[COCO_to_ours_2[i]-1]==2){
            jo.isVisible[i] = 2;
            jo.joints[i] = Point2f(0, 0); 
        }
        else if(j.isVisible[COCO_to_ours_1[i]-1]==3 || j.isVisible[COCO_to_ours_2[i]-1]==3){
            jo.isVisible[i] = 3;
        }
        else {
            jo.isVisible[i] = j.isVisible[COCO_to_ours_1[i]-1] && j.isVisible[COCO_to_ours_2[i]-1];
        }
    }

    j = jo;
}

void TransformMetaJoints(MetaData& meta) {
    //transform joints in meta from np_in_lmdb (specified in prototxt) to np (specified in prototxt)
    TransformJoints(meta.joint_self);
    for(int i=0;i<meta.joint_others.size();i++){
        TransformJoints(meta.joint_others[i]);
    }
}

void DecodeFloats(const string& data, size_t idx, float* pf, size_t len) {
    memcpy(pf, const_cast<char*>(&data[idx]), len * sizeof(float));
}

string DecodeString(const string& data, size_t idx) {
    string result = "";
    int i = 0;
    while(data[idx+i] != 0){
        result.push_back(char(data[idx+i]));
        i++;
    }
    return result;
}

void ReadMetaData_bottomup(MetaData& meta, const string& data, size_t offset3, size_t offset1) { //very specific to genLMDB.py
    // ------------------- Dataset name ----------------------
    meta.dataset = DecodeString(data, offset3);
    // ------------------- Image Dimension -------------------
    float height, width;
    DecodeFloats(data, offset3+offset1, &height, 1);
    DecodeFloats(data, offset3+offset1+4, &width, 1);
    meta.img_size = Size(width, height);
    float image_id;
    DecodeFloats(data, offset3+offset1+8, &image_id, 1);
    meta.image_id = (int) image_id;

    // ----------- Validation, nop, counters -----------------
    meta.isValidation = (data[offset3+2*offset1]==0 ? false : true);
    meta.numOtherPeople = (int)data[offset3+2*offset1+1];
    meta.people_index = (int)data[offset3+2*offset1+2];
    float annolist_index;
    DecodeFloats(data, offset3+2*offset1+3, &annolist_index, 1);
    meta.annolist_index = (int)annolist_index;
    float write_number;
    DecodeFloats(data, offset3+2*offset1+7, &write_number, 1);
    meta.write_number = (int)write_number;
    float total_write_number;
    DecodeFloats(data, offset3+2*offset1+11, &total_write_number, 1);
    meta.total_write_number = (int)total_write_number;
    meta.num_keypoints_self = (int)data[offset3+2*offset1+15];
    DecodeFloats(data, offset3+2*offset1+16, &(meta.segmentation_area), 1);

    // count epochs according to counters
    static int cur_epoch = -1;
    if(meta.write_number == 0){
        cur_epoch++;
    }
    meta.epoch = cur_epoch;
    if(meta.write_number % 1000 == 0){
        LOG(INFO) << "dataset: " << meta.dataset << " mete.type " << meta.type << "; img_size: " << meta.img_size <<  "; image_id: " << meta.image_id << "; meta.annolist_index: " << meta.annolist_index << "; meta.write_number: " << meta.write_number << "; meta.total_write_number: " << meta.total_write_number << "; meta.num_keypoints_self: " << meta.num_keypoints_self << "; meta.segmentation_area: " << meta.segmentation_area << "; meta.epoch: " << meta.epoch;
        LOG(INFO) << "np_in_lmdb" << np_in_lmdb;
    }

    // ------------------- objpos and bbox -----------------------
    DecodeFloats(data, offset3+3*offset1, &meta.objpos.x, 1);
    DecodeFloats(data, offset3+3*offset1+4, &meta.objpos.y, 1);
    //meta.objpos -= Point2f(1,1);
    DecodeFloats(data, offset3+3*offset1+8, &meta.bbox.left, 1);
    DecodeFloats(data, offset3+3*offset1+12, &meta.bbox.top, 1);
    DecodeFloats(data, offset3+3*offset1+16, &meta.bbox.width, 1);
    DecodeFloats(data, offset3+3*offset1+20, &meta.bbox.height, 1);
    // LOG(INFO) << " get o box: pos: " << meta.objpos.x << " " << meta.objpos.y << " , box " << meta.bbox.left << " " << meta.bbox.top << " " << meta.bbox.width << " " << meta.bbox.height << "area "<< meta.segmentation_area;
    //meta.bbox.left -= 1;
    //meta.bbox.top -= 1;

    //LOG(INFO) << "objpos: " << meta.objpos << "; bbox: " << meta.bbox.left << "," << meta.bbox.top << ","
    //                                                     << meta.bbox.width << "," << meta.bbox.height;

    // ------------ scale_self, joint_self --------------
    DecodeFloats(data, offset3+4*offset1, &meta.scale_self, 1);
    meta.joint_self.joints.resize(np_in_lmdb);
    meta.joint_self.isVisible.resize(np_in_lmdb);
    for(int i=0; i<np_in_lmdb; i++){
        DecodeFloats(data, offset3+5*offset1+4*i, &meta.joint_self.joints[i].x, 1);
        DecodeFloats(data, offset3+6*offset1+4*i, &meta.joint_self.joints[i].y, 1);
        meta.joint_self.joints[i] -= Point2f(1,1); //from matlab 1-index to c++ 0-index
        float isVisible;
        DecodeFloats(data, offset3+7*offset1+4*i, &isVisible, 1);
        if (isVisible == 3){ 
            meta.joint_self.isVisible[i] = 3; 
        } else{ 
            meta.joint_self.isVisible[i] = (isVisible == 0) ? 0 : 1; 
        }
            // if(meta.joint_self.joints[i].x < 0 || meta.joint_self.joints[i].y < 0 || meta.joint_self.joints[i].x >= meta.img_size.width || meta.joint_self.joints[i].y >= meta.img_size.height){ meta.joint_self.isVisible[i] = 2; // 2 means cropped, 0 means occluded by still on image }
        
        //LOG(INFO) << meta.joint_self.joints[i].x << " " << meta.joint_self.joints[i].y << " " << meta.joint_self.isVisible[i];
    }

    //others (7 lines loaded)
    meta.objpos_other.resize(meta.numOtherPeople);
    meta.bboxes_other.resize(meta.numOtherPeople);
    meta.num_keypoints_others.resize(meta.numOtherPeople);
    meta.segmentation_area_others.resize(meta.numOtherPeople);
    meta.scale_other.resize(meta.numOtherPeople);
    meta.joint_others.resize(meta.numOtherPeople);

    for(int p=0; p<meta.numOtherPeople; p++){
        DecodeFloats(data, offset3+(8+p)*offset1, &meta.objpos_other[p].x, 1);
        DecodeFloats(data, offset3+(8+p)*offset1+4, &meta.objpos_other[p].y, 1);
        //meta.objpos_other[p] -= Point2f(1,1);
        DecodeFloats(data, offset3+(8+p)*offset1+8, &meta.bboxes_other[p].left, 1);
        DecodeFloats(data, offset3+(8+p)*offset1+12, &meta.bboxes_other[p].top, 1);
        DecodeFloats(data, offset3+(8+p)*offset1+16, &meta.bboxes_other[p].width, 1);
        DecodeFloats(data, offset3+(8+p)*offset1+20, &meta.bboxes_other[p].height, 1);
        //meta.bboxes_other[p].left -= 1;
        //meta.bboxes_other[p].top -= 1;

        meta.num_keypoints_others[p] = int(data[offset3+(8+p)*offset1+24]);
        DecodeFloats(data, offset3+(8+p)*offset1+25, &meta.segmentation_area_others[p], 1);

        DecodeFloats(data, offset3+(8+meta.numOtherPeople)*offset1+4*p, &meta.scale_other[p], 1);

        //LOG(INFO) << "other " << p << ": objpos: " << meta.objpos_other[p];
        //LOG(INFO) << "other " << p << ": bbox: " << meta.bboxes_other[p].left << "," << meta.bboxes_other[p].top << ","
        //                                         << meta.bboxes_other[p].width << "," << meta.bboxes_other[p].height;
        //LOG(INFO) << "other " << p << ": num_keypoints: " << meta.num_keypoints_others[p];
        //LOG(INFO) << "other " << p << ": segmentation_area: " << meta.segmentation_area_others[p];
    }
    //8 + numOtherPeople lines loaded
    for(int p=0; p<meta.numOtherPeople; p++){
        meta.joint_others[p].joints.resize(np_in_lmdb);
        meta.joint_others[p].isVisible.resize(np_in_lmdb);
        for(int i=0; i<np_in_lmdb; i++){
            DecodeFloats(data, offset3+(9+meta.numOtherPeople+3*p)*offset1+4*i, &meta.joint_others[p].joints[i].x, 1);
            DecodeFloats(data, offset3+(9+meta.numOtherPeople+3*p+1)*offset1+4*i, &meta.joint_others[p].joints[i].y, 1);
            meta.joint_others[p].joints[i] -= Point2f(1,1);
            float isVisible;
            DecodeFloats(data, offset3+(9+meta.numOtherPeople+3*p+2)*offset1+4*i, &isVisible, 1);
            meta.joint_others[p].isVisible[i] = (isVisible == 0) ? 0 : 1;
            
            //LOG(INFO) << meta.joint_others[p].joints[i].x << " " << meta.joint_others[p].joints[i].y << " " << meta.joint_others[p].isVisible[i];
        }
    }
    // commend on label of visible of joint:
    // 0: occur/hide
    // 1: normal
    // 2: miss label
    // LOG(INFO) << "Meta read done.";
    ofstream myfile;
    myfile.open("COCO_train_list.txt", ios::out | ios::app);

    char filename[100];
    sprintf(filename, "COCO_train_%012d_%d.png", meta.annolist_index, meta.numOtherPeople+1);
    myfile << filename << " ";
    myfile << 1 + meta.numOtherPeople << " " ;
    // myfile << meta.bbox.left << " " << meta.bbox.top << " " << meta.bbox.width << " " << meta.bbox.height << " ";
    myfile 
        << meta.objpos.x + 1 << " " 
        << meta.objpos.y + 1 << " " 
        << meta.bbox.width << " " 
        << meta.bbox.height << " ";
    MetaData meta_temp = meta;
    // TransformMetaJoints(meta_temp); 
    int left = 6;
    int right = 7;
    for(int i=0; i<np_in_lmdb; i++){
        // c++; myfile << "(" << c << ") ";
        /**
          if(meta_temp.joint_self.isVisible[i] == 2) {
          myfile  << 0 << " " << 0 << " " << 0 << " ";
          continue;
          }else if(meta_temp.joint_self.isVisible[i] == 0 &&
          meta_temp.joint_self.joints[i].x != 0 &&
          meta_temp.joint_self.joints[i].y != 0 )
          meta_temp.joint_self.isVisible[i] = 1;
         **/
        myfile 
            // << joint_name_vec_[i] << " "
            << meta_temp.joint_self.joints[i].x + 1 << " "
            << meta_temp.joint_self.joints[i].y + 1<< " "  
            << meta_temp.joint_self.isVisible[i] << " ";
    }

    for(int p=0; p<meta_temp.numOtherPeople; p++){
        // c = 0;
        myfile 
            << meta_temp.objpos_other[p].x + 1 << " "
            << meta_temp.objpos_other[p].y + 1 << " "
            << meta_temp.bboxes_other[p].width << " "
            << meta_temp.bboxes_other[p].height << " ";
        for(int i=0; i<np_in_lmdb; i++){
            /**
              if(meta_temp.joint_others[p].isVisible[i] == 2){
              myfile  << 0 << " " << 0 << " " << 0 << " ";
              continue;
              }else if(meta_temp.joint_others[p].isVisible[i] == 0 &&
              meta_temp.joint_others[p].joints[i].x != 0 &&
              meta_temp.joint_others[p].joints[i].y != 0)
              meta_temp.joint_others[p].isVisible[i] = 1;
             **/  
            myfile 
                << meta_temp.joint_others[p].joints[i].x + 1<< " " 
                << meta_temp.joint_others[p].joints[i].y + 1<< " "
                << meta_temp.joint_others[p].isVisible[i] << " " ;
        }
    }
    myfile << "\n";
    myfile.close();
}

int main(){
    scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
    db->Open("/data2/zengxiaohui/dataset/lmdb_trainVal/", db::READ);

    scoped_ptr<db::Cursor> cursor(db->NewCursor());
    int count  = 0;
    while(cursor->valid()) {
        Datum datum;
        datum.ParseFromString(cursor->value());
        DecodeDatumNative(&datum);
        const std::string& data = datum.data();
        const int datum_channels = datum.channels();
        const int datum_height = datum.height();
        const int datum_width = datum.width();

            Mat img = Mat::zeros(datum.height(), datum.width(), CV_8UC3);
            int offset = img.rows * img.cols;
        int offset3 = 3 * offset;
        int offset1 = datum_width;
        MetaData meta;
        ReadMetaData_bottomup(meta, data, offset3, offset1);
        if(false){
            Mat mask_miss = Mat::ones(datum_height, datum_width, CV_8UC1);
            Mat mask_all = Mat::zeros(datum_height, datum_width, CV_8UC1);
            // img

            int dindex;
            float d_element;
            const bool has_uint8 = data.size() > 0;
            for (int i = 0; i < img.rows; ++i) {
                for (int j = 0; j < img.cols; ++j) {
                    Vec3b& rgb = img.at<Vec3b>(i, j);
                    for(int c = 0; c < 3; c++){
                        dindex = c*offset + i*img.cols + j;
                        if (has_uint8)
                            d_element = static_cast<float>(static_cast<uint8_t>(data[dindex]));
                        else
                            d_element = datum.float_data(dindex);
                        rgb[c] = d_element;
                    }

                    //if(mode >= 5){
                    dindex = 4*offset + i*img.cols + j;
                    if (has_uint8)
                        d_element = static_cast<float>(static_cast<uint8_t>(data[dindex]));
                    else
                        d_element = datum.float_data(dindex);
                    // if (round(d_element/255)!=1 && round(d_element/255)!=0){
                    //   cout << d_element << " " << round(d_element/255) << endl;
                    // }
                    mask_miss.at<uchar>(i, j) = d_element; //round(d_element/255);
                    //}

                    //if(mode == 6){
                    dindex = 5*offset + i*img.cols + j; // i-y, img.cols-grid_x, j-x
                    if (has_uint8)
                        d_element = static_cast<float>(static_cast<uint8_t>(data[dindex]));
                    else
                        d_element = datum.float_data(dindex);
                    mask_all.at<uchar>(i, j) = d_element;
                    //}
                }
            }

            char name[100];
            sprintf(name, "COCO_train_%012d_%d.png", meta.annolist_index, meta.numOtherPeople+1);
            meta.file_name = name;

            // LOG(INFO) << " testing image save mask_miss & mask_all" ;
            static int count = 0; count ++;
            sprintf(name, "./COCO_LMDB_png/train2014mask/mask_COCO_train_%012d_%d.png", meta.annolist_index, meta.numOtherPeople+1);
            imwrite(name, mask_miss);
            // sprintf(name, "%d_all.jpg", meta.annolist_index);
            // imwrite(name, mask_all);
            char filename[100];
            sprintf(name, "./COCO_LMDB_png/train2014/COCO_train_%012d_%d.png", meta.annolist_index, meta.numOtherPeople+1);
            imwrite(name, img);
        }
        // end img
        count ++;
        if (count % 1000 == 0) {
            LOG(INFO) << "Processed " << count << " files.";
            // LOG(INFO) << "save as " << name ;
        }
        cursor->Next();
    }
    LOG(INFO) << count ;
    //  const string& data = datum.data();
}
