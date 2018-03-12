#ifndef DATA_H
#define DATA_H

#include "tf/util/Base.h"
#include "tf/struct/Tuple.h"
#include "tf/struct/Triple.h"
#include "tf/util/FileUtil.h"

using namespace FileUtil;

class Data {
public:

    int N; // num of entity
    int K; // num of relation
    int num_of_training_triples;
    int num_of_testing_triples;
    int num_of_validation_triples;

    int num_of_testing_triples_positive;
    int num_of_validation_triples_positive;

    int num_of_testing_triples_negative;
    int num_of_validation_triples_negative;

    vector<Triple<string> > training_triple_strs;
    vector<Triple<int> > training_triples;
    vector<Triple<string> > testing_triple_strs;
    vector<Triple<string> > valiation_triple_strs;

    vector<Triple<string> > testing_triple_strs_positive;
    vector<Triple<string> > valiation_triple_strs_positive;

    vector<Triple<string> > testing_triple_strs_negative;
    vector<Triple<string> > valiation_triple_strs_negative;

    unordered_map<string, int> entity_encoder;
    unordered_map<string, int> relation_encoder;
    unordered_map<int, string> entity_decoder;
    unordered_map<int, string> relation_decoder;

    unordered_map<int, vector<Tuple<int> > > relation2tupleList_mapping;
    unordered_map<int, vector<Tuple<int> > > relation2tupleTestList_mapping;
    unordered_map<int, vector<Tuple<int> > > relation2tupleValidationList_mapping;

    unordered_map<int, vector<Tuple<int> > > relation2tupleTestPosList_mapping;
    unordered_map<int, vector<Tuple<int> > > relation2tupleTestNegList_mapping;
    unordered_map<int, vector<Tuple<int> > > relation2tupleValidationPosList_mapping;
    unordered_map<int, vector<Tuple<int> > > relation2tupleValidationNegList_mapping;

    // relation -> (Subject, (Object1, Object2...) )
    unordered_map<int, unordered_map<int, list<int> > > relation2TrainTupleSet_sub;
    unordered_map<int, unordered_map<int, list<int> > > relation2TrainTupleSet_obj;

    unordered_map<int, unordered_map<int, list<int> > > relation2TestTupleSet_sub;
    unordered_map<int, unordered_map<int, list<int> > > relation2TestTupleSet_obj;

    unordered_map<int, unordered_map<int, list<int> > > relation2ValidTupleSet_sub;
    unordered_map<int, unordered_map<int, list<int> > > relation2ValidTupleSet_obj;

    map<pair<int, int>, vector<int> > trainSubRel2Obj;
    map<pair<int, int>, vector<int> > trainObjRel2Sub;
    vector<pair<int, int> > trainSubRel2ObjKeys;
    vector<pair<int, int> > trainObjRel2SubKeys;

    map<pair<int, int>, vector<int> > testSubRel2Obj;
    map<pair<int, int>, vector<int> > testObjRel2Sub;
    vector<pair<int, int> > testSubRel2ObjKeys;
    vector<pair<int, int> > testObjRel2SubKeys;

    map<pair<int, int>, vector<int> > validSubRel2Obj;
    map<pair<int, int>, vector<int> > validObjRel2Sub;
    vector<pair<int, int> > validSubRel2ObjKeys;
    vector<pair<int, int> > validObjRel2SubKeys;

    unordered_map<int, int> subject_count;
    unordered_map<int, int> object_count;
    unordered_map<int, int> relation_count;

    // for TransE
    unordered_map<int, unordered_map<int, int> > left_entity, right_entity;
    unordered_map<int, value_type> left_num, right_num;

    // cube for RESCAL
#ifdef use_mkl

    // CSR format: https://software.intel.com/en-us/node/599882
    vector<value_type *> X_values;
    vector<int *> X_columns;
    vector<int *> X_pointer;
#else
    vector<SparseMatrix> X;
    vector<SparseMatrix> XT;
#endif

    ~Data() {
#ifdef use_mkl
        for (int i = 0; i < X_values.size(); i++) {
            delete[] X_values[i];
        }

        for (int i = 0; i < X_columns.size(); i++) {
            delete[] X_columns[i];
        }

        for (int i = 0; i < X_pointer.size(); i++) {
            delete[] X_pointer[i];
        }

#endif
    }

    void read_triples(const string train_data_path, const string valid_data_path, const string test_data_path) {
        read_triple_data(train_data_path, training_triple_strs, num_of_training_triples);
        read_triple_data(test_data_path, testing_triple_strs, num_of_testing_triples);
        read_triple_data(valid_data_path, valiation_triple_strs, num_of_validation_triples);
    }

//    void read_triples(const string train_data_path, const string positive_valid_data_path,
//                      const string negative_valid_data_path, const string positive_test_data_path,
//                      const string negative_test_data_path) {
//
//        read_triple_data(train_data_path, training_triple_strs, num_of_training_triples);
//
//        read_triple_data(positive_test_data_path, testing_triple_strs, num_of_testing_triples, true);
//        read_triple_data(negative_test_data_path, testing_triple_strs, num_of_testing_triples, false);
//
//        read_triple_data(positive_valid_data_path, valiation_triple_strs, num_of_validation_triples, true);
//        read_triple_data(negative_valid_data_path, valiation_triple_strs, num_of_validation_triples, false);
//    }

    void read_triples_separate(const string train_data_path, const string positive_valid_data_path,
                      const string negative_valid_data_path, const string positive_test_data_path,
                      const string negative_test_data_path) {

        read_triple_data(train_data_path, training_triple_strs, num_of_training_triples);

        read_triple_data(positive_test_data_path, testing_triple_strs_positive, num_of_testing_triples_positive);
        read_triple_data(negative_test_data_path, testing_triple_strs_negative, num_of_testing_triples_negative);

        read_triple_data(positive_valid_data_path, valiation_triple_strs_positive, num_of_validation_triples_positive);
        read_triple_data(negative_valid_data_path, valiation_triple_strs_negative, num_of_validation_triples_negative);
    }

    void encode_triples_rescal_rank(){
        encode_triples();
    }

    void encode_triples_hole(){
        encode_triples();
    }

    void encode_triples_ensemble(){
        encode_triples();

        for (int i = 0; i < relation_encoder.size(); i++) {
            value_type sum1 = 0, sum2 = 0;
            for (unordered_map<int, int>::iterator it = left_entity[i].begin(); it != left_entity[i].end(); it++) {
                sum1++;
                sum2 += it->second;
            }
            left_num[i] = sum2 / sum1;
        }

        for (int i = 0; i < relation_encoder.size(); i++) {
            double sum1 = 0, sum2 = 0;
            for (unordered_map<int, int>::iterator it = right_entity[i].begin(); it != right_entity[i].end(); it++) {
                sum1++;
                sum2 += it->second;
            }
            right_num[i] = sum2 / sum1;
        }
    }

    void encode_triples_triple_classification(){

        int relation_id, subject_id, object_id;
        pair<int, int> subRel;
        pair<int, int> objRel;

        for (Triple<string> triple_str : training_triple_strs) {

            if (relation_encoder.find(triple_str.relation) == relation_encoder.end()) {
                relation_id = relation_encoder.size();
                relation_encoder[triple_str.relation] = relation_id;
                relation_decoder[relation_id] = triple_str.relation;
            } else {
                relation_id = relation_encoder[triple_str.relation];
            }

            if (entity_encoder.find(triple_str.subject) == entity_encoder.end()) {
                subject_id = entity_encoder.size();
                entity_encoder[triple_str.subject] = subject_id;
                entity_decoder[subject_id] = triple_str.subject;
            } else {
                subject_id = entity_encoder[triple_str.subject];
            }

            if (entity_encoder.find(triple_str.object) == entity_encoder.end()) {
                object_id = entity_encoder.size();
                entity_encoder[triple_str.object] = object_id;
                entity_decoder[object_id] = triple_str.object;
            } else {
                object_id = entity_encoder[triple_str.object];
            }

            if (relation2tupleList_mapping.find(relation_id) == relation2tupleList_mapping.end()) {
                relation2tupleList_mapping[relation_id] = vector<Tuple<int> >();
            }

            relation2tupleList_mapping[relation_id].push_back(Tuple<int>(subject_id, object_id));

            subRel = make_pair(subject_id, relation_id);
            objRel = make_pair(object_id, relation_id);

            if(trainSubRel2Obj.find(subRel) == trainSubRel2Obj.end()){
                trainSubRel2Obj[subRel] = vector<int>();
            }
            trainSubRel2Obj[subRel].push_back(object_id);

            if(trainObjRel2Sub.find(objRel) == trainObjRel2Sub.end()){
                trainObjRel2Sub[objRel] = vector<int>();
            }
            trainObjRel2Sub[objRel].push_back(subject_id);

            if(subject_count.count(subject_id)>0){
                subject_count[subject_id]++;
            } else{
                subject_count[subject_id] = 1;
            }

            if(object_count.count(object_id)>0){
                object_count[object_id]++;
            } else{
                object_count[object_id] = 1;
            }

            if(relation_count.count(relation_id)>0){
                relation_count[relation_id]++;
            }else{
                relation_count[relation_id] = 1;
            }

            // TransE
            if(left_entity.find(relation_id) == left_entity.end()){
                left_entity[relation_id] = unordered_map<int, int>();
            }

            if(left_entity[relation_id].find(subject_id) == left_entity[relation_id].end()){
                left_entity[relation_id][subject_id] = 1;
            } else {
                left_entity[relation_id][subject_id]++;
            }

            if(right_entity.find(relation_id) == right_entity.end()){
                right_entity[relation_id] = unordered_map<int, int>();
            }

            if(right_entity[relation_id].find(object_id) == right_entity[relation_id].end()){
                right_entity[relation_id][object_id] = 1;
            } else {
                right_entity[relation_id][object_id]++;
            }

            training_triples.push_back(Triple<int>(subject_id, relation_id, object_id));
        }

        for (auto ptr = trainSubRel2Obj.begin(); ptr != trainSubRel2Obj.end(); ptr++) {
            trainSubRel2ObjKeys.push_back(ptr->first);
        }

        for (auto ptr = trainObjRel2Sub.begin(); ptr != trainObjRel2Sub.end(); ptr++) {
            trainObjRel2SubKeys.push_back(ptr->first);
        }

        N = entity_decoder.size();
        K = relation_decoder.size();

        for (Triple<string> triple_str: testing_triple_strs_positive) {
            if (relation_encoder.find(triple_str.relation) == relation_encoder.end() ||
                entity_encoder.find(triple_str.subject) == entity_encoder.end() ||
                entity_encoder.find(triple_str.object) == entity_encoder.end()) {
                cerr << "relation or entity is not in training set" << endl;
                exit(1);
            }

            relation_id = relation_encoder[triple_str.relation];
            subject_id = entity_encoder[triple_str.subject];
            object_id = entity_encoder[triple_str.object];
            relation2tupleTestPosList_mapping[relation_id].push_back(Tuple<int>(subject_id, object_id));
        }

        for (Triple<string> triple_str: testing_triple_strs_negative) {
            if (relation_encoder.find(triple_str.relation) == relation_encoder.end() ||
                entity_encoder.find(triple_str.subject) == entity_encoder.end() ||
                entity_encoder.find(triple_str.object) == entity_encoder.end()) {
                cerr << "relation or entity is not in training set" << endl;
                exit(1);
            }

            relation_id = relation_encoder[triple_str.relation];
            subject_id = entity_encoder[triple_str.subject];
            object_id = entity_encoder[triple_str.object];
            relation2tupleTestNegList_mapping[relation_id].push_back(Tuple<int>(subject_id, object_id));
        }

        for (Triple<string> triple_str: valiation_triple_strs_positive) {
            if (relation_encoder.find(triple_str.relation) == relation_encoder.end() ||
                entity_encoder.find(triple_str.subject) == entity_encoder.end() ||
                entity_encoder.find(triple_str.object) == entity_encoder.end()) {
                cerr << "relation or entity is not in training set" << endl;
                exit(1);
            }

            relation_id = relation_encoder[triple_str.relation];
            subject_id = entity_encoder[triple_str.subject];
            object_id = entity_encoder[triple_str.object];
            relation2tupleValidationPosList_mapping[relation_id].push_back(Tuple<int>(subject_id, object_id));
        }

        for (Triple<string> triple_str: valiation_triple_strs_negative) {
            if (relation_encoder.find(triple_str.relation) == relation_encoder.end() ||
                entity_encoder.find(triple_str.subject) == entity_encoder.end() ||
                entity_encoder.find(triple_str.object) == entity_encoder.end()) {
                cerr << "relation or entity is not in training set" << endl;
                exit(1);
            }

            relation_id = relation_encoder[triple_str.relation];
            subject_id = entity_encoder[triple_str.subject];
            object_id = entity_encoder[triple_str.object];
            relation2tupleValidationNegList_mapping[relation_id].push_back(Tuple<int>(subject_id, object_id));
        }

    }

    void encode_triples_transe(){
        encode_triples();

        for (int i = 0; i < relation_encoder.size(); i++) {
            value_type sum1 = 0, sum2 = 0;
            for (unordered_map<int, int>::iterator it = left_entity[i].begin(); it != left_entity[i].end(); it++) {
                sum1++;
                sum2 += it->second;
            }
            left_num[i] = sum2 / sum1;
        }

        for (int i = 0; i < relation_encoder.size(); i++) {
            double sum1 = 0, sum2 = 0;
            for (unordered_map<int, int>::iterator it = right_entity[i].begin(); it != right_entity[i].end(); it++) {
                sum1++;
                sum2 += it->second;
            }
            right_num[i] = sum2 / sum1;
        }

    }

    void encode_triples_rescal(){
        encode_triples();

#ifdef use_mkl

        X_values.resize(relation2tupleList_mapping.size());
        X_columns.resize(relation2tupleList_mapping.size());
        X_pointer.resize(relation2tupleList_mapping.size());

        // convert to CSR
        for (int relation_id = 0; relation_id < relation2tupleList_mapping.size(); relation_id++) {

            vector<Tuple<int> > &tuples = relation2tupleList_mapping[relation_id];

            X_values[relation_id] = new value_type[tuples.size()];
            X_columns[relation_id] = new int[tuples.size()];
            X_pointer[relation_id] = new int[entity_encoder.size() + 1];

            value_type *sub_X_values = X_values[relation_id];
            int *sub_X_columns = X_columns[relation_id];
            int *sub_X_pointer = X_pointer[relation_id];

            vector<int> row_counts(entity_encoder.size(), 0);

            for (int i = 0; i < tuples.size(); i++) {

                Tuple<int> &tuple = tuples[i];
                sub_X_values[i] = 1;
                sub_X_columns[i] = tuple.object;  // zero based indexing

                row_counts[tuple.subject]++;
            }

            sub_X_pointer[0] = 0; // it is the same as row_index in 3 array CSR
            for (int i = 1; i < entity_encoder.size() + 1; i++) {
                sub_X_pointer[i] = sub_X_pointer[i - 1] + row_counts[i - 1];
            }

        }
#else
        X.resize(relation_encoder.size());
        XT.resize(relation_encoder.size());

        for (int i = 0; i < relation_encoder.size(); i++) {
            cout << "format sparse matrix " << i << " of " << relation_encoder.size() <<  endl;

            X[i] = SparseMatrix(entity_encoder.size(), entity_encoder.size(), relation2tupleList_mapping[i].size());
            SparseMatrix &Xi = X[i];

            vector<Tuple<int> > &tuples = relation2tupleList_mapping[i];
            for (Tuple<int> tuple:tuples) {
                Xi(tuple.subject, tuple.object) = 1;
            }

            XT[i] = trans(Xi);
        }
#endif
    }

    // check whether faked triple already exists in training data
    bool faked_tuple_exist_train(const int relation_id, const int subject_id, const int object_id) {
        auto train_ptr = relation2tupleList_mapping.find(relation_id);
        if (train_ptr != relation2tupleList_mapping.end()) {
            for (Tuple<int> tuple:train_ptr->second) {
                if ((tuple.subject == subject_id) && (tuple.object == object_id)) {
                    return true;
                }
            }
        }

        return false;
    }

    // check whether faked triple already exists in training data
    bool faked_tuple_exist_train(const int relation_id, const Tuple<int> &faked_tuple) {
        auto train_ptr = relation2tupleList_mapping.find(relation_id);
        if (train_ptr != relation2tupleList_mapping.end()) {
            for (Tuple<int> tuple:train_ptr->second) {
                if (tuple == faked_tuple) {
                    return true;
                }
            }
        }

        return false;
    }

    bool faked_tuple_exist_test(const int relation_id, const Tuple<int> &faked_tuple) {
        auto train_ptr = relation2tupleList_mapping.find(relation_id);
        if (train_ptr != relation2tupleList_mapping.end()) {
            for (Tuple<int> tuple:train_ptr->second) {
                if (tuple == faked_tuple) {
                    return true;
                }
            }
        }

        auto valid_ptr = relation2tupleValidationList_mapping.find(relation_id);
        if (valid_ptr != relation2tupleValidationList_mapping.end()) {
            for (Tuple<int> tuple:valid_ptr->second) {
                if (tuple == faked_tuple) {
                    return true;
                }
            }
        }

        auto test_ptr = relation2tupleTestList_mapping.find(relation_id);
        if (test_ptr != relation2tupleTestList_mapping.end()) {
            for (Tuple<int> tuple:test_ptr->second) {
                if (tuple == faked_tuple) {
                    return true;
                }
            }
        }

        return false;
    }

    bool faked_s_tuple_exist(const int subject_id, const int relation_id, const int object_id) {
        pair<int, int> key = make_pair(object_id, relation_id);

        auto train_ptr = trainObjRel2Sub.find(key);
        if (train_ptr != trainObjRel2Sub.end()) {
            for (int real_subject_id:train_ptr->second) {
                if (real_subject_id == subject_id) {
                    return true;
                }
            }
        }

        auto valid_ptr = validObjRel2Sub.find(key);
        if (valid_ptr != validObjRel2Sub.end()) {
            for (int real_subject_id:valid_ptr->second) {
                if (real_subject_id == subject_id) {
                    return true;
                }
            }
        }

        auto test_ptr = testObjRel2Sub.find(key);
        if (test_ptr != testObjRel2Sub.end()) {
            for (int real_subject_id:test_ptr->second) {
                if (real_subject_id == subject_id) {
                    return true;
                }
            }
        }

        return false;
    }

    bool faked_o_tuple_exist(const int subject_id, const int relation_id, const int object_id) {

        pair<int, int> key = make_pair(subject_id, relation_id);

        auto train_ptr = trainSubRel2Obj.find(key);
        if (train_ptr != trainSubRel2Obj.end()) {
            for (int real_object_id:train_ptr->second) {
                if (real_object_id == object_id) {
                    return true;
                }
            }
        }

        auto valid_ptr = validSubRel2Obj.find(key);
        if (valid_ptr != validSubRel2Obj.end()) {
            for (int real_object_id:valid_ptr->second) {
                if (real_object_id == object_id) {
                    return true;
                }
            }
        }

        auto test_ptr = testSubRel2Obj.find(key);
        if (test_ptr != testSubRel2Obj.end()) {
            for (int real_object_id:test_ptr->second) {
                if (real_object_id == object_id) {
                    return true;
                }
            }
        }

        return false;
    }

    void output_decoder(string output_folder) {

        boost::filesystem::create_directories(output_folder);

        std::ofstream output(output_folder + "/entity_decoder.dat");

        for (int i = 0; i < entity_decoder.size(); i++) {
            output << i << " " << entity_decoder.find(i)->second << endl;
        }

        output.close();

        output.open(output_folder + "/relation_decoder.dat");
        for (int i = 0; i < relation_decoder.size(); i++) {
            output << i << " " << relation_decoder.find(i)->second << endl;
        }
        output.close();
    }

private:

    void encode_triples() {

        int relation_id, subject_id, object_id;
        pair<int, int> subRel;
        pair<int, int> objRel;

        for (Triple<string> triple_str : training_triple_strs) {

            if (relation_encoder.find(triple_str.relation) == relation_encoder.end()) {
                relation_id = relation_encoder.size();
                relation_encoder[triple_str.relation] = relation_id;
                relation_decoder[relation_id] = triple_str.relation;
            } else {
                relation_id = relation_encoder[triple_str.relation];
            }

            if (entity_encoder.find(triple_str.subject) == entity_encoder.end()) {
                subject_id = entity_encoder.size();
                entity_encoder[triple_str.subject] = subject_id;
                entity_decoder[subject_id] = triple_str.subject;
            } else {
                subject_id = entity_encoder[triple_str.subject];
            }

            if (entity_encoder.find(triple_str.object) == entity_encoder.end()) {
                object_id = entity_encoder.size();
                entity_encoder[triple_str.object] = object_id;
                entity_decoder[object_id] = triple_str.object;
            } else {
                object_id = entity_encoder[triple_str.object];
            }

            if (relation2tupleList_mapping.find(relation_id) == relation2tupleList_mapping.end()) {
                relation2tupleList_mapping[relation_id] = vector<Tuple<int> >();
            }

            relation2tupleList_mapping[relation_id].push_back(Tuple<int>(subject_id, object_id));

            subRel = make_pair(subject_id, relation_id);
            objRel = make_pair(object_id, relation_id);

            if(trainSubRel2Obj.find(subRel) == trainSubRel2Obj.end()){
                trainSubRel2Obj[subRel] = vector<int>();
            }
            trainSubRel2Obj[subRel].push_back(object_id);

            if(trainObjRel2Sub.find(objRel) == trainObjRel2Sub.end()){
                trainObjRel2Sub[objRel] = vector<int>();
            }
            trainObjRel2Sub[objRel].push_back(subject_id);

            if(subject_count.count(subject_id)>0){
                subject_count[subject_id]++;
            } else{
                subject_count[subject_id] = 1;
            }

            if(object_count.count(object_id)>0){
                object_count[object_id]++;
            } else{
                object_count[object_id] = 1;
            }

            if(relation_count.count(relation_id)>0){
                relation_count[relation_id]++;
            }else{
                relation_count[relation_id] = 1;
            }

            // TransE
            if(left_entity.find(relation_id) == left_entity.end()){
                left_entity[relation_id] = unordered_map<int, int>();
            }

            if(left_entity[relation_id].find(subject_id) == left_entity[relation_id].end()){
                left_entity[relation_id][subject_id] = 1;
            } else {
                left_entity[relation_id][subject_id]++;
            }

            if(right_entity.find(relation_id) == right_entity.end()){
                right_entity[relation_id] = unordered_map<int, int>();
            }

            if(right_entity[relation_id].find(object_id) == right_entity[relation_id].end()){
                right_entity[relation_id][object_id] = 1;
            } else {
                right_entity[relation_id][object_id]++;
            }

            training_triples.push_back(Triple<int>(subject_id, relation_id, object_id));
        }

        for (auto ptr = trainSubRel2Obj.begin(); ptr != trainSubRel2Obj.end(); ptr++) {
            trainSubRel2ObjKeys.push_back(ptr->first);
        }

        for (auto ptr = trainObjRel2Sub.begin(); ptr != trainObjRel2Sub.end(); ptr++) {
            trainObjRel2SubKeys.push_back(ptr->first);
        }

        for (Triple<string> triple_str: testing_triple_strs) {
            if (relation_encoder.find(triple_str.relation) == relation_encoder.end() ||
                entity_encoder.find(triple_str.subject) == entity_encoder.end() ||
                entity_encoder.find(triple_str.object) == entity_encoder.end()) {
                continue;
            }
            relation_id = relation_encoder[triple_str.relation];
            subject_id = entity_encoder[triple_str.subject];
            object_id = entity_encoder[triple_str.object];
            relation2tupleTestList_mapping[relation_id].push_back(Tuple<int>(subject_id, object_id));

            subRel = make_pair(subject_id, relation_id);
            objRel = make_pair(object_id, relation_id);

            if(testSubRel2Obj.find(subRel) == testSubRel2Obj.end()){
                testSubRel2Obj[subRel] = vector<int>();
            }
            testSubRel2Obj[subRel].push_back(object_id);

            if(testObjRel2Sub.find(objRel) == testObjRel2Sub.end()){
                testObjRel2Sub[objRel] = vector<int>();
            }
            testObjRel2Sub[objRel].push_back(subject_id);
        }

        for(auto ptr = testSubRel2Obj.begin(); ptr!=testSubRel2Obj.end(); ptr++){
            testSubRel2ObjKeys.push_back(ptr->first);
        }

        for(auto ptr = testObjRel2Sub.begin(); ptr!=testObjRel2Sub.end(); ptr++){
            testObjRel2SubKeys.push_back(ptr->first);
        }

        for (Triple<string> triple_str: valiation_triple_strs) {
            if (relation_encoder.find(triple_str.relation) == relation_encoder.end() ||
                entity_encoder.find(triple_str.subject) == entity_encoder.end() ||
                entity_encoder.find(triple_str.object) == entity_encoder.end()) {
                continue;
            }
            relation_id = relation_encoder[triple_str.relation];
            subject_id = entity_encoder[triple_str.subject];
            object_id = entity_encoder[triple_str.object];
            relation2tupleValidationList_mapping[relation_id].push_back(Tuple<int>(subject_id, object_id));

            subRel = make_pair(subject_id, relation_id);
            objRel = make_pair(object_id, relation_id);

            if(validSubRel2Obj.find(subRel) == validSubRel2Obj.end()){
                validSubRel2Obj[subRel] = vector<int>();
            }
            validSubRel2Obj[subRel].push_back(object_id);

            if(validObjRel2Sub.find(objRel) == validObjRel2Sub.end()){
                validObjRel2Sub[objRel] = vector<int>();
            }
            validObjRel2Sub[objRel].push_back(subject_id);
        }

        for(auto ptr = validSubRel2Obj.begin(); ptr!=validSubRel2Obj.end(); ptr++){
            validSubRel2ObjKeys.push_back(ptr->first);
        }

        for(auto ptr = validObjRel2Sub.begin(); ptr!=validObjRel2Sub.end(); ptr++){
            validObjRel2SubKeys.push_back(ptr->first);
        }

        for (int relation_id = 0; relation_id < relation2tupleList_mapping.size(); relation_id++) {

            relation2TrainTupleSet_sub[relation_id] = unordered_map<int, list<int> >();
            relation2TrainTupleSet_obj[relation_id] = unordered_map<int, list<int> >();

            relation2TestTupleSet_sub[relation_id] = unordered_map<int, list<int> >();
            relation2TestTupleSet_obj[relation_id] = unordered_map<int, list<int> >();

            relation2ValidTupleSet_sub[relation_id] = unordered_map<int, list<int> >();
            relation2ValidTupleSet_obj[relation_id] = unordered_map<int, list<int> >();

            vector<Tuple<int> > &train_tuples = relation2tupleList_mapping[relation_id];
            sort(train_tuples.begin(), train_tuples.end());

            for (auto tuple:train_tuples) {
                if(relation2TrainTupleSet_sub[relation_id].find(tuple.subject)==relation2TrainTupleSet_sub[relation_id].end()){
                    relation2TrainTupleSet_sub[relation_id][tuple.subject] = list<int>();
                }
                relation2TrainTupleSet_sub[relation_id][tuple.subject].push_back(tuple.object);

                if(relation2TrainTupleSet_obj[relation_id].find(tuple.object)==relation2TrainTupleSet_obj[relation_id].end()){
                    relation2TrainTupleSet_obj[relation_id][tuple.object] = list<int>();
                }
                relation2TrainTupleSet_obj[relation_id][tuple.object].push_back(tuple.subject);
            }

            vector<Tuple<int> > &test_tuples = relation2tupleTestList_mapping[relation_id];
            sort(test_tuples.begin(), test_tuples.end());

            for (auto tuple:test_tuples) {
                if(relation2TestTupleSet_sub[relation_id].find(tuple.subject)==relation2TestTupleSet_sub[relation_id].end()){
                    relation2TestTupleSet_sub[relation_id][tuple.subject] = list<int>();
                }
                relation2TestTupleSet_sub[relation_id][tuple.subject].push_back(tuple.object);

                if(relation2TestTupleSet_obj[relation_id].find(tuple.object)==relation2TestTupleSet_obj[relation_id].end()){
                    relation2TestTupleSet_obj[relation_id][tuple.object] = list<int>();
                }
                relation2TestTupleSet_obj[relation_id][tuple.object].push_back(tuple.subject);
            }

            vector<Tuple<int> > &valid_tuples = relation2tupleValidationList_mapping[relation_id];
            sort(valid_tuples.begin(), valid_tuples.end());

            for (auto tuple:valid_tuples) {
                if(relation2ValidTupleSet_sub[relation_id].find(tuple.subject)==relation2ValidTupleSet_sub[relation_id].end()){
                    relation2ValidTupleSet_sub[relation_id][tuple.subject] = list<int>();
                }
                relation2ValidTupleSet_sub[relation_id][tuple.subject].push_back(tuple.object);

                if(relation2ValidTupleSet_obj[relation_id].find(tuple.object)==relation2ValidTupleSet_obj[relation_id].end()){
                    relation2ValidTupleSet_obj[relation_id][tuple.object] = list<int>();
                }
                relation2ValidTupleSet_obj[relation_id][tuple.object].push_back(tuple.subject);
            }
        }

        N = entity_decoder.size();
        K = relation_decoder.size();
    }
};

#endif //DATA_H