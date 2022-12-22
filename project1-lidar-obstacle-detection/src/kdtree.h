#include "render/render.h"

struct Node {
    std::vector<float> point;
    int id;
    Node* left;
    Node* right;

    Node(std::vector<float> arr, int setId)
        : point(arr), id(setId), left(NULL), right(NULL) {}

    ~Node() {
        delete left;
        delete right;
    }
};

struct KdTree {
    Node* root;

    KdTree() : root(NULL) {}

    ~KdTree() { delete root; }

    void insert(std::vector<float> point, int id) {
        // TODO: Fill in this function to insert a new point into the tree
        // the function should create a new node and place correctly with in the
        // root
        Node* new_node = new Node(point, id);
        if (root == NULL) {
            root = new_node;
            return;
        }

        int depth = 0;
        Node* iter_node = root;
        while (true) {
            if (new_node->point.at(depth % point.size()) >
                iter_node->point.at(depth % point.size())) {
                if (iter_node->right != NULL) {
                    iter_node = iter_node->right;
                } else {
                    iter_node->right = new_node;
                    break;
                }
            } else {
                if (iter_node->left != NULL) {
                    iter_node = iter_node->left;
                } else {
                    iter_node->left = new_node;
                    break;
                }
            }
            depth += 1;
        }
    }

    void searchHelper(std::vector<float> target, Node* node, int depth,
                      float distanceTol, std::vector<int>& ids) {
        if (node != NULL) {
            if (node->point.at(0) >= (target.at(0) - distanceTol) &&
                node->point.at(0) <= (target.at(0) + distanceTol) &&
                node->point.at(1) >= (target.at(1) - distanceTol) &&
                node->point.at(1) <= (target.at(1) + distanceTol) &&
                node->point.at(2) >= (target.at(2) - distanceTol) &&
                node->point.at(2) <= (target.at(2) + distanceTol)) {
                float distance = sqrt(pow(node->point.at(0) - target.at(0), 2) +
                                      pow(node->point.at(1) - target.at(1), 2) +
                                      pow(node->point.at(2) - target.at(2), 2));
                if (distance <= distanceTol) {
                    ids.push_back(node->id);
                }
            }

            if (target.at(depth % target.size()) - distanceTol <=
                node->point.at(depth % target.size())) {
                searchHelper(target, node->left, depth + 1, distanceTol, ids);
            }
            if ((target.at(depth % target.size()) + distanceTol >
                 node->point.at(depth % target.size()))) {
                searchHelper(target, node->right, depth + 1, distanceTol, ids);
            }
        }
    }

    // return a list of point ids in the tree that are within distance of target
    std::vector<int> search(std::vector<float> target, float distanceTol) {
        std::vector<int> ids;
        searchHelper(target, root, 0, distanceTol, ids);
        return ids;
    }
};
