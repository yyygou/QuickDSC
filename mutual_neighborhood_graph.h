/*
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include <map>
#include <stack>
#include <set>
#include <vector>
#include <algorithm>
#include <string>
#include <stdlib.h>
#include <fstream>
#include <math.h>
using namespace std;

struct Node
{
    /*
    Node struct for our k-NN or neighborhood Graph
    NodeBasic has no attribute set<>
    */
    int index;
    int rank;
    Node *parent;
    set<Node *> children;
    Node(int idx)
    {
        index = idx;
        rank = 0;
        parent = NULL;
        children.clear();
    }
};

struct Graph
{
    /*
    Graph struct.
    Allows us to build the graph one node at a time
    */
    vector<Node *> nodes;
    map<int, Node *> M;
    set<Node *> intersecting_sets;
    Graph()
    {
        M.clear();
        intersecting_sets.clear();
        nodes.clear();
    }

    // Node *get_root(Node *node)
    // {

    //     if (node->parent != NULL)
    //     {
    //         node->parent->children.erase(node);
    //         node->parent = get_root(node->parent);
    //         node->parent->children.insert(node);
    //         return node->parent;
    //     }
    //     else
    //     {
    //         return node;
    //     }
    // }

    // this part is modified
    Node *get_root(Node *node)
    {

        Node *res = node;

        while (res->parent != NULL)
        {
            res = res->parent;
        }
        return res;
    }

    void add_node(int idx)
    {
        nodes.push_back(new Node(idx));
        M[idx] = nodes[nodes.size() - 1];
    }

    //comment: significant to the graph construction, key function
    //form a mutual knn graph, and the absolute parent order is not significant here, just connect them together
    void add_edge(int n1, int n2)
    {
        //find the roots of two nodes
        Node *r1 = get_root(M[n1]);
        Node *r2 = get_root(M[n2]);
        //2. compare the rank of two roots
        if (r1 != r2)
        {
            //3. if n1's root is higher in terms of the rank, set r1 as the parent of r1
            if (r1->rank > r2->rank)
            {
                r2->parent = r1;
                r1->children.insert(r2);
                //4. remove r2 from the intersecting sets
                if (intersecting_sets.count(r2))
                {
                    intersecting_sets.erase(r2);
                    intersecting_sets.insert(r1);
                }
            }
            else
            {
                //5. if r1.rank <= r2.rank ????
                r1->parent = r2;
                r2->children.insert(r1);
                if (intersecting_sets.count(r1))
                {
                    intersecting_sets.erase(r1);
                    intersecting_sets.insert(r2);
                }

                //6. add r2's rank
                if (r1->rank == r2->rank)
                {
                    r2->rank++;
                }
            }
        }
    }

    vector<pair<int, int>> get_connected_component_with_parent(int n)
    {
        Node *r = get_root(M[n]);
        vector<pair<int, int>> L;
        stack<Node *> s;
        s.push(r);
        while (!s.empty())
        {
            Node *top = s.top();
            s.pop();

            if (top->parent != NULL)
            {
                /*point to self*/
                L.push_back(pair<int, int>(top->index, top->parent->index));
            }
            else
            {
                /* point to self */
                L.push_back(pair<int, int>(top->index, top->index));
            }

            for (set<Node *>::iterator it = top->children.begin();
                 it != top->children.end();
                 ++it)
            {
                s.push(*it);
            }
        }
        return L;
    }

    vector<int> get_connected_component(int n)
    {
        Node *r = get_root(M[n]);
        vector<int> L;
        stack<Node *> s;
        s.push(r);
        while (!s.empty())
        {
            Node *top = s.top();
            s.pop();
            L.push_back(top->index);
            for (set<Node *>::iterator it = top->children.begin();
                 it != top->children.end();
                 ++it)
            {
                s.push(*it);
            }
        }
        return L;
    }

    //1. fetch the root of n and get r
    // if r is in cc, return
    //else, insert r in to the set
    bool component_seen(int n)
    {
        Node *r = get_root(M[n]);
        if (intersecting_sets.count(r))
        {
            return true;
        }
        intersecting_sets.insert(r);
        return false;
    }

    int GET_ROOT(int idx)
    {
        Node *r = get_root(M[idx]);
        return r->index;
    }

    vector<int> GET_CHILDREN(int idx)
    {
        Node *r = M[idx];
        vector<int> to_ret;
        for (set<Node *>::iterator it = r->children.begin();
             it != r->children.end();
             ++it)
        {
            to_ret.push_back((*it)->index);
        }
        return to_ret;
    }
};

struct NodeBasic
{
    int index;
    int rank;
    NodeBasic *parent;
    NodeBasic(int idx)
    {
        index = idx;
        rank = 0;
        parent = NULL;
    }
};

struct GraphBasic
{
    /*
    Basic disjoint set data structure.    */
    vector<NodeBasic *> M;
    GraphBasic(const int n)
    {
        M.clear();
        for (int i = 0; i < n; ++i)
        {
            M.push_back(new NodeBasic(i));
        }
    }

    NodeBasic *get_root(NodeBasic *node)
    {
        if (!node)
            return NULL;
        if (!node->parent)
            return node;
        node->parent = get_root(node->parent);
        return node->parent;
    }

    void add_edge(const int n1, const int n2)
    {
        NodeBasic *r1 = get_root(M[n1]);
        NodeBasic *r2 = get_root(M[n2]);
        if (!r1 || !r2)
            return;
        if (r1 != r2)
        {
            if (r1->rank > r2->rank)
            {
                r2->parent = r1;
            }
            else
            {
                r1->parent = r2;
                if (r1->rank == r2->rank)
                {
                    r2->rank++;
                }
            }
        }
    }
};

//distance, actually it is distantce^2
double dist(int i, int j, int d, double **dataset)
{
    double sum = 0.;
    for (int m = 0; m < d; ++m)
    {
        sum += (dataset[i][m] - dataset[j][m]) * (dataset[i][m] - dataset[j][m]);
    }
    return sum;
}

// sort by the second value des order
bool cmp2(pair<int, double>a, pair<int, double>b)
{
    return a.second > b.second;
}

bool is_element_in_vector(vector<int> v,int element){
    vector<int>::iterator it;
    it=find(v.begin(),v.end(),element);
    if (it!=v.end()){
        return true;
    }
    else{
        return false;
    }
}

void compute_mutual_knn(int n, int k,
                        int d,
                        double *dataset,
                        double *radii,
                        int *neighbors,
                        double beta,
                        double epsilon,
                        int *result,
                        int *result_parent,
                        double *mode_result
                        )
{
    /* Given the kNN density and neighbors
        We build the k-NN graph / cluster tree and return the estimated modes.
        Note that here, we don't require the dimension of the dataset 
        Returns array of estimated mode membership, where each index cosrresponds
        the respective index in the density array. Points without
        membership are assigned -1 */

    
    vector<pair<double, int>> knn_radii(n); //density array
    vector<set<int>> knn_neighbors(n);  //neighborhood

    vector<int> m_hat(n);              //m_hat
    vector<int> cluster_membership(n); //membership initialization
    vector<int> member_parent(n);      //rank init

    // added the dataset to attain the mode_result
    double **data;
    data = new double *[n];
    for (int i = 0; i < n; ++i)
    {
        data[i] = new double[d];
    }
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < d; ++j)
        {
            data[i][j] = dataset[i * d + j];
        }
    }


    //initialization for density
    for (int i = 0; i < n; ++i)
    {
        knn_radii[i].first = radii[i]; //record the density of the point, actually it is the kth distance
        knn_radii[i].second = i;       //record the id of the point

        for (int j = 0; j < k; ++j)
        {
            knn_neighbors[i].insert(neighbors[i * k + j]);
        }

        m_hat.push_back(0);
        cluster_membership.push_back(0);
        member_parent.push_back(0);
    }

    int n_chosen_points = 0;
    int n_chosen_clusters = 0;
    //1.sort by the density, while second serves as id
    //2. actually, the kth neighbor distance in asc order, so it is density in desc order
    sort(knn_radii.begin(), knn_radii.end());

    Graph G = Graph(); //the G(lambda) graph which is used to store the mutual KNN graph

    int last_considered = 0;
    int last_pruned = 0;

    //1. i iter from highest to the lowest, in terms of the density, in the reverse order of kth neighbor dist
    for (int i = 0; i < n; ++i)
    {
        //2. countinue if (p = last_pruned) p.density is smaller than power(1 + e, 1/d) * i.density
        //notice: pow(1. + epsilon, 1. / d) * knn_radii[i].first, compute the density on the fly
        //this part cost O(n), since both i and p scan the X linearly
        //the knn_radii[i].first works as lambda here, to conduct the G(lambda) as desribed in paper
        while (last_pruned < n && pow(1. + epsilon, 1. / d) * knn_radii[i].first > knn_radii[last_pruned].first)
        {

            //3. add p into graph
            G.add_node(knn_radii[last_pruned].second);

            //4. k (it) is the the KNN of p, so at least, the distance of (k-p) <= the dist between p and the kth neighbor
            //actually this paper add edge , refer to the definition 3
            for (set<int>::iterator it = knn_neighbors[knn_radii[last_pruned].second].begin();
                 it != knn_neighbors[knn_radii[last_pruned].second].end();
                 ++it)
            {
                //5. if k is included by the graph
                if (G.M.count(*it))
                {
                    //6. check the whether the j is also the neighbor of k, if j and k is the mutual knn
                    if (knn_neighbors[*it].count(knn_radii[last_pruned].second))
                    {
                        //7. add the edge between j and k, described as Def.3
                        G.add_edge(knn_radii[last_pruned].second, *it);
                    }
                }
            }
            last_pruned++;
        }

        //8. check the cc (connected component), start with h (last considered)
        while (knn_radii[i].first * pow(1. - beta, 1. / d) > knn_radii[last_considered].first)
        {

            //9. if h not in cc
            if (!G.component_seen(knn_radii[last_considered].second))
            {
                //10. retrieval h's cc
                // vector <int> res = G.get_connected_component(knn_radii[last_considered].second);
                vector<pair<int, int>> res = G.get_connected_component_with_parent(knn_radii[last_considered].second);

                //11. j is the cc of h
                for (size_t j = 0; j < res.size(); j++)
                {
                    //12. dist smaller than lambda, so density is larger than lamda
                    if (radii[res[j].first] <= knn_radii[i].first)
                    {
                        cluster_membership[n_chosen_points] = n_chosen_clusters;
                        member_parent[n_chosen_points] = res[j].second;
                        m_hat[n_chosen_points++] = res[j].first;
                    }
                }
                n_chosen_clusters++;
            }
            last_considered++;
        }
    }

    for (int i = 0; i < n; ++i)
    {
        result[i] = -1;
        result_parent[i] = -1;
        mode_result[i] = -1;
    }

    for (int i = 0; i < n_chosen_points; ++i)
    {
        result[m_hat[i]] = cluster_membership[i];
        result_parent[m_hat[i]] = member_parent[i];
    }

    // cluster_modes: the highest density point in density subgraph
    map<int, pair<int, double>> cluster_modes;

    for (int i = 0; i < n; ++i)
    {
        int cluster_id = result[i];
        // cluster_id = -1 means it's not cc point
        if (cluster_id == -1)
        {
            continue;
        }
        else if (cluster_modes.count(cluster_id) == 0)
        {
            cluster_modes[cluster_id].first = i;
            cluster_modes[cluster_id].second = radii[i];
        }
        else if (radii[i] <= cluster_modes[cluster_id].second)
        {
            cluster_modes[cluster_id].first = i;
            cluster_modes[cluster_id].second = radii[i];
        }
        else
        {
            continue;
        }
    }


    for (int i = 0; i < cluster_modes.size(); ++i)
    {
        int mode_id = cluster_modes[i].first;
        mode_result[mode_id] = 1;
    }
//    for (int i = 0; i < n; ++i)
//    {
//        knn_radii[i].first = knn_radii[i].first * knn_radii[i].first;
//    }
}



void cluster_remaining(
    int n, int k, int d,
    double *dataset,
    double *radii,
    int *neighbors,
    int *initial_memberships,
    int *result)
{

    int **knn_neighbors = new int *[n];
    double **data;
    data = new double *[n];
    for (int i = 0; i < n; ++i)
    {
        data[i] = new double[d];
        knn_neighbors[i] = new int[k];
    }
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            knn_neighbors[i][j] = neighbors[i * k + j];
        }
    }
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < d; ++j)
        {
            data[i][j] = dataset[i * d + j];
        }
    }

    // Final clusters.
    GraphBasic H = GraphBasic(n);

    int n_chosen_clusters = 0;
    for (int i = 0; i < n; ++i)
    {
        if (n_chosen_clusters < initial_memberships[i])
        {
            n_chosen_clusters = initial_memberships[i];
        }
    }
    n_chosen_clusters += 1;
    vector<vector<int>> modal_sets(n_chosen_clusters);
    for (int c = 0; c < n_chosen_clusters; ++c)
    {
        modal_sets.push_back(vector<int>());
    }
    for (int i = 0; i < n; ++i)
    {
        if (initial_memberships[i] >= 0)
        {
            modal_sets[initial_memberships[i]].push_back(i);
        }
    }
    for (int c = 0; c < n_chosen_clusters; ++c)
    {
        for (size_t i = 0; i < modal_sets[c].size() - 1; ++i)
        {
            H.add_edge(modal_sets[c][i], modal_sets[c][i + 1]);
        }
    }
    int next = -1;
    double dt, best_distance = 0.;
    for (int i = 0; i < n; ++i)
    {
        if (initial_memberships[i] >= 0)
        {
            continue;
        }
        next = -1;
        for (int j = 0; j < k; ++j)
        {
            if (radii[knn_neighbors[i][j]] < radii[i])
            {
                next = knn_neighbors[i][j];
                break;
            }
        }

        if (next < 0)
        {
            best_distance = 1000000000.;
            for (int j = 0; j < n; ++j)
            {
                if (radii[j] >= radii[i])
                {
                    continue;
                }
                dt = 0.0;
                for (int m = 0; m < d; ++m)
                {
                    dt += (data[i][m] - data[j][m]) * (data[i][m] - data[j][m]);
                }
                if (best_distance > dt)
                {
                    best_distance = dt;
                    next = j;
                }
            }
        }
        H.add_edge(i, next);
    }
    for (int i = 0; i < n; ++i)
    {
        result[i] = -1;
    }
    int n_clusters = 0;
    map<int, int> label_mapping;
    for (int i = 0; i < n; ++i)
    {
        if (result[i] < 0)
        {
            int label = (H.get_root(H.M[i]))->index;
            if (label_mapping.count(label))
            {
                result[i] = label_mapping[label];
            }
            else
            {
                label_mapping[label] = n_clusters;
                result[i] = n_clusters;
                n_clusters++;
            }
        }
    }
}


void center_remaining(
    int n, int k, int d,
    double *dataset,
    double *radii,
    int *neighbors,
    int *parent,
    double *weight)
{

    int **knn_neighbors = new int *[n];
    double **data;
    data = new double *[n];
    for (int i = 0; i < n; ++i)
    {
        data[i] = new double[d];
        knn_neighbors[i] = new int[k];
    }
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            knn_neighbors[i][j] = neighbors[i * k + j];
        }
    }
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < d; ++j)
        {
            data[i][j] = dataset[i * d + j];
        }
    }

    for (int i = 0; i < n; ++i)
    {
        parent[i] = -1;
    }

    int next = -1;
    double dt, best_distance = 0.;
    for (int i = 0; i < n; ++i)
    {
        next = -1;
        for (int j = 0; j < k; ++j)
        {
            if (radii[knn_neighbors[i][j]] < radii[i])
            {
                next = knn_neighbors[i][j];
                for (int m = 0; m < d; ++m)
                {
                    best_distance = (data[i][m] - data[next][m]) * (data[i][m] - data[next][m]);
                }
                break;
            }
        }

        if (next < 0)
        {
            best_distance = 1000000000.;
            for (int j = 0; j < n; ++j)
            {
                if (radii[j] >= radii[i])
                {
                    continue;
                }
                dt = 0.0;
                for (int m = 0; m < d; ++m)
                {
                    dt += (data[i][m] - data[j][m]) * (data[i][m] - data[j][m]);
                }
                if (best_distance > dt)
                {
                    best_distance = dt;
                    next = j;
                }
            }
        }
        parent[i] = next;
        weight[i] = best_distance;
    }
}
