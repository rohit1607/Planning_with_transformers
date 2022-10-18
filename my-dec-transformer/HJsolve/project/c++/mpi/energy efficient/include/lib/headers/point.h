#ifndef POINT_H
#define POINT_H

#include <iostream>
#include <utility>
#include <math.h>
#include <cmath>
#include <vector>
#include <set>
#include <limits>
#include <list>
using namespace std;

typedef pair<float,float> Point;

/*
struct Point{
    float first;
    float second;
};*/

struct Edge{
    Point one;
    Point two;
};

#endif