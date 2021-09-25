
We use MPI15 skeletion:
```
0 : neck,
1 : head,
2 : pelvis,
3 : LShoulder,
4 : LElbow,
5 : LWrist,
6 : LHip,
7 : LKnee,
8 : LAnkle,
9 : RShoulder,
10: RElbow,
11: RWrist,
12: RHip,
13: RKnee,
14: RAnkle 

Pairs:
[[0, 1] 0, [0, 2] 1, [0, 9] 2, [9, 10] 3, [10, 11] 4, [0, 3] 5, [3, 4] 6, [4, 5] 7, [2, 12] 8, [12, 13] 9, [13, 14] 10, [2, 6] 11, [6, 7] 12, [7, 8] 13]
```



Our json is organized as follows:

```
xxx.json
{ "root":
  # image 0  
  [     
    "img_height"   : int,  
    "img_width"    : int,  
    "img_paths"    : "images/path/xxxxxx.jpg",  
    "dataset"      : str,  # dataset name 
    "isValidation" : 0 for train, 1 for test  
    "bodys"        : nested list, N x J x 11
  ],  
  [...],  # image 1 
  [...],  # image 2
}  


For "bodys": 
  N : the number of people in the image,
  J : the number of joints. mpi15 is used,
  11: [x, y, Z, v, X, Y, Z, fx, fy, cx, cy]
    x, y    : 2D keypoints,
    X, Y, Z : 3D keypoints (cm),
    v       : 0: not labeled, 1: occluded, 2: visible,
    fx, fy  : focal length,
    cx, cy  : principal point.

Note that for internet images (unknown intrinsics), fx, fy equal the image width, and cx, cy equal the image center.
```