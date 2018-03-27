## Project: 3D Motion Planning
![Quad Image](./misc/enroute.png)

---


# Required Steps for a Passing Submission:
1. Load the 2.5D map in the colliders.csv file describing the environment.
2. Discretize the environment into a grid or graph representation.
3. Define the start and goal locations.
4. Perform a search using A* or other search algorithm.
5. Use a collinearity test or ray tracing method (like Bresenham) to remove unnecessary waypoints.
6. Return waypoints in local ECEF coordinates (format for `self.all_waypoints` is [N, E, altitude, heading], where the drone’s start location corresponds to [0, 0, 0, 0].
7. Write it up.
8. Congratulations!  Your Done!

## [Rubric](https://review.udacity.com/#!/rubrics/1534/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it! Below I describe how I addressed each rubric point and where in my code each point is handled.

### Explain the Starter Code

#### 1. Explain the functionality of what's provided in `motion_planning.py` and `planning_utils.py`

Both scripts contain a basic planning implementation. For one thing, in `backyard_flyer_solution.py` the drone flies from one point to another without considering obstacles. On the other hand, `motion_planning.py` adds one more state which is **PLANNING**. Once the drone is armed and before takeoff it calls `plan_path()` method, here basically the drone computer creates the grid of the map considering the obstacles provided in `colliders.csv`, then defines the start and goal states. To find the path to the goal state the A* algorithm is used, finally the waypoints are created and the drone can takeoff.
`plan_path()` method uses helper functions to create the grid and find the path which are included in `planning_utils.py`. For instance, `create_grid()` creates the grid representation of the map given the obstacles for a certain drone altitude and a safety distance values. `a_star()` looks for a path considering the eucliden distance as a heuristics.

And here's a lovely image after running the initial `motion_planning.py` script.
![Top Down View](./misc/motion_test.png)

### Implementing Your Path Planning Algorithm

#### 1. Set your global home position
To accomplish this I used numpy `genfromtxt` function and passed `max_rows=1` to read the first line as string. In order to get the values and store it as floating points, first I split the string, which gives me the name and value of the latitude and longitude, then I convert to floating point the numerical value.

Here is the code of this part, it also is included in the `motion_planning.py` script from line 128 to 131.
```python
# TODO: read lat0, lon0 from colliders into floating point values
globalhl = np.genfromtxt('colliders.csv', delimiter=',', dtype=None, max_rows=1, encoding='utf-8')
lat0 = float(globalhl[0].strip().split(" ")[1])
lon0 = float(globalhl[1].strip().split(" ")[1])
```


#### 2. Set your current local position
Here as long as you successfully determine your local position relative to global home you'll be all set. Explain briefly how you accomplished this in your code.


Meanwhile, here's a picture of me flying through the trees!
![Forest Flying](./misc/in_the_trees.png)

#### 3. Set grid start position from local position
This is another step in adding flexibility to the start location. As long as it works you're good to go!

#### 4. Set grid goal position from geodetic coords
This step is to add flexibility to the desired goal location. Should be able to choose any (lat, lon) within the map and have it rendered to a goal location on the grid.

#### 5. Modify A* to include diagonal motion (or replace A* altogether)
Minimal requirement here is to modify the code in planning_utils() to update the A* implementation to include diagonal motions on the grid that have a cost of sqrt(2), but more creative solutions are welcome. Explain the code you used to accomplish this step.

#### 6. Cull waypoints 
For this step you can use a collinearity test or ray tracing method like Bresenham. The idea is simply to prune your path of unnecessary waypoints. Explain the code you used to accomplish this step.



### Execute the flight
#### 1. Does it work?
It works!

### Double check that you've met specifications for each of the [rubric](https://review.udacity.com/#!/rubrics/1534/view) points.
  
# Extra Challenges: Real World Planning

For an extra challenge, consider implementing some of the techniques described in the "Real World Planning" lesson. You could try implementing a vehicle model to take dynamic constraints into account, or implement a replanning method to invoke if you get off course or encounter unexpected obstacles.


