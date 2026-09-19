# Main algorithm

## Data
- mapquery 
```
{
	chunk_coord:(i32,i32),
	localx:u16,
	localy:u16,
	prop_id:u32, // flattened proposition id
}
```

- Particle poses: x,y,theta tuples for all particles
- Map: Every particle holds a hashmap from outer indices (i32,i32) to a nxn patch of map data. The patch is represented as Arc<[f32;256] by keeping them as refcounted objects they can be updated lazily via Rc::make_mut() only when particles no longer share that map segment. This evens out expensive mem operations. Use a fast hasher like rustc-hahs for the hashmap.
- pointbuffers: A buffer of the projected lidar points for all the different scanmatch samples every element will be a MapQuery[Nray,(searchRadius*2+1)^3] for each particle
- logprob_buffers: a buffer of all the running lop probs for the different sampled particles [Npart,(searchRadius*2+1)^3] give each particle each own logprob buffer
- 
## Input
odometry+scan pair odometry is change in pose and scan an array of lidar distances

## Kinematics
- for all particles modify them by the odometry with an added noise model do drive them in different ways. Store in priors
## Scan match
The scan match scales as particles x rays x scanmatch_rad^3 x searchspace^2 and all of it is trivial operations, so completely memory limited. Computing distances by chunk rather than by particle sample will be critical. 

- compute all projected points into the particlewise mapquery buffers (wrap pi for the pose samples)
- ```sort_usntable_by_key(|q| q.chunk_coord)``` unstable sort over chunk coords
- Do: 
```
for chunk_group in queries.chunk_by(|a, b| a.chunk_coord == b.chunk_coord) {
    
    let current_chunk_coord = chunk_group[0].chunk_coord;
    let chunk_data = map.get_chunk(current_chunk_coord);
    
    for q in chunk_group {
        let dist = chunk_data.get_distance(q.local_x, q.local_y);
        let likelihood = compute_likelihood(dist);
        log_probs[q.prop_id as usize] += likelihood;
    }
}
```
- weigths and covariance are computed from the lp so a running tally instead of raywise distances is fine

## state update
- pull new poses from the computed distributions (make sure to wrap pi!)
- update weights
- update each particles map via bresenham. 

# Implementation structure
The structure here has too many classes. Let environment own the gridmap instead of composition. 

One struct object for all the simulation/reading of data nad one for the slam implementation, both implemented as traits

slam trait:
- get control/sim
- grab scan+action
- update state
- render?

Robot trait:
- 



# Map representation

Every particle has its own map. Store it as an "outer map" of grid cells that are alligned between all particles. 

Outer map can be a hashmap or a contiguous array. Slightly less overhead with contiguous array but infinite resizing flexibility with hashmap. I guess. I think use hashmap for low hassle and flexibility. We can swap the hasher from default (SipHash) to rustc-hash (FxHasher), which is built for maxxxed out speed on small keys. With a hashmap the inner entries also dont need to be options since the map knows wether or not they exist.

Let the inner grid cells be Option<Arc<[f32;256]>> (atomic reference counted, refcounted memory that can be safely multithreaded on) When a particles map cells fails out and needs to be replaced only its outer map is updated so that it shares memory with all other particles holding that map cell. Rust has the function Arc::make_mut() that mutates in place if refcount is 1 and copies otherwise. 



