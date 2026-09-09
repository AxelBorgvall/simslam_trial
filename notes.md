# Implementation structure
The structure here has too many classes. Let environment own the gridmap instead of composition (If we write a sim in rust). 

# Map representation

Every particle has its own map. Store it as an "outer map" of grid cells that are alligned between all particles. 

Outer map can be a hashmap or a contiguous array. Slightly less overhead with contiguous array but infinite resizing flexibility with hashmap. I guess. I think use hashmap for low hassle and flexibility. We can swap the hasher from default (SipHash) to rustc-hash (FxHasher), which is built for maxxxed out speed on small keys. With a hashmap the inner entries also dont need to be options since the map knows wether or not they exist.

Let the inner grid cells be Option<Arc<[f32;256]>> (atomic reference counted, refcounted memory that can be safely multithreaded on) When a particles map cells fails out and needs to be replaced only its outer map is updated so that it shares memory with all other particles holding that map cell. Rust has the function Arc::make_mut() that mutates in place if refcount is 1 and copies otherwise. 



