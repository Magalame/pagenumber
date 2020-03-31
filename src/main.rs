use std::collections::{HashSet,HashMap};
use std::cmp::{max};
use std::ops::{Add};
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use std::mem;
/*
todo:
- remove unecessary allocation of vertices and edges for each solution, instead reference to a graph struct
- in *DFS don't use v_read HashSet, instead check if present in labels HashMap directly
*/

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Vertex(usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Position(usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Page(usize);

#[derive(Copy, Clone, Debug, Hash)]
struct Edge(Vertex,Vertex);

// impl Add for Page {
//     type Output = Self;

//     fn add(self, other: Self) -> Self::Output {

//         let Page(p1) = self;
//         let Page(p2) = other;

//         Page(p1+p2) 
//     }
// }

impl PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool {
        (self.0 == other.0 && self.1 == other.1) || (self.0 == other.1 && self.1 == other.0)
    }
}
impl Eq for Edge {}

struct Graph {
    vertices: Vec<Vertex>,
    edges: Vec<Edge>,
}

struct Solution {
    vertices: Vec<Vertex>,
    labels: Option<HashMap<Vertex, Position>>,
    edges: Vec<Edge>,
    pages: Option<HashMap<Edge, Page>>
}

fn empty_sol() -> Solution {
    Solution {
        vertices: Vec::new(),
        labels: None,
        edges: Vec::new(),
        pages: None
    }
}

impl Solution {
    fn pagenumber(&self) -> usize{
        let mut max_pg = 0;
        let pages = self.pages.as_ref().unwrap();
        for (_, Page(page)) in pages.iter() {
            max_pg = max(max_pg, *page);
        }
        max_pg + 1
    }
}


fn is_crossing(u: &Position, v: &Position, p: &Position, q: &Position) -> bool {
    (u < p && p < v && v < q)
    || (u < q && q < v && v < p)
    || (v < p && p < u && u < q)
    || (v < q && q < u && u < p)
    || (u > p && p > v && v > q)
    || (u > q && q > v && v > p)
    || (v > p && p > u && u > q)
    || (v > q && q > u && u > p)

}

fn give_birth(parent: &Solution) -> Solution {

    let original_labels = parent.labels.as_ref().unwrap();

    let mut rng = rand::thread_rng();
    let die = Uniform::from(0..parent.vertices.len());

    //we pick a vertex then all vertices with label less or equal to the vertex's are included as is
    let max_vertex = Vertex(die.sample(&mut rng));
    let max_pos = original_labels.get(&max_vertex).unwrap();

    let vertices = parent.vertices.clone();
    let edges = parent.edges.clone();

    let mut new_labels = HashMap::new();

    for vertex in &vertices {
        let label = original_labels.get(vertex).unwrap();
        if  label <= max_pos {
            new_labels.insert(*vertex, *label);
        }
    }

    let Position(max_pos_index) = *max_pos;

    let mut max_pos_index = max_pos_index;

    RDFS_sub(&edges, &max_vertex, &mut max_pos_index, &mut new_labels);

    for (v,p) in original_labels.iter() {
        if !new_labels.contains_key(v) {
            new_labels.insert(*v, *p);
        }
    } 

    // println!("nb vertices:{}, nb labels:{}", vertices.len(), new_labels.len());

    let child = Solution {
        edges,
        vertices,
        labels: Some(new_labels),
        pages: None
    };

    child

}

fn mutation(solution: &mut Solution){

    let mut rng = rand::thread_rng();
    let die = Uniform::from(0..solution.vertices.len());

    let p1 = Position(die.sample(&mut rng));
    let p2 = Position(die.sample(&mut rng));

    let labels = solution.labels.as_mut().unwrap(); 

    let mut v1 = Vertex(0);
    let mut v2 = Vertex(0);

    for (v,p) in labels.iter(){

        if p == &p1 {
            v1 = *v;
        } else if p == &p2 {
            v2 = *v;
        }

    }

    // we switch dem pages

    let p1_old = labels.get_mut(&v1).unwrap();
    *p1_old = p2;

    let p2_old = labels.get_mut(&v2).unwrap();
    *p2_old = p1;


}

//takes in a solution, and modifies its paging. it modifiesnothing else
fn naive_paging(solution: Solution) -> Solution {

    let mut solution = solution;

    let mut pages = HashMap::new();
    let edges = &solution.edges;
    let labels = solution.labels.as_ref().unwrap();

    let first_edge = edges[0];

    pages.insert(first_edge, Page(0)); // the first edge is necessarily on the first page

    let mut max_page = 0;

    'next_edge: for i in 1..edges.len() {

        let Edge(Vertex(u),Vertex(v)) = edges[i];
        
        'next_page: for cur_page in 0..max_page+1 {

            for j in 0..edges.len() {

                if j != i && pages.contains_key(&edges[j]) && pages.get(&edges[j]).unwrap() == &Page(cur_page) {

                    let Edge(Vertex(p),Vertex(q)) = edges[j];

                    let u_l = labels.get(&Vertex(u)).unwrap();
                    let v_l = labels.get(&Vertex(v)).unwrap();
                    let p_l = labels.get(&Vertex(p)).unwrap();
                    let q_l = labels.get(&Vertex(q)).unwrap();
                    let cross = is_crossing(u_l, v_l, p_l, q_l);

                    if cross {
                        continue 'next_page;
                    }

                }
  
            }

            //if we get there it means that we met not conflict on this page
            pages.insert(edges[i], Page(cur_page));

            continue 'next_edge;

        }

        // if we get there it means that the edge fit in no page
        pages.insert(edges[i], Page(max_page));
        max_page += 1;
        
    }

    solution.pages = Some(pages);

    solution
    

}

//the idea is, take the vertices, create edges like if it were the complete graph, then try to embed them in book
fn EEH(solution: Solution) -> Solution {
    let Vertex(n) = *solution.vertices.iter().max().unwrap();
    let mut s: Vec<Edge> = Vec::new();

    for v in 0..n/2+1 {


        let mut cur_label = v;

        for w in 1..n/2+1 {
            let edge = Edge(Vertex(cur_label), Vertex((v + w).rem_euclid(n)));

            s.push(edge);


            let edge = Edge(Vertex((v + w).rem_euclid(n)), Vertex((n + v - w).rem_euclid(n)));

            s.push(edge);

            cur_label = (n + v - w).rem_euclid(n);
        }

        let edge = Edge(Vertex((n + v + 1 - n/2).rem_euclid(n) ), Vertex((v + n/2).rem_euclid(n) ));

        s.push(edge);
    }

    let mut edges = Vec::new();

    for i in 0..s.len() {

        let Edge(v1,v2) = s[i];

        if solution.edges.contains(&Edge(v1,v2)) && !edges.contains(&Edge(v1,v2)){

            edges.push(Edge(v1,v2));

        } 

    }


    // println!("Edges:{:?}",edges);


    naive_paging(solution)


    
}

//the purpose is to assign labels to vertexes from a RDFS
fn RDFS(vertices: &Vec<Vertex>, edges: &Vec<Edge>) -> HashMap<Vertex, Position> {

    let mut rng = rand::thread_rng();
    let die = Uniform::from(0..vertices.len());
    let rand_index = die.sample(&mut rng); //we get a random index
    // let rand_index = 4;

    let mut labels: HashMap<Vertex, Position> = HashMap::new();
    let cur_pos = &mut 0; // initiate pos at 0

    labels.insert(vertices[rand_index], Position(*cur_pos)); 

    let cur_vertex = &vertices[rand_index];

    RDFS_sub(&edges, cur_vertex, cur_pos, &mut labels);

    labels
    
}

fn RDFS_sub(edges: &Vec<Edge>, cur_vertex: &Vertex, cur_pos:&mut usize, labels: &mut HashMap<Vertex, Position>){


    // println!("vertex:{:?}",cur_vertex);
    
    let mut neighbors = HashSet::new();

    for edge in edges {
        let Edge(u,v) = edge;
        if u == cur_vertex || v == cur_vertex {
               neighbors.insert(edge);
        }
    }

    for Edge(u,v) in neighbors { //since the hashmap has random access, access to neighbors is random, which is what we want

        if u == cur_vertex {
            if !labels.contains_key(v){
                *cur_pos += 1;
                labels.insert(*v, Position(*cur_pos));
                RDFS_sub(edges, v, cur_pos, labels);
            }

        } else if v == cur_vertex {
            if !labels.contains_key(u){
                *cur_pos += 1;
                labels.insert(*u, Position(*cur_pos));
                RDFS_sub(edges, u, cur_pos, labels);
            }
        }
    }
    // println!("end:{:?}",cur_vertex);

}


fn DFS(vertices: &Vec<Vertex>, edges: &Vec<Edge>) -> HashMap<Vertex, Position> {

    let mut labels: HashMap<Vertex, Position> = HashMap::new();
    let cur_pos = &mut 0;

    labels.insert(vertices[0], Position(*cur_pos));

    let mut v_read: HashSet<Vertex> = HashSet::new();

    let cur_vertex = &vertices[0];

    v_read.insert(*cur_vertex);

    DFS_sub(&edges, cur_vertex, cur_pos, &mut labels);

    labels
    
}

fn DFS_sub(edges: &Vec<Edge>, cur_vertex: &Vertex, cur_pos:&mut usize, labels: &mut HashMap<Vertex, Position>){


    for Edge(u,v) in edges { 

        if u == cur_vertex {
            if !labels.contains_key(v){ //if vertex doesn't have a label yet then it means we haven't visited it yet
                *cur_pos += 1;
                labels.insert(*v, Position(*cur_pos));
                RDFS_sub(edges, v, cur_pos, labels);
            }

        } else if v == cur_vertex {
            if !labels.contains_key(u){
                *cur_pos += 1;
                labels.insert(*u, Position(*cur_pos));
                RDFS_sub(edges, u, cur_pos, labels);
            }
        }
    }

}

fn child_with_min_pages(children: &[Solution]) -> usize {
    let mut best_kid = 0;
    for i in 1..children.len() {
        if children[i].pagenumber() < children[best_kid].pagenumber(){
            best_kid = i;
        }
    } 
    best_kid
}

fn best_pg_number(parents: &[Solution]) -> usize {
    parents.iter().map(|p| p.pagenumber()).max().unwrap()
}

fn HEA(vertices: &Vec<Vertex>, edges: &Vec<Edge>) -> usize {
    let pop_size: usize = 10;
    let k: usize = 10;
    let alpha = 0.99;
    let Ti = 1.;
    let Tf = 0.01;

    let rm = 0.2;

    let mut rng = rand::thread_rng();

    let mut t = 0;
    let mut T = Ti;

    let mut ch_num = vec![k;pop_size];

    let mut parents: Vec<Solution> = Vec::with_capacity(pop_size);
    // let mut next_parents: Vec<Solution> = Vec::with_capacity(pop_size);
    // unsafe {next_parents.set_len(pop_size)};
    
    let mut children: Vec<Vec<Solution>> = Vec::with_capacity(pop_size);

    for _ in 0..pop_size {
        children.push(Vec::with_capacity(k));
    }

    // println!("Length of children:{}",children.len());

    for _ in 0..pop_size {
        let labels = RDFS(&vertices, &edges);

        let vertices = vertices.clone();
        let edges = edges.clone();

        let sol = Solution{
            vertices,
            edges,
            labels: Some(labels),
            pages: None
        };

        let sol = EEH(sol);

        parents.push(sol);
    }

    let mut best_pg_nb = best_pg_number(&parents);

    // println!("after init");

    while T > Tf {

        // println!("T:{}",T);

        for parent_index in 0..pop_size {

            // println!("children[parent_index].clear() 1");

            children[parent_index].clear(); // we remove all former kids

            // println!("children[parent_index].clear() 2");

            for child_index in 0..ch_num[parent_index] {

                //apply IDFS to create new child
                let child = give_birth(&parents[parent_index]);

                // println!("Past birth");

                // apply pagination
                let mut child = EEH(child);

                // println!("Past EEH");

                //apply conditional mutation here
                if rng.gen::<f64>() < rm {
                     mutation(&mut child);
                }

                children[parent_index].push(child);

            }
        }

        // println!("chnum");

        update_ch_num(&parents, &children, &mut ch_num, best_pg_nb, T, k);

        // println!("chnum end");

        for parent_index in 0..pop_size {

            if children[parent_index].len() > 0 {

                // println!("parent {}",parent_index);

                let best_kid_index = child_with_min_pages(&children[parent_index]);

                // println!("found best kid: {}", best_kid_index);

                let beta = (parents[parent_index].pagenumber() as f64) - (children[parent_index][best_kid_index].pagenumber() as f64);

                // println!("beta ok");

                if beta > 0. || (beta/T).exp() > rng.gen::<f64>() { // if children better, becomes parent with certain pb
                    // we take child out of vec, and assign it as parent
                    let best_child = mem::replace(&mut children[parent_index][best_kid_index], empty_sol());
                    parents[parent_index] = best_child;
                } else { //otherwise, parent remains parent
                }

            }

            

        }

        

        best_pg_nb = best_pg_number(&parents);

        // println!("Best pg numebr:{}",best_pg_nb);

        T = alpha*T;

        t = t + 1;

        
    }

    

    best_pg_nb

}

fn update_ch_num(parents: &[Solution], children: &Vec<Vec<Solution>>, ch_num: &mut Vec<usize>, best_pg_nb: usize, T: f64, k: usize){
    let mut sum = 0;
    let mut rng = rand::thread_rng();
    let mut count = vec![0;parents.len()];
    
    for i in 0..parents.len() {

        for j in 0..ch_num[i] {
            let beta = best_pg_nb as f64 - children[i][j].pagenumber() as f64;

            if beta > 0. || (beta/T).exp() > rng.gen::<f64>() {
                count[i] = count[i] + 1;
            }
        }

        sum = sum + count[i];
    }

    for i in 0..parents.len() {
        ch_num[i] = (k*parents.len()*count[i])/sum; 
    }

}

fn main() {

    let mut vertices = Vec::new();

    for i in 0..11 {
        vertices.push(Vertex(i));
    }

    let mut edges = Vec::new();

    // edges.push(Edge(Vertex(0),Vertex(1)));
    // edges.push(Edge(Vertex(1),Vertex(5)));
    // edges.push(Edge(Vertex(5),Vertex(4)));
    // edges.push(Edge(Vertex(4),Vertex(3)));
    // edges.push(Edge(Vertex(5),Vertex(6)));
    // edges.push(Edge(Vertex(6),Vertex(10)));
    // edges.push(Edge(Vertex(6),Vertex(7)));
    // edges.push(Edge(Vertex(7),Vertex(8)));
    // edges.push(Edge(Vertex(8),Vertex(9)));
    // edges.push(Edge(Vertex(1),Vertex(2)));

    edges.push(Edge(Vertex(0),Vertex(1)));
    edges.push(Edge(Vertex(3),Vertex(4)));
    edges.push(Edge(Vertex(4),Vertex(8)));
    edges.push(Edge(Vertex(6),Vertex(7)));
    edges.push(Edge(Vertex(6),Vertex(10)));
    edges.push(Edge(Vertex(1),Vertex(5)));
    edges.push(Edge(Vertex(0),Vertex(5)));
    edges.push(Edge(Vertex(6),Vertex(8)));
    edges.push(Edge(Vertex(4),Vertex(5)));
    edges.push(Edge(Vertex(1),Vertex(3)));
    edges.push(Edge(Vertex(9),Vertex(10)));
    edges.push(Edge(Vertex(7),Vertex(9)));
    edges.push(Edge(Vertex(7),Vertex(8)));
    edges.push(Edge(Vertex(3),Vertex(5)));
    edges.push(Edge(Vertex(5),Vertex(6)));
    edges.push(Edge(Vertex(0),Vertex(7)));
    edges.push(Edge(Vertex(8),Vertex(9)));
    edges.push(Edge(Vertex(4),Vertex(10)));
    edges.push(Edge(Vertex(5),Vertex(10)));
    edges.push(Edge(Vertex(4),Vertex(6)));
    edges.push(Edge(Vertex(1),Vertex(2)));

    // let labels = RDFS(&vertices, &edges);

    // for (e,p) in labels.iter() {
    //     println!("\t {:?},\t {:?}",e,p);
    // }

    // let sol = Solution{
    //     vertices,
    //     edges,
    //     labels: Some(labels),
    //     pages: None
    // };

    // let sol = EEH(sol);

    // println!("len:{}",sol.pages.as_ref().unwrap().len());
 
    // for (e,p) in sol.pages.unwrap().iter() {
    //     println!("\t {:?},\t {:?}",e,p);
    // }

    let pg = HEA(&vertices, &edges);

    println!("Pg nb:{}",pg);

}
