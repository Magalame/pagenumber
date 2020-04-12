#![allow(dead_code)]
#![allow(non_snake_case)]
use std::collections::{HashSet};
use std::cmp::{max,min};
use rand::distributions::{Distribution, Uniform};
use rand::{Rng};
use std::mem;


#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Vertex(pub usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Position(usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Page(usize);

#[derive(Copy, Clone, Debug, Hash)]
pub struct Edge(pub Vertex,pub Vertex);

impl PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool {
        (self.0 == other.0 && self.1 == other.1) || (self.0 == other.1 && self.1 == other.0)
    }
}
impl Eq for Edge {}

#[derive(Debug)]
struct Solution<'a> {
    vertices: Option<&'a Vec<Vertex>>,
    labels: Vec<Option<Position>>, 
    edges: Option<&'a Vec<Edge>>,
    pages: Vec<Option<Page>>
}

impl <'a> Solution<'a> {

    fn pagenumber(&self) -> usize{
        let mut max_pg = 0;
        let pages = &self.pages;
        for page in pages.iter() {
            let Page(page) = page.unwrap();
            max_pg = max(max_pg, page);
        }
        max_pg + 1
    }

    fn empty() -> Solution<'a> {
        Solution {
            vertices: None,
            labels: Vec::new(),
            edges: None,
            pages: Vec::new()
        }
    }

    fn thickness_of_last_page(&self) -> usize {
        let last_page = self.pagenumber() -1;
        let thickness = self.pages.iter().filter(|p| p == &&Some(Page(last_page))).count(); 
        thickness

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

fn give_birth<'a>(parent: &Solution<'a>) -> Solution<'a> {

    
    let original_labels = &parent.labels;
    
    let mut rng = rand::thread_rng();
    let die = Uniform::from(0..parent.vertices.unwrap().len());

    //we randomly select a vertex by choosing a random index
    let max_vertex_index = die.sample(&mut rng);
    let max_pos = original_labels[max_vertex_index].unwrap();

    let vertices = parent.vertices.unwrap();
    let edges = parent.edges.unwrap();

    let mut new_labels = vec![None;vertices.len()];

    // all the vertices whose position is less than the one of the randomly selected vertex are directly copied
    for vertex_index in 0..vertices.len() {
        let label = original_labels[vertex_index].unwrap();
        if  label <= max_pos {
            new_labels[vertex_index] = Some(label);
        }
    }

    let Position(max_pos_index) = max_pos;
    let mut max_pos_index = max_pos_index;

    //starting from the randomly selected vertex, we do a new RDFS, to introduce a new label ordering
    RDFS_sub(&edges, &Vertex(max_vertex_index), &mut max_pos_index, &mut new_labels);

    //for all vertices that haven't been assigned a new label, and whose old label doesn't collide with any new label, we just copy their old label
    for i in 0..original_labels.len() {
        if new_labels[i].is_none() {
            if !new_labels.contains(&original_labels[i]) { 
                new_labels[i] = original_labels[i];
            } 
        }
    } 

    //now the vertices who haven't been assigned a new label are the ones whose old label collides with a new one. So we pick up where the RDFS stopped
    //and we look for a "free spot" by incrementing the max_pos_index
    for i in 0..original_labels.len() {
        if new_labels[i].is_none() {
            
            max_pos_index += 1;
            while new_labels.contains(&Some(Position(max_pos_index))){ //increment until there is no collision
                max_pos_index += 1;
            } 

            new_labels[i] = Some(Position(max_pos_index));
                
        }
    }

    let child = Solution {
        edges: Some(edges),
        vertices: Some(vertices),
        labels: new_labels,
        pages: Vec::new()
    };

    child

}

fn mutation(solution: &mut Solution){

    let mut rng = rand::thread_rng();
    let die = Uniform::from(0..solution.vertices.unwrap().len());

    let v1 = die.sample(&mut rng);
    let v2 = die.sample(&mut rng);

    // we switch dem label

    let p1 = solution.labels[v1];
    let p2 = solution.labels[v2];

    solution.labels[v1] = p2;
    solution.labels[v2] = p1;
}

//takes in a solution, and modifies its paging. it modifiesnothing else
fn naive_paging(solution: Solution) -> Solution {

    let mut solution = solution;

    let mut pages: Vec<Option<Page>> = vec![None; solution.edges.unwrap().len()];
    let edges = solution.edges.unwrap();
    let labels = &mut solution.labels;

    pages[0] = Some(Page(0)); // the first edge is necessarily on the first page

    let mut max_page = 0;

    //O(edges^2) runtime, there's definitely a better way to do this
    'next_edge: for i in 1..edges.len() {

        let Edge(Vertex(u),Vertex(v)) = edges[i];
        
        'next_page: for cur_page in 0..max_page+1 {

            for j in 0..edges.len() {

                if j != i && !pages[j].is_none() && pages[j].unwrap() == Page(cur_page) { // we check that the edge edges[j] is on the page of interest

                    let Edge(Vertex(p),Vertex(q)) = edges[j];

                    let u_l = labels[u].unwrap();
                    let v_l = labels[v].unwrap();
                    let p_l = labels[p].unwrap();
                    let q_l = labels[q].unwrap();


                    let cross = is_crossing(&u_l, &v_l, &p_l, &q_l);

                    if cross { //if there is a crossing, skip to the next page
                        continue 'next_page;
                    }

                }
  
            }

            //if we get there it means that we met not conflict on this page
            pages[i] = Some(Page(cur_page));

            continue 'next_edge;

        }
        // if we get there it means that the edge fits into no page
        max_page += 1;
        pages[i] = Some(Page(max_page));
        
        
    }

    solution.pages = pages;

    solution
    

}

//the idea is, take the vertices, create edges like if it were the complete graph, and then output this list
fn EEH(old_edges: &Vec<Edge>, vertices: &Vec<Vertex>) -> Vec<Edge> {
 
    let Vertex(n) = *vertices.iter().max().unwrap();

    let n = n + 1;//if we have vertices from 0 to n, then it is a subset of K(n+1)

    let mut s: Vec<Edge> = Vec::new();

    for v in 0..((n as f64 / 2.0).ceil() as usize)  {

        let mut cur_label = v;

        for w in 1..((n as f64 / 2.0).ceil() as usize) + 1 { // here we actually want to reach ceil(n/2)
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

        if old_edges.contains(&Edge(v1,v2)) && !edges.contains(&Edge(v1,v2)){

            edges.push(Edge(v1,v2));

        } 

    }

    edges


    
}

//the purpose is to assign labels to vertexes from a RDFS
fn RDFS(vertices: &Vec<Vertex>, edges: &Vec<Edge>) -> Vec<Option<Position>> {

    let mut rng = rand::thread_rng();
    let die = Uniform::from(0..vertices.len());
    let mut rand_index = die.sample(&mut rng); //we get a random index

    let mut labels: Vec<Option<Position>> = vec![None;vertices.len()]; //holds the new labels
    let cur_pos = &mut 0; // we initiate the labeling at 0

    while labels.contains(&None){// as long as we havent labeled all the vertices (eg disconnected graph or something)

        while labels[rand_index] != None { // we check that the pivot index is unlabeled, so that we don't risk to relabel it
            rand_index = die.sample(&mut rng);
        }

        labels[rand_index] = Some(Position(*cur_pos)); 
        let cur_vertex = &Vertex(rand_index);

        RDFS_sub(&edges, cur_vertex, cur_pos, &mut labels);

        *cur_pos += 1;

    }

    


    labels
    
}

fn RDFS_sub(edges: &Vec<Edge>, cur_vertex: &Vertex, cur_pos:&mut usize, labels: &mut Vec<Option<Position>>){


    // println!("vertex:{:?}",cur_vertex);
    
    let mut neighbors = HashSet::new();

    for edge in edges { //we put all the edges involving the vertex of interest in neighbors
        let Edge(u,v) = edge;
        if u == cur_vertex || v == cur_vertex {
               neighbors.insert(edge);
        }
    }

    for Edge(u,v) in neighbors { //since the hashmap has random access, access to neighbors is random, which is what we want according the reference paper
        // although it'd be much more efficient to go through them in sequential order, look into rand::seq::IteratorRandom

        let Vertex(v_index) = v;
        let Vertex(u_index) = u;

        if u == cur_vertex {
            if labels[*v_index].is_none() { //if neighbor vertex is not labeled, label it, and run RDFS_sub on it
                *cur_pos += 1;
                labels[*v_index] = Some(Position(*cur_pos));
                RDFS_sub(edges, v, cur_pos, labels);
            }

        } else if v == cur_vertex {
            if labels[*u_index].is_none(){
                *cur_pos += 1;
                labels[*u_index] = Some(Position(*cur_pos));
                RDFS_sub(edges, u, cur_pos, labels);
            }
        }
    }

}



fn child_with_min_pages(children: &[Solution]) -> usize {
    let mut best_kid = 0;
    for i in 1..children.len() {
        if children[i].pagenumber() < children[best_kid].pagenumber() {
            best_kid = i;
        }
    } 
    best_kid
}

fn best_pg_number(parents: &[Solution]) -> usize {
    parents.iter().map(|p| p.pagenumber()).min().unwrap()
}

fn best_parent(parents: &[Solution])  {

    let best_pg = best_pg_number(parents);
    
    print_sol(&parents.iter().filter(|p| p.pagenumber() == best_pg).take(1).next().unwrap());

}

fn print_sol(sol: &Solution) {
    for i in 0..sol.labels.len() {
        println!("v:{:?}, l:{:?}",sol.vertices.unwrap()[i], sol.labels[i].unwrap());
    }
    for i in 0..sol.pages.len() {
        println!("e:{:?}, p:{:?}",sol.edges.unwrap()[i], sol.pages[i].unwrap());
    }
    println!("thickness:{}", sol.thickness_of_last_page());

}

pub fn HEA(vertices: &Vec<Vertex>, edges: &Vec<Edge>) -> usize {
    

    let edges = &EEH(edges, vertices);

    let pop_size = 10;
    let k = 10;
    let alpha = 0.99;
    let Ti = 1.;
    let Tf = 0.01;
    let rm = 0.2;
    let mut t = 0;
    let mut T = Ti;

    let mut rng = rand::thread_rng();

    let mut ch_num = vec![k;pop_size];

    let mut parents: Vec<Solution> = Vec::with_capacity(pop_size);

    let mut children: Vec<Vec<Solution>> = Vec::with_capacity(pop_size);

    for _ in 0..pop_size {
        children.push(Vec::with_capacity(k));
    }

    for _ in 0..pop_size { // we generate the initial population of solution


        let labels = RDFS(vertices, edges);

        let sol = Solution{
            vertices: Some(vertices),
            edges: Some(edges),
            labels: labels,
            pages: Vec::new()
        };

        
        let sol = naive_paging(sol);

        parents.push(sol);
    }

    let mut best_pg_nb = best_pg_number(&parents);

    'all: while T > Tf {


        for parent_index in 0..pop_size {// we generate new kids, and randomly mutate them


            children[parent_index].clear(); // we remove all former kids


            for _ in 0..ch_num[parent_index] {

                let mut child = give_birth(&parents[parent_index]);

                if rng.gen::<f64>() < rm {
                    mutation(&mut child);
                } 

                let child = naive_paging(child);

                children[parent_index].push(child);

            }
        }


        update_ch_num(&parents, &children, &mut ch_num, best_pg_nb, T, k); //we update the number of children for each parent
        

        for parent_index in 0..pop_size {//for each parent we decide which, if any, child should replace them

            if children[parent_index].len() > 0 {


                let best_kid_index = child_with_min_pages(&children[parent_index]);


                let beta = (parents[parent_index].pagenumber() as f64) - (children[parent_index][best_kid_index].pagenumber() as f64);


                if beta > 0. || (beta/T).exp() > rng.gen::<f64>() { // if children is better, becomes parent with certain pb
                    // we take child out of vec, and assign it as parent
                    let best_child = mem::replace(&mut children[parent_index][best_kid_index], Solution::empty());
                    parents[parent_index] = best_child;
                } 

            }

        }

        best_pg_nb = min(best_pg_number(&parents),best_pg_nb);
        T = alpha*T;
        t = t + 1;

        
    }

    best_pg_nb

}

fn update_ch_num(parents: &[Solution], children: &Vec<Vec<Solution>>, ch_num: &mut Vec<usize>, best_pg_nb: usize, T: f64, k: usize){
    let mut sum = 0;
    let mut rng = rand::thread_rng();
    let mut count = vec![0;parents.len()];
    
    for i in 0..parents.len() {//we count the number of good children

        for j in 0..ch_num[i] {
            let beta = best_pg_nb as f64 - children[i][j].pagenumber() as f64;

            if beta > 0. || (beta/T).exp() > rng.gen::<f64>() {
                count[i] = count[i] + 1;
            }
        }

        sum = sum + count[i];
    }

    if sum != 0 {
        for i in 0..parents.len() {
            ch_num[i] = (k*parents.len()*count[i])/sum; 
        }
    } else {
        for i in 0..parents.len() {
            ch_num[i] = 0; 
        }
    }


    

}
