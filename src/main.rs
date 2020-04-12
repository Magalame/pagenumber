use pagenumber::*;

fn k_graph(n: usize) -> (Vec<Vertex>, Vec<Edge>){

    let vertices = (0..n).map(|x| Vertex(x)).collect();

    let edges = (0..n).flat_map(|i| (0..i+1).filter(move |j| j != &i ).map(move |j| Edge(Vertex(i),Vertex(j)))).collect();

    (vertices, edges)

}

fn main() {

    let (vertices,edges) = k_graph(14);

    let pg = HEA(&vertices, &edges);

    println!("Pg nb:{}",pg);

}
