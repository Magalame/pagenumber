use pagenumber::*;

fn k_graph(n: usize) -> (Vec<Vertex>, Vec<Edge>){

    let vertices = (0..n).map(|x| Vertex(x)).collect();

    let edges = (0..n).flat_map(|i| (0..i+1).filter(move |j| j != &i ).map(move |j| Edge(Vertex(i),Vertex(j)))).collect();

    (vertices, edges)

}

fn main() {

    let (vertices,edges) = k_graph(14);

    // println!("{:?}",edges);

    // let mut vertices = Vec::new();

    // for i in 0..11 {
    //     vertices.push(Vertex(i));
    // }

    // let mut edges = Vec::new();

    // edges.push(Edge(Vertex(0),Vertex(1)));
    // edges.push(Edge(Vertex(3),Vertex(4)));
    // edges.push(Edge(Vertex(4),Vertex(8)));
    // edges.push(Edge(Vertex(6),Vertex(7)));
    // edges.push(Edge(Vertex(6),Vertex(10)));
    // edges.push(Edge(Vertex(1),Vertex(5)));
    // edges.push(Edge(Vertex(0),Vertex(5)));
    // edges.push(Edge(Vertex(6),Vertex(8)));
    // edges.push(Edge(Vertex(4),Vertex(5)));
    // edges.push(Edge(Vertex(1),Vertex(3)));
    // edges.push(Edge(Vertex(9),Vertex(10)));
    // edges.push(Edge(Vertex(7),Vertex(9)));
    // edges.push(Edge(Vertex(7),Vertex(8)));
    // edges.push(Edge(Vertex(3),Vertex(5)));
    // edges.push(Edge(Vertex(5),Vertex(6)));
    // edges.push(Edge(Vertex(0),Vertex(7)));
    // edges.push(Edge(Vertex(8),Vertex(9)));
    // edges.push(Edge(Vertex(4),Vertex(10)));
    // edges.push(Edge(Vertex(5),Vertex(10)));
    // edges.push(Edge(Vertex(4),Vertex(6)));
    // edges.push(Edge(Vertex(1),Vertex(2)));

    let pg = HEA(&vertices, &edges);

    println!("Pg nb:{}",pg);

}
