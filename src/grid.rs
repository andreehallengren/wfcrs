use petgraph::graph::UnGraph;

pub fn make_grid<T, F: Fn() -> T>(width: usize, height: usize, func: F) -> UnGraph<T, ()> {
    let mut graph = UnGraph::default();
    let mut nodes = Vec::with_capacity(width * height);

    for y in 0..height {
        for x in 0..width {
            let idx = graph.add_node(func());
            nodes.push(idx);
        }
    }

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if x > 0 {
                graph.add_edge(nodes[idx], nodes[idx - 1], ());
            }
            if x < width - 1 {
                graph.add_edge(nodes[idx], nodes[idx + 1], ());
            }

            if y > 0 {
                graph.add_edge(nodes[idx], nodes[idx - width], ());
            }

            if y < height - 1 {
                graph.add_edge(nodes[idx], nodes[idx + width], ());
            }
        }
    }

    return graph;
}
