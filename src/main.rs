use std::collections::HashMap;
use std::collections::HashSet;

use petgraph::graph::NodeIndex;
use petgraph::graph::UnGraph;

mod graph;
mod grid;
mod util;

use graph::*;

struct HashOracle<T>(HashMap<T, HashSet<T>>);

impl<T: std::hash::Hash + std::cmp::Eq> Oracle<T> for HashOracle<T> {
    fn is_compatible(&self, a: &T, b: &T) -> bool {
        if let Some(counters) = self.0.get(a) {
            if counters.contains(b) {
                return true;
            }
        }
        if let Some(counters) = self.0.get(b) {
            if counters.contains(a) {
                return true;
            }
        }
        false
    }
}

struct GridPrinter(usize, usize);

impl Print<char> for GridPrinter {
    fn print(&self, graph: &UnGraph<Wavepoint<char>, ()>, index: NodeIndex) {
        if index.index() != 0 && index.index() % self.0 == 0 {
            println!();
        }

        match graph[index].get_value() {
            Some(character) => {
                print!("{}", character);
            }
            None => {
                print!("~");
            }
        }
    }
}

fn main() {
    let graph = grid::make_grid(5, 5, || Wavepoint::new(vec!['A', 'B', 'C']));

    let mut compatibilities = HashMap::new();
    let entry = compatibilities.entry('A').or_insert(HashSet::new());
    entry.insert('A');
    entry.insert('B');
    entry.insert('C');

    let entry = compatibilities.entry('B').or_insert(HashSet::new());
    entry.insert('B');
    let entry = compatibilities.entry('C').or_insert(HashSet::new());
    entry.insert('C');

    let oracle = HashOracle(compatibilities);
    let printer = GridPrinter(5, 5);
    let mut model = Model::new(graph, 1, oracle, printer);
    model.run();
}
