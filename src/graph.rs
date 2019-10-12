use std::collections::VecDeque;
use std::fmt::Debug;

use petgraph::graph::NodeIndex;
use petgraph::graph::UnGraph;

use rand::prelude::*;
use rand::rngs::SmallRng;

use crate::util::Hashable;
use crate::util::Stack;
use crate::util::WeightTable;

#[derive(Clone, Debug)]
pub struct Wavepoint<T> {
    options: Vec<T>,
    entropy: f64,
    value: Option<T>,
}

impl<T: Hashable + Copy> Wavepoint<T> {
    pub fn new(options: Vec<T>) -> Wavepoint<T> {
        Wavepoint {
            options,
            entropy: 0.5,
            value: None,
        }
    }

    pub fn get_value(&self) -> Option<T> {
        self.value
    }

    fn is_collapsed(&self) -> bool {
        self.value.is_some()
    }

    fn entropy(&mut self, weights: &WeightTable<T>) -> f64 {
        let mut sum_of_weights = 0f64;
        let mut sum_of_weights_log_weights = 0f64;

        for kind in &self.options {
            let weight = weights.get(kind);
            sum_of_weights += weight;
            sum_of_weights_log_weights += weight.ln() * weight;
        }

        assert!(sum_of_weights > 0.0);

        self.entropy = sum_of_weights.ln() - (sum_of_weights_log_weights / sum_of_weights);
        self.entropy
    }

    fn ban(&mut self, kind: &T) {
        if let Some(position) = self.options.iter().position(|x| *x == *kind) {
            self.options.swap_remove(position);
        }
    }

    fn collapse(&mut self, weights: &WeightTable<T>, rng: &mut SmallRng) {
        if self.is_collapsed() {
            panic!();
        }

        let valid_weights = &self
            .options
            .iter()
            .filter_map(|item| {
                if weights.contains(item) {
                    Some((item, weights.get(item)))
                } else {
                    None
                }
            })
            .collect::<Vec<(&T, f64)>>();

        let total_weights: f64 = valid_weights.iter().map(|(_, weight)| weight).sum();

        let mut rnd = total_weights * rng.gen::<f64>();
        let mut chosen = None;
        for (kind, weight) in valid_weights {
            rnd -= weight;
            if rnd < 0f64 {
                chosen = Some(kind);
                break;
            }
        }

        let chosen = **chosen.unwrap();
        self.value = Some(chosen);
        self.options.clear();
        self.options.push(chosen);
    }
}

pub struct Model<T, O, P> where
    O: Oracle<T>,
    P: Print<T>,
{
    graph: UnGraph<Wavepoint<T>, ()>,
    rng: SmallRng,
    weights: WeightTable<T>,
    propagation_stack: Stack<NodeIndex>,
    oracle: O,
    printer: P,
}

impl<T, O, P> Model<T, O, P> where 
    T: Hashable + Copy + Debug,
    O: Oracle<T>,
    P: Print<T>,
{
    pub fn new(
        graph: UnGraph<Wavepoint<T>, ()>,
        seed: u64,
        oracle: O,
        printer: P,
    ) -> Model<T, O, P> {
        let mut weights = WeightTable::new();
        let index = graph.node_indices().nth(0).unwrap();
        for kind in graph[index].options.iter() {
            weights.entry(*kind).or_insert(0.5f64);
        }

        Model {
            graph,
            weights,
            propagation_stack: Stack::new(),
            rng: SmallRng::seed_from_u64(seed),
            oracle,
            printer,
        }
    }

    pub fn run(&mut self) {
        loop {
            if self.is_fully_collapsed() {
                return;
            }

            self.iterate();
            println!();
            println!();
            self.print();
        }
    }

    pub fn iterate(&mut self) {
        let idx = self.min_entropy();
        self.collapse(idx);
        self.propagate(idx);
    }

    pub fn propagate(&mut self, index: NodeIndex) {
        self.propagation_stack.push(index);

        let mut to_ban = VecDeque::new();

        while let Some(index) = self.propagation_stack.pop() {
            let current_options = &self.graph[index].options;
            for neighbour in self.graph.neighbors_undirected(index) {
                for neighbour_kind in &self.graph[neighbour].options {
                    let is_compatible = current_options
                        .iter()
                        .any(|kind| self.check_compatibility(kind, neighbour_kind));

                    if !is_compatible {
                        to_ban.push_back((neighbour, *neighbour_kind));
                    }
                }
            }

            while let Some((index, kind)) = to_ban.pop_back() {
                self.ban(index, kind);
                self.propagation_stack.push(index);
            }
        }
    }

    pub fn is_fully_collapsed(&self) -> bool {
        for index in self.graph.node_indices() {
            let node = &self.graph[index];
            if !node.is_collapsed() {
                return false;
            }
        }

        true
    }

    pub fn collapse(&mut self, index: NodeIndex) {
        self.graph[index].collapse(&self.weights, &mut self.rng);
    }

    pub fn min_entropy(&mut self) -> NodeIndex {
        let mut min_entropy = std::f64::MAX;
        let mut min_entropy_node = None;

        for index in self.graph.node_indices() {
            let node = &mut self.graph[index];

            if node.is_collapsed() {
                continue;
            }

            let entropy = node.entropy(&self.weights);
            let entropy_plus_noise = entropy - self.rng.gen::<f64>();
            if entropy_plus_noise < min_entropy {
                min_entropy = entropy_plus_noise;
                min_entropy_node = Some(index);
            }
        }

        min_entropy_node.unwrap()
    }

    fn ban(&mut self, index: NodeIndex, kind: T) {
        self.graph[index].ban(&kind);
    }

    fn check_compatibility(&self, a: &T, b: &T) -> bool {
        self.oracle.is_compatible(a, b)
    }

    pub fn print(&self) {
        for index in self.graph.node_indices() {
            self.printer.print(&self.graph, index);
        }
    }
}

pub trait Oracle<T> {
    fn is_compatible(&self, a: &T, b: &T) -> bool;
}

pub trait Print<T> {
    fn print(&self, graph: &UnGraph<Wavepoint<T>, ()>, index: NodeIndex);
}
