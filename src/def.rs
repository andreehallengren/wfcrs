use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;

use crate::Matrix;

use rand::prelude::*;

pub struct Stack<T> {
    inner: Vec<T>,
}

impl<T> Stack<T> {
    pub fn new() -> Stack<T> {
        Stack {
            inner: Vec::new(),
        }
    }

    pub fn push(&mut self, value: T) {
        self.inner.push(value);
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.inner.len() == 0 {
            return None;
        }
        Some(self.inner.swap_remove(self.inner.len() - 1))
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct Vec2(pub isize, pub isize);

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct UVec2(pub usize, pub usize);

pub struct WeightTable<T> {
    inner: HashMap<T, f64>,
}

pub trait Hashable : std::hash::Hash + std::cmp::Eq { }
impl<T: std::hash::Hash + std::cmp::Eq> Hashable for T { }

impl WeightTable<char> {
    pub fn print(&self) {
        println!("{:?}", self.inner);
    }
}

impl<T: Hashable + Copy> WeightTable<T> {
    pub fn new() -> WeightTable<T> {
        WeightTable {
            inner: HashMap::new()
        }
    }

    pub fn normalize(&mut self) {
        let total: f64 = self.inner.values().sum();
        for value in self.inner.values_mut() {
            *value /= total;
        }
    }

    pub fn contains(&self, kind: &T) -> bool {
        self.inner.contains_key(kind)
    }

    pub fn entry(&mut self, kind: T) -> std::collections::hash_map::Entry<T, f64> {
        self.inner.entry(kind)
    }

    pub fn get(&self, kind: &T) -> f64 {
        self.inner[kind]
    }

    pub fn kinds(&self) -> HashSet<T> {
        let mut set = HashSet::new();
        for key in self.inner.keys() {
            set.insert(*key);
        }
        set
    }

    pub fn insert(&mut self, value: T, weight: f64) {
        let current_weight = self.inner.entry(value).or_insert(0f64);
        *current_weight += weight;
    }
}

pub type CompatibilityMap = HashMap<char, HashSet<CompatibleTile>>;
pub type CompatibleTile = (char, Vec2);

pub struct Oracle {
    compatibilites: CompatibilityMap,
}

impl Oracle {
    pub fn new(compatibilites: CompatibilityMap) -> Oracle {
        Oracle {
            compatibilites,
        }
    }

    pub fn check(&self, tile: char, other_tile: char, direction: Vec2) -> bool {
        if self.compatibilites.contains_key(&tile) {
            let tiles = self.compatibilites.get(&tile).unwrap();
            return tiles.contains(&(other_tile, direction));
        }
        false
    }
}

type Coefficients = Vec<Vec<HashSet<char>>>;

pub struct Wavefunction {
    size: UVec2,
    weights: WeightTable<char>,
    coefficients: Vec<Vec<HashSet<char>>>,
}

impl Wavefunction {
    pub fn new(size: UVec2, weights: WeightTable<char>) -> Wavefunction {
        let coefficients = Wavefunction::derive_coefficients(size, weights.kinds());

        Wavefunction {
            size,
            weights,
            coefficients,
        }
    }

    fn derive_coefficients(size: UVec2, kinds: HashSet<char>) -> Coefficients {
        let mut coefficients = vec!(vec!(HashSet::<char>::new(); size.1); size.0);

        for x in 0..size.0 {
            for y in 0..size.1 {
                coefficients[x][y] = kinds.clone();
            }
        }

        coefficients
    }

    pub fn possible_tiles(&self, coords: UVec2) -> &HashSet<char> {
        &self.coefficients[coords.0][coords.1]
    }

    pub fn get_all_potentially_collapsed(&self) -> Matrix {
        let mut matrix: Matrix = [['o'; 4]; 7];

        for x in 0..self.size.0 {
            for y in 0..self.size.1 {
                matrix[y][x] = *self.get_potentially_collapsed(UVec2(x, y));
            }
        }

        matrix
    }

    pub fn get_potentially_collapsed(&self, coords: UVec2) -> &char {
        let tiles = self.possible_tiles(coords);
        if tiles.len() > 1 {
            return &'~';
        }
        else {
            tiles.iter().next().unwrap()
        }
    }

    pub fn shannon_entropy(&self, coords: UVec2) -> f64 {
        let mut sum_of_weights = 0f64;
        let mut sum_of_weights_log_weights = 0f64;

        for option in self.coefficients[coords.0][coords.1].iter() {
            let weight = self.weights.get(option);
            sum_of_weights += weight;
            sum_of_weights_log_weights += weight * weight.ln();
        }

        sum_of_weights.ln() - (sum_of_weights_log_weights / sum_of_weights)
    }

    pub fn is_fully_collapsed(&self) -> bool {
        self.coefficients
            .iter()
            .flat_map(|x| x.iter())
            .filter(|x| x.len() > 1)
            .count() == 0
    }

    // Collapses the wavefunction at the given coordinates
    pub fn collapse(&mut self, coords: UVec2, rng: &mut Box<dyn RngCore>) {
        let options = &self.coefficients[coords.0][coords.1];
        let valid_weights: Vec<(&char, f64)> = options.iter().filter_map(|item| {
            if self.weights.contains(item) {
                Some((item, self.weights.get(item)))
            } else {
                None
            }
        }).collect();

        let total_weights:f64 = valid_weights.iter().map(|i| i.1).sum();

        let mut rnd = total_weights * rng.gen::<f64>();
        let mut chosen = None;
        for (tile, weight) in valid_weights {
            rnd -= weight;
            if rnd < 0f64 {
                chosen = Some(tile);
                break;
            }
        }

        let mut set = HashSet::new();
        set.insert(*chosen.unwrap());
        self.coefficients[coords.0][coords.1] = set;
    }

    // Removed 'tile' from the list of possible tiles at 'coords'
    pub fn ban(&mut self, coords: UVec2, tile: char) {
        self.coefficients[coords.0][coords.1].remove(&tile);
    }
}

pub struct Model {
    size: UVec2,
    wavefunction: Wavefunction,
    stack: Stack<UVec2>,
    rng: Box<dyn RngCore>,
    oracle: Oracle,
}

impl Model {
    pub fn new(wavefunction: Wavefunction, oracle: Oracle) -> Model {
        Model {
            size: wavefunction.size,
            wavefunction,
            stack: Stack::new(),
            rng: Box::new(rand::thread_rng()),
            oracle,
        }
    }

    pub fn run(&mut self) {
        let mut iteration_count = 0;
        while !self.iterate() {
            iteration_count += 1;
        }
        println!("DONE!");
    }

    pub fn iterate(&mut self) -> bool {
        if self.wavefunction.is_fully_collapsed() {
            return false;
        }

        let coords = self.min_entropy_coords().unwrap();
        self.wavefunction.collapse(coords, &mut self.rng);
        self.propagate(coords);
        true
    }

    pub fn print(&self) {
        let result = self.wavefunction.get_all_potentially_collapsed();
        crate::print_matrix(&result);
    }

    fn propagate(&mut self, coords: UVec2) {
        self.stack.push(coords);

        let mut to_ban = VecDeque::with_capacity(4);
        
        while let Some(coords) = self.stack.pop() {
            let current_possible_tiles = self.wavefunction.possible_tiles(coords);

            for direction in valid_dirs(coords, self.size) {
                let other_coords = UVec2(((coords.0 as isize) + direction.0) as usize, ((coords.1 as isize) + direction.1) as usize);

                for other_tile in self.wavefunction.possible_tiles(other_coords) {
                    let other_tile_is_possible = current_possible_tiles
                        .iter()
                        .any(|current_tile| self.oracle.check(*current_tile, *other_tile, direction));

                    if !other_tile_is_possible {
                        to_ban.push_front((other_coords, *other_tile));
                    }
                }
            }

            while let Some((coord, tile)) = to_ban.pop_back() {
                self.wavefunction.ban(coord, tile);
                self.stack.push(coord);
            }
        }
    }

    fn min_entropy_coords(&mut self) -> Option<UVec2> {
        let mut min_entropy = std::f64::MAX;
        let mut min_entropy_coords = None;

        for x in 0..self.size.0 {
            for y in 0..self.size.1 {
                let coords = UVec2(x, y);
                if self.wavefunction.possible_tiles(coords).len() == 1 {
                    continue;
                }

                let entropy = self.wavefunction.shannon_entropy(coords);
                let entropy_plus_noise = entropy - self.rng.gen::<f64>();
                if entropy_plus_noise < min_entropy  {
                    min_entropy = entropy_plus_noise;
                    min_entropy_coords = Some(coords);
                }
            }
        }

        min_entropy_coords
    }
}

const UP: Vec2      = Vec2(0, 1);
const LEFT: Vec2    = Vec2(-1, 0);
const DOWN: Vec2    = Vec2(0, -1);
const RIGHT: Vec2   = Vec2(1, 0);
const DIRS: [Vec2; 4] = [UP, DOWN, LEFT, RIGHT];

pub fn valid_dirs(coords: UVec2, size: UVec2) -> Vec<Vec2> {
    let mut dirs = Vec::new();
    let (x, y) = (coords.0, coords.1);
    let (width, height) = (size.0, size.1);
    
    if x > 0 {
        dirs.push(LEFT);
    }
    if x < width - 1 {
        dirs.push(RIGHT);
    }

    if y > 0 {
        dirs.push(DOWN);
    }
    if y < height - 1 {
        dirs.push(UP);
    }

    dirs
}