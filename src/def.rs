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
        Stack { inner: Vec::new() }
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

pub trait Hashable: std::hash::Hash + std::cmp::Eq {}
impl<T: std::hash::Hash + std::cmp::Eq> Hashable for T {}

impl WeightTable<char> {
    pub fn print(&self) {
        println!("{:?}", self.inner);
    }
}

impl<T: Hashable + Copy> WeightTable<T> {
    pub fn new() -> WeightTable<T> {
        WeightTable {
            inner: HashMap::new(),
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
}

pub type CompatibilityMap = HashMap<char, HashSet<CompatibleTile>>;
pub type CompatibleTile = (char, Vec2);

pub struct Oracle {
    compatibilites: CompatibilityMap,
}

impl Oracle {
    pub fn new(compatibilites: CompatibilityMap) -> Oracle {
        Oracle { compatibilites }
    }

    pub fn check(&self, tile: char, other_tile: char, direction: Vec2) -> bool {
        if self.compatibilites.contains_key(&tile) {
            let tiles = self.compatibilites.get(&tile).unwrap();
            return tiles.contains(&(other_tile, direction));
        }
        false
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum WavepointState {
    Uncollapsed,
    Collapsed,
}

#[derive(Clone, Debug)]
struct Wavepoint<T: Hashable>
{
    variants: HashSet<T>,
    state: WavepointState,
}

impl<T: Hashable> Wavepoint<T> {
    pub fn new(value: HashSet<T>) -> Wavepoint<T> {
        Wavepoint {
            variants: value,
            state: WavepointState::Uncollapsed,
        }
    }

    pub fn len(&self) -> usize {
        self.variants.len()
    }

    pub fn variants(&self) -> &HashSet<T> {
        &self.variants
    }

    pub fn remove_variant(&mut self, value: &T) {
        assert!(self.variants.len() > 1);
        self.variants.remove(value);
    }

    pub fn collapse(&mut self, value: T) {
        assert_ne!(self.state, WavepointState::Collapsed);
        self.variants.clear();
        self.variants.insert(value);
        self.state = WavepointState::Collapsed;
    }

    pub fn is_collapsed(&self) -> bool {
        match self.state {
            WavepointState::Uncollapsed => false,
            WavepointState::Collapsed => true,
        }
    }

    pub fn get_value(&self) -> &T {
        assert_eq!(self.state, WavepointState::Collapsed);
        self.iter().next().unwrap()
    }

    pub fn iter(&self) -> impl std::iter::Iterator<Item = &'_ T> {
        self.variants.iter()
    }
}

use ndarray::Array;
use ndarray::Array2;
use ndarray::ArrayView2;

pub struct Wavefunction {
    size: UVec2,
    weights: WeightTable<char>,
    coefficients: Array2<Wavepoint<char>>,
}

impl std::fmt::Display for Wavefunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (width, height) = (self.size.0, self.size.1);

        for row in 0..height {
            for col in 0..width {
                let wavepoint = &self.coefficients[[col, row]];
                if wavepoint.is_collapsed() {
                    write!(f, "[{}]", wavepoint.get_value())?;
                } else {
                    write!(f, "[~]")?;
                }
            }
            write!(f, "\n")?;
        }
        std::fmt::Result::Ok(())
    }
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

    fn derive_coefficients(size: UVec2, kinds: HashSet<char>) -> Array2<Wavepoint<char>> {
        Array::from_elem((size.0, size.1), Wavepoint::new(kinds.clone()))
    }

    pub fn possible_tiles(&self, coords: UVec2) -> &HashSet<char> {
        self.coefficients[[coords.0, coords.1]].variants()
    }

    pub fn shannon_entropy(&self, coords: UVec2) -> f64 {
        let mut sum_of_weights = 0f64;
        let mut sum_of_weights_log_weights = 0f64;

        for option in self.coefficients[[coords.0, coords.1]].iter() {
            let weight = self.weights.get(option);
            sum_of_weights += weight;
            sum_of_weights_log_weights += weight * weight.ln();
        }

        sum_of_weights.ln() - (sum_of_weights_log_weights / sum_of_weights)
    }

    pub fn is_fully_collapsed(&self) -> bool {
        self.coefficients
            .iter()
            .filter(|x| !x.is_collapsed())
            .count() == 0
    }

    fn get_collapse_state(&self, coords: UVec2, rng: &mut Box<dyn RngCore>) -> Option<&char> {
        let options = &self.coefficients[[coords.0, coords.1]];
        let valid_weights: Vec<(&char, f64)> = options
            .iter()
            .filter_map(|item| {
                if self.weights.contains(item) {
                    Some((item, self.weights.get(item)))
                } else {
                    None
                }
            })
            .collect();

        let total_weights: f64 = valid_weights.iter().map(|i| i.1).sum();

        let mut rnd = total_weights * rng.gen::<f64>();
        let mut chosen = None;
        for (tile, weight) in valid_weights {
            rnd -= weight;
            if rnd < 0f64 {
                chosen = Some(tile);
                break;
            }
        }

        chosen
    }

    // Collapses the wavefunction at the given coordinates
    pub fn collapse(&mut self, coords: UVec2, rng: &mut Box<dyn RngCore>) {
        let collapsed_state = *self.get_collapse_state(coords, rng).unwrap();
        self.coefficients[[coords.0, coords.1]].collapse(collapsed_state);
    }

    // Removed 'tile' from the list of possible tiles at 'coords'
    pub fn ban(&mut self, coords: UVec2, tile: char) {
        self.coefficients[[coords.0, coords.1]].remove_variant(&tile);
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
        while self.iterate() {
            println!("{}", self.wavefunction);
            iteration_count += 1;
        }
        println!("Done! Generation took {} iterations!", iteration_count);
    }

    pub fn iterate(&mut self) -> bool {
        // for point in self.wavefunction.coefficients.iter() {
        //     println!("{:?}", point)
        // }

        if self.wavefunction.is_fully_collapsed() {
            return false;
        }

        let coords = self.min_entropy_coords();
        self.wavefunction.collapse(coords, &mut self.rng);
        self.propagate(coords);
        true
    }

    pub fn print(&self) {
        println!("{}", self.wavefunction);
    }

    fn propagate(&mut self, coords: UVec2) {
        self.stack.push(coords);

        let mut to_ban = VecDeque::with_capacity(4);

        while let Some(coords) = self.stack.pop() {
            let current_possible_tiles = self.wavefunction.possible_tiles(coords);

            for direction in valid_dirs(coords, self.size) {
                let other_coords = UVec2(
                    ((coords.0 as isize) + direction.0) as usize,
                    ((coords.1 as isize) + direction.1) as usize,
                );

                for other_tile in self.wavefunction.possible_tiles(other_coords) {
                    let other_tile_is_possible =
                        current_possible_tiles.iter().any(|current_tile| {
                            self.oracle.check(*current_tile, *other_tile, direction)
                        });

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

    fn min_entropy_coords(&mut self) -> UVec2 {
        let mut min_entropy = std::f64::MAX;
        let mut min_entropy_coords = None;

        for x in 0..self.size.0 {
            for y in 0..self.size.1 {
                let coords = UVec2(x, y);
                if self.wavefunction.coefficients[[coords.0, coords.1]].is_collapsed() {
                    continue;
                }

                let entropy = self.wavefunction.shannon_entropy(coords);
                let entropy_plus_noise = entropy - self.rng.gen::<f64>();
                if entropy_plus_noise < min_entropy {
                    min_entropy = entropy_plus_noise;
                    min_entropy_coords = Some(coords);
                }
            }
        }

        min_entropy_coords.unwrap()
    }
}

const UP: Vec2 = Vec2(0, 1);
const LEFT: Vec2 = Vec2(-1, 0);
const DOWN: Vec2 = Vec2(0, -1);
const RIGHT: Vec2 = Vec2(1, 0);

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
