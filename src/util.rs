use std::collections::HashMap;
use std::hash::Hash;

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

pub struct WeightTable<T> {
    inner: HashMap<T, f64>,
}

pub trait Hashable: Hash + Eq {}
impl<T: Hash + Eq> Hashable for T {}

impl<T: Hashable> WeightTable<T> {
    pub fn new() -> WeightTable<T> {
        WeightTable {
            inner: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
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

    pub fn kinds(&self) -> impl Iterator<Item = &T> {
        self.inner.keys()
    }
}
