use std::collections::HashMap;
use std::collections::HashSet;

use crate::def::valid_dirs;
use crate::def::CompatibilityMap;
use crate::def::CompatibleTile;
use crate::def::UVec2;
use crate::def::WeightTable;
use crate::Matrix;

fn parse_matrix(
    matrix: &Matrix,
    weights: &mut WeightTable<char>,
    compatabilities: &mut CompatibilityMap<char>,
) {
    let matrix_height = matrix.len();
    let matrix_width = matrix[0].len();

    for (y, row) in matrix.iter().enumerate() {
        for (x, col) in row.iter().enumerate() {
            let weight = weights.entry(*col).or_insert(0f64);
            *weight += 1f64;

            for direction in valid_dirs(UVec2(x, y), UVec2(matrix_width, matrix_height)) {
                let o_x = ((x as isize) + direction.0) as usize;
                let o_y = ((y as isize) + direction.1) as usize;
                let other_tile = matrix[o_y][o_x];

                let compats = compatabilities
                    .entry(*col)
                    .or_insert(HashSet::<CompatibleTile<char>>::new());

                compats.insert((other_tile, direction));
            }
        }
    }
}

pub fn provide(matrixes: &[Matrix]) -> (CompatibilityMap<char>, WeightTable<char>) {
    let mut weights = WeightTable::new();
    let mut compatabilities = HashMap::new();

    for matrix in matrixes {
        parse_matrix(matrix, &mut weights, &mut compatabilities);
    }

    weights.normalize();

    (compatabilities, weights)
}
