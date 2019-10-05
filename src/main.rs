mod def;
mod estm;

use def::Model;
use def::Oracle;
use def::Wavefunction;

use def::UVec2;

const MATRIX_WIDTH: usize = 4;
const MATRIX_HEIGHT: usize = 7;

type Matrix = [[char; MATRIX_WIDTH]; MATRIX_HEIGHT];
const INPUT_MATRIX: Matrix = [
    ['L', 'L', 'L', 'L'],
    ['L', 'L', 'L', 'L'],
    ['L', 'L', 'L', 'L'],
    ['L', 'C', 'C', 'L'],
    ['C', 'S', 'S', 'C'],
    ['S', 'S', 'S', 'S'],
    ['S', 'S', 'S', 'S'],
];

const INPUT_MATRIX_2: Matrix = [
    ['A', 'A', 'A', 'A'],
    ['A', 'A', 'A', 'A'],
    ['A', 'A', 'A', 'A'],
    ['A', 'C', 'C', 'A'],
    ['C', 'B', 'B', 'C'],
    ['C', 'B', 'B', 'C'],
    ['A', 'C', 'C', 'A'],
];

fn print_matrix(matrix: &Matrix) {
    for y in 0..MATRIX_HEIGHT {
        for x in 0..MATRIX_WIDTH {
            print!("[{}]", matrix[y][x]);
        }
        println!();
    }
}

fn main() {
    let output_size = UVec2(15, 15);

    let (compats, weights) = estm::provide(&[INPUT_MATRIX, INPUT_MATRIX_2]);
    let oracle = Oracle::new(compats);
    let wavefunction = Wavefunction::new(output_size, weights);
    let mut model = Model::new(wavefunction, oracle);

    model.run();
    println!();
    println!("result:");
    model.print();
}
