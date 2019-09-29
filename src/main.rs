mod def;
mod estm;

use def::Oracle;
use def::Model;
use def::Wavefunction;

use def::UVec2;

const MATRIX_WIDTH: usize = 4;
const MATRIX_HEIGHT: usize = 7;

type Matrix = [[char; MATRIX_WIDTH]; MATRIX_HEIGHT];
const INPUT_MATRIX: Matrix = [
    ['L','L','L','L'],
    ['L','L','L','L'],
    ['L','L','L','L'],
    ['L','C','C','L'],
    ['C','S','S','C'],
    ['S','S','S','S'],
    ['S','S','S','S'],
];

const INPUT_MATRIX_2 : Matrix = [
    ['A','A','A','A'],
    ['A','A','A','A'],
    ['A','A','A','A'],
    ['A','C','C','A'],
    ['C','B','B','C'],
    ['C','B','B','C'],
    ['A','C','C','A'],
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
    let size = UVec2(MATRIX_WIDTH, MATRIX_HEIGHT);

    print_matrix(&INPUT_MATRIX);
    println!();
    print_matrix(&INPUT_MATRIX_2);
    println!();
    
    let (compats, weights) = estm::provide(&[INPUT_MATRIX, INPUT_MATRIX_2]);
    // weights.print();
    let oracle = Oracle::new(compats);
    let wavefunction = Wavefunction::new(size, weights);
    let mut model = Model::new(wavefunction, oracle);

    let mut iter_count = 0;
    while model.iterate() {
        println!("Iteration: {}", iter_count);
        model.print();
        iter_count += 1;
        println!();
    }

    // model.run();
    // println!();
    // println!("result:");
    // model.print();
}