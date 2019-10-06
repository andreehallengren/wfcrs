use rand::prelude::*;

mod wfc;
mod estm;

use wfc::UVec2;
use wfc::Model;
use wfc::Wavefunction;

const MATRIX_WIDTH: usize = 4;
const MATRIX_HEIGHT: usize = 7;

type Matrix = [[char; MATRIX_WIDTH]; MATRIX_HEIGHT];

const INPUT_MATRIX_1: Matrix = [
    ['L', 'L', 'L', 'L'],
    ['L', 'L', 'L', 'L'],
    ['L', 'L', 'L', 'L'],
    ['L', 'C', 'C', 'L'],
    ['C', 'S', 'S', 'C'],
    ['S', 'S', 'S', 'S'],
    ['S', 'S', 'S', 'S'],
];

const INPUT_MATRIX_2: Matrix = [
    ['S', 'S', 'S', 'S'],
    ['S', 'S', 'S', 'S'],
    ['S', 'S', 'S', 'S'],
    ['S', 'S', 'S', 'S'],
    ['S', 'S', 'S', 'S'],
    ['S', 'S', 'S', 'S'],
    ['S', 'S', 'S', 'S'],
];

fn main() {
    let output_size = UVec2(25, 25);

    let runs = 1;
    for _ in 0..runs {
        let (compats, weights) = estm::provide(&[INPUT_MATRIX_1, INPUT_MATRIX_2]);
        let wavefunction = Wavefunction::new(output_size, weights);

        let seed = rand::thread_rng().gen::<u64>();
        let mut model = Model::new(134, wavefunction, compats);
    
        let before = std::time::Instant::now();
        // model.run_with_callback(|model, iteration| {
        //     model.print();
        //     std::thread::sleep(std::time::Duration::from_millis(500));
        // });
        model.run();
        let after = std::time::Instant::now();
        let duration = after.duration_since(before);
        // print!("{}", termion::clear::All); 
        println!("result:");
        model.print();
        println!("seed: {}", seed);
        println!("generated in {}ms", duration.as_millis());
    }
}
