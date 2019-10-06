mod def;
mod estm;

use def::Model;
use def::Wavefunction;

use def::UVec2;

const MATRIX_WIDTH: usize = 4;
const MATRIX_HEIGHT: usize = 7;

type Matrix = [[char; MATRIX_WIDTH]; MATRIX_HEIGHT];
const INPUT_MATRIX_0: Matrix = [
    ['A', 'A', 'A', 'A'],
    ['A', 'A', 'A', 'A'],
    ['A', 'A', 'A', 'A'],
    ['B', 'B', 'B', 'A'],
    ['B', 'B', 'B', 'A'],
    ['B', 'B', 'B', 'A'],
    ['B', 'B', 'B', 'A'],
    // ['A', 'A', 'A', 'A'],
];

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

fn main() {
    let output_size = UVec2(25, 25);

    let runs = 100;
    for _ in 0..runs {
        let (compats, weights) = estm::provide(&[INPUT_MATRIX_0]);//, INPUT_MATRIX_2]);
        let wavefunction = Wavefunction::new(output_size, weights);
        let mut model = Model::new(wavefunction, compats);
    
        let before = std::time::Instant::now();
        // model.run_with_callback(|model, iteration| {
        //     print!("{}", termion::clear::All); 
        //     model.print();
        //     // std::thread::sleep(std::time::Duration::from_millis(150));
        // });
        model.run();
        let after = std::time::Instant::now();
        let duration = after.duration_since(before);
        // print!("{}", termion::clear::All); 
        // println!("result:");
        // model.print();
        println!("generated in {}ms", duration.as_millis());
    }
}
