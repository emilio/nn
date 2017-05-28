/*
 * Copyright (C) 2017 Emilio Cobos Álvarez <emilio@crisal.io>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#![feature(link_args)]

extern crate rand;

use rand::Rng;
use std::{io, fs, path, slice};

trait ActivationFunction {
    fn activation_function(z: f32) -> f32;
    fn activation_function_derivative(z: f32) -> f32;
}

#[derive(Debug)]
struct Neuron {
    /// The current weights of the connections arriving to this neuron.
    weights: Vec<f32>,
    /// The weights of the previous round of learning for this neuron.
    previous_weights: Vec<f32>,
    /// The last output of the neuron.
    output: f32,
    /// The sum value of the neuron.
    sum: f32,
    /// The last gradient computing during back-propagation.
    gradient: f32,
}

impl Neuron {
    fn new<R: Rng>(input_count: usize, rng: &mut R) -> Self {
        let mut weights = Vec::with_capacity(input_count);
        for _ in 0..input_count {
            weights.push(rng.next_f32() - 0.5);
        }

        Self {
            weights: weights.clone(),
            previous_weights: weights,
            output: 0.0,
            sum: 0.0,
            gradient: 0.0,
        }
    }

    fn sum<I>(&self, inputs: I) -> f32
        where I: ExactSizeIterator<Item = f32>,
    {
        assert_eq!(inputs.len(), self.weights.len());
        let mut ret = 0.0;

        for (input, weight) in inputs.zip(self.weights.iter()) {
            ret += input * weight;
        }

        ret
    }

    fn activate<A, I>(&mut self, inputs: I)
        where A: ActivationFunction,
              I: ExactSizeIterator<Item = f32>,
    {
        let sum = self.sum(inputs);
        let output = A::activation_function(sum);
        self.sum = sum;
        self.output = output;
    }

}

#[derive(Debug)]
struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn new<R: Rng>(
        neuron_count: usize,
        input_count: usize,
        rng: &mut R)
        -> Self
    {
        let mut neurons = Vec::with_capacity(neuron_count);
        for _ in 0..neuron_count {
            neurons.push(Neuron::new(input_count, rng))
        }
        Layer {
            neurons: neurons,
        }
    }

    fn input_iter<'a>(&'a self) -> LayerIterator<'a> {
        LayerIterator(self.neurons.iter())
    }

    // Feed forward this layer with a given input.
    fn feed<A, I>(&mut self, inputs: I)
        where A: ActivationFunction,
              I: ExactSizeIterator<Item = f32> + Clone,
    {
        for neuron in &mut self.neurons {
            neuron.activate::<A, _>(inputs.clone())
        }
    }
}

#[derive(Clone)]
struct LayerIterator<'a>(slice::Iter<'a, Neuron>);

impl<'a> Iterator for LayerIterator<'a> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|n| n.output)
    }
}

impl<'a> ExactSizeIterator for LayerIterator<'a> {
    fn len(&self) -> usize {
        self.0.len()
    }
}


struct LogisticActivationFunction;

// Standard logistic sigmoid function i.e. k = 1 , x_0 = 0, L = 1
//
// https://en.wikipedia.org/wiki/Logistic_function
impl ActivationFunction for LogisticActivationFunction {
    fn activation_function(z: f32) -> f32 {
        use std::f32;
        1.0 / (1.0 + f32::consts::E.powf(- z))
    }

    fn activation_function_derivative(z: f32) -> f32 {
        let f = Self::activation_function(z);
        f * (1. - f)
    }
}

#[derive(Debug)]
pub struct NeuralNetwork {
    // NB: We only represent in this field the hidden layers plus the output.
    //
    // The "input" layer is made up while forward-propagating.
    layers: Vec<Layer>,
    learning_rate: f32,
    momentum_rate: f32,
}

impl NeuralNetwork {
    fn new<R: Rng>(
        input_count: usize,
        output_count: usize,
        hidden_layer_count: usize,
        hidden_neuron_count_per_layer: usize,
        learning_factor: f32,
        momentum_rate: f32,
        rng: &mut R)
        -> Self
    {
        println!("NeuralNetwork::new({}, {}, {}, {}, {}, {}, <rng>)",
                 input_count, output_count, hidden_layer_count,
                 hidden_neuron_count_per_layer, learning_factor, momentum_rate);

        Self {
            layers: if hidden_layer_count == 0 {
                vec![
                    Layer::new(input_count, input_count, rng),
                    Layer::new(output_count, input_count, rng),
                ]
            } else {
                let mut layers = Vec::with_capacity(hidden_layer_count + 2);
                layers.push(Layer::new(hidden_neuron_count_per_layer,
                                       input_count,
                                       rng));
                for _ in 0..hidden_layer_count {
                    let hidden_layer =
                        Layer::new(hidden_neuron_count_per_layer,
                                   hidden_neuron_count_per_layer,
                                   rng);
                    layers.push(hidden_layer);
                }

                layers.push(Layer::new(output_count,
                                       hidden_neuron_count_per_layer,
                                       rng));
                layers
            },
            learning_rate: learning_factor,
            momentum_rate: momentum_rate,
        }
    }

    // Returns all the outputs of all the neurons.
    //
    // This uses forward-propagation on all the existing layers.
    fn feed<A>(&mut self, input: &[f32])
        where A: ActivationFunction,
    {
        self.layers[0].feed::<A, _>(input.iter().cloned());

        for i in 1..self.layers.len() {
            // self.layers[i].feed::<A, _>(self.layers[i - 1].input_iter());
            let (l, mut r) = self.layers.split_at_mut(i);
            r[0].feed::<A,_>(l.last().unwrap().input_iter());
        }
    }

    fn backpropagate<A>(&mut self, expected_output: &[f32])
        where A: ActivationFunction,
    {
        debug_assert_eq!(self.layers.last().unwrap().neurons.len(), expected_output.len());
        {
            // Grab the output layer gradients.
            let mut output_layer = self.layers.last_mut().unwrap();
            for (neuron, expected_output) in output_layer.neurons.iter_mut().zip(expected_output.iter()) {
                neuron.gradient = *expected_output - neuron.output;
            }
        }

        {
            let mut iter = self.layers.iter_mut().rev();
            let mut next = iter.next();

            // Calculate hidden layer gradients.
            while let Some(next_layer) = next {
                let mut this_layer = match iter.next() {
                    Some(n) => n,
                    None => break,
                };

                for (i, neuron) in this_layer.neurons.iter_mut().enumerate() {
                    let mut pd_error = 0.;

                    for next_layer_neuron in &mut next_layer.neurons {
                        pd_error += next_layer_neuron.gradient * next_layer_neuron.weights[i];
                    }

                    neuron.gradient = pd_error;
                }

                next = Some(this_layer);
            }
        }

        let mut iter = self.layers.iter_mut();
        let mut next = iter.next();
        while let Some(mut previous_layer) = next {
            let mut this_layer = match iter.next() {
                Some(n) => n,
                None => break,
            };

            for neuron in &mut this_layer.neurons {
                let df_input = A::activation_function_derivative(neuron.output);
                let delta = neuron.gradient;
                let previous_weights = neuron.weights.clone();
                for (i, previous_neuron) in previous_layer.neurons.iter_mut().enumerate() {
                    let momentum = self.momentum_rate * (neuron.weights[i] - neuron.previous_weights[i]);
                    neuron.weights[i] +=
                        momentum +
                        self.learning_rate * delta * df_input * previous_neuron.output;
                }
                neuron.previous_weights = previous_weights;
            }

            next = Some(this_layer);
        }
    }

    fn train_one<Data, A>(&mut self, data: &Data)
        where Data: TrainingData,
              A: ActivationFunction,
    {
        self.feed::<A>(data.input());
        self.backpropagate::<A>(data.output());
    }

    fn output_size(&self) -> usize {
        self.layers.last().unwrap().input_iter().len()
    }

    fn last_output(&self) -> usize {
        let mut max = 0.;
        let mut max_index = 0;
        let mut i = 0;

        for v in self.layers.last().unwrap().input_iter() {
            if v > max {
                max_index = i;
                max = v;
            }

            i += 1;
        }

        max_index
    }
}


struct MNISTImageIterator {
    images: io::Bytes<fs::File>,
    labels: io::Bytes<fs::File>,
    count: usize,
    rows: usize,
    columns: usize,
}

fn next_byte(bytes: &mut io::Bytes<fs::File>) -> io::Result<u8> {
    match bytes.next() {
        Some(b) => Ok(b?),
        None => {
            Err(io::Error::new(io::ErrorKind::UnexpectedEof,
                               "Expected one byte at least"))
        }
    }
}

fn read_u32(bytes: &mut io::Bytes<fs::File>) -> io::Result<u32> {
    let first = next_byte(bytes)?;
    let second = next_byte(bytes)?;
    let third = next_byte(bytes)?;
    let fourth = next_byte(bytes)?;

    Ok((first as u32) << 24 |
       (second as u32) << 16 |
       (third as u32) << 8 |
       (fourth as u32))
}

impl MNISTImageIterator {
    fn training(mnist_path: &path::Path) -> io::Result<Self> {
        use std::io::Read;
        Self {
            images: fs::File::open(mnist_path.join("train-images-idx3-ubyte"))?.bytes(),
            labels: fs::File::open(mnist_path.join("train-labels-idx1-ubyte"))?.bytes(),
            count: 0,
            rows: 0,
            columns: 0,
        }.init()
    }

    fn testing(mnist_path: &path::Path) -> io::Result<Self> {
        use std::io::Read;
        Self {
            images: fs::File::open(mnist_path.join("t10k-images-idx3-ubyte"))?.bytes(),
            labels: fs::File::open(mnist_path.join("t10k-labels-idx1-ubyte"))?.bytes(),
            count: 0,
            rows: 0,
            columns: 0,
        }.init()
    }

    fn init(mut self) -> io::Result<Self> {
        // TODO(emilio): Return errors instead of asserting.
        let magic = read_u32(&mut self.labels)?;
        assert_eq!(magic, 2049);

        let magic = read_u32(&mut self.images)?;
        assert_eq!(magic, 2051);

        let label_count = read_u32(&mut self.labels)?;
        let image_count = read_u32(&mut self.images)?;

        assert_eq!(label_count, image_count);
        self.count = label_count as usize;
        self.rows = read_u32(&mut self.images)? as usize;
        self.columns = read_u32(&mut self.images)? as usize;

        Ok(self)
    }
}

trait TrainingData {
    fn input(&self) -> &[f32];
    fn output(&self) -> &[f32];
}

struct SimpleTrainingData<'a> {
    input: &'a [f32],
    output: &'a [f32],
}

impl<'a> TrainingData for SimpleTrainingData<'a> {
    fn input(&self) -> &[f32] {
        self.input
    }

    fn output(&self) -> &[f32] {
        self.output
    }
}

#[derive(Debug)]
struct MNISTLabeledImage {
    pixels: Vec<f32>, // Bytes normalized from 0..255 to 0.0..1.0
    expected_output: Vec<f32>,
    label: u8,
    rows: usize,
    columns: usize,
}

impl TrainingData for MNISTLabeledImage {
    fn input(&self) -> &[f32] {
        &self.pixels
    }
    fn output(&self) -> &[f32] {
        &self.expected_output
    }
}

impl Iterator for MNISTImageIterator {
    type Item = io::Result<MNISTLabeledImage>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count == 0 {
            return None;
        }
        self.count -= 1;

        let label = match self.labels.next() {
            Some(Ok(l)) => l,
            Some(Err(e)) => {
                self.count = 0;
                return Some(Err(e));
            }
            None => {
                self.count = 0;
                return None;
            }
        };

        let mut pixels = Vec::with_capacity(self.rows * self.columns);
        for _ in 0..self.rows {
            for _ in 0..self.columns {
                let pixel = match self.images.next() {
                    Some(Ok(l)) => l,
                    Some(Err(e)) => {
                        self.count = 0;
                        return Some(Err(e));
                    }
                    None => {
                        self.count = 0;
                        return None;
                    }
                };

                pixels.push(pixel as f32 / 255.0)
            }
        }

        assert!(label <= 9);
        let mut output = vec![0.0; 10];
        output[label as usize] = 1.0;

        Some(Ok(MNISTLabeledImage {
            label: label,
            expected_output: output,
            pixels: pixels,
            rows: self.rows,
            columns: self.columns,
        }))
    }
}

#[no_mangle]
pub unsafe extern "C" fn neural_network_destroy(nn: *mut NeuralNetwork) {
    if nn.is_null() {
        return;
    }

    let _ = Box::from_raw(nn);
}

#[no_mangle]
pub unsafe extern "C" fn neural_network_create(
    input_count: usize,
    output_count: usize,
    hidden_layer_count: usize,
    hidden_neuron_count_per_layer: usize,
    learning_factor: f32,
    momentum_rate: f32)
    -> *mut NeuralNetwork
{
    use std::ptr;

    let mut rng = match rand::OsRng::new() {
        Ok(rng) => rng,
        Err(..) => return ptr::null_mut(),
    };

    let nn = Box::new(NeuralNetwork::new(
        input_count,
        output_count,
        hidden_layer_count,
        hidden_neuron_count_per_layer,
        learning_factor,
        momentum_rate,
        &mut rng
    ));

    Box::into_raw(nn)
}

#[no_mangle]
pub unsafe extern "C" fn neural_network_feed(
    nn: *mut NeuralNetwork,
    input: *const f32,
    len: usize)
    -> isize
{
    use std::slice;

    if nn.is_null() {
        return -1;
    }

    let slice = slice::from_raw_parts(input, len);
    (*nn).feed::<LogisticActivationFunction>(slice);

    (*nn).last_output() as isize
}

#[no_mangle]
pub unsafe extern "C" fn neural_network_train_one(
    nn: *mut NeuralNetwork,
    label: usize,
    input: *const f32,
    len: usize)
    -> isize
{
    use std::slice;

    if nn.is_null() {
        return -1;
    }

    let output_len = (*nn).output_size();
    if label >= output_len {
        return -2;
    }

    let slice = slice::from_raw_parts(input, len);
    let mut output = vec![0.0; output_len];
    output[label] = 1.0;

    let data = SimpleTrainingData {
        input: slice,
        output: &output,
    };

    (*nn).train_one::<_, LogisticActivationFunction>(&data);
    (*nn).last_output() as isize
}

#[cfg(target_os = "emscripten")]
#[link_args = "-s ALLOW_MEMORY_GROWTH=20 -s NO_EXIT_RUNTIME=1"]
extern {}

#[cfg(target_os = "emscripten")]
fn main() { /* Intentionally empty */ }

#[cfg(not(target_os = "emscripten"))]
fn main() {
    const INPUT_COUNT: usize = 28 * 28;
    const OUTPUT_COUNT: usize = 10;
    const HIDDEN_LAYERS: usize = 1;
    const NEURONS_PER_HIDDEN_LAYER: usize = INPUT_COUNT;
    const LEARNING_FACTOR: f32 = 0.3;
    const MOMENTUM_RATE: f32 = 0.03;

    // http://yann.lecun.com/exdb/mnist/
    let mnist_path = path::Path::new("./mnist");
    let mut rng = rand::OsRng::new().unwrap();

    let mut network = NeuralNetwork::new(
        INPUT_COUNT,
        OUTPUT_COUNT,
        HIDDEN_LAYERS,
        NEURONS_PER_HIDDEN_LAYER,
        LEARNING_FACTOR,
        MOMENTUM_RATE,
        &mut rng
    );

    {
        let mut count = 1;
        let training_images = MNISTImageIterator::training(&mnist_path).unwrap();
        let total = training_images.count;
        for image in training_images {
            let image = image.unwrap();
            network.train_one::<_, LogisticActivationFunction>(&image);
            // let output: Vec<_> = network.layers.last().unwrap().input_iter().collect();

            if count % 100 == 0 {
                println!("Trained...{} / {}", count, total);
                // println!("{:?}", output);
                // println!("{:?}", image.output());
                // ::std::thread::sleep_ms(50);
            }
            count += 1;
        }
    }

    {
        let mut count = 1;
        let mut hits = 0;
        let test_images = MNISTImageIterator::testing(&mnist_path).unwrap();
        let total = test_images.count;
        for image in test_images.take(5000) {
            let image = image.unwrap();

            network.feed::<LogisticActivationFunction>(image.input());

            let index = network.last_output();

            let expected = image.output().iter().position(|v| *v > 0.).unwrap();
            if expected == index {
                hits += 1;
            }

            let ratio = hits as f32 / count as f32;

            if count % 1000 == 0 {
                println!("{} / {} ({}, {} -> {}): {}%",
                         count, total, expected, index,
                         expected == index, ratio * 100.);
            }

            // println!("{:?}", output);
            // println!("{:?}", image.output());
            // ::std::thread::sleep_ms(50);
            count += 1;
        }
    }
}
