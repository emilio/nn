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

extern crate rand;

use rand::Rng;
use std::marker::PhantomData;
use std::{io, fs, path};

trait ActivationFunction {
    fn activation_function(z: f32) -> f32;
}

struct Neuron {
    bias: f32,
    weights: Vec<f32>, // Of length `input_count + 1`, for the bias.
}

impl Neuron {
    fn new<R: Rng>(bias: f32, input_count: usize, rng: &mut R) -> Self {
        let mut weights = Vec::with_capacity(input_count + 1);
        for _ in 0..input_count + 1 {
            weights.push(rng.next_f32());
        }

        Self {
            bias: bias,
            weights: weights,
        }
    }

    fn input_count(&self) -> usize {
        self.weights.len() - 1
    }

    fn weighted_bias(&self) -> f32 {
        self.bias * self.weights[self.weights.len() - 1]
    }

    fn unbiased_sum(&self, inputs: &[f32]) -> f32 {
        assert_eq!(inputs.len(), self.input_count());
        let mut ret = 0.0;
        for (input, weight) in inputs.iter().zip(self.weights.iter()) {
            ret += input * weight;
        }

        ret
    }

    fn biased_sum(&self, inputs: &[f32]) -> f32 {
        self.unbiased_sum(inputs) + self.weighted_bias()
    }

    fn output<A>(&self, inputs: &[f32]) -> f32
        where A: ActivationFunction,
    {
        A::activation_function(self.biased_sum(inputs))
    }
}

struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn new<R: Rng>(neuron_count: usize,
                   input_count: usize,
                   bias: f32,
                   rng: &mut R) -> Self {
        let mut neurons = Vec::with_capacity(neuron_count);
        for _ in 0..neuron_count {
            neurons.push(Neuron::new(bias, input_count, rng))
        }
        Layer {
            neurons: neurons,
        }
    }
}

struct LogisticActivationFunction;

// Standard logistic sigmoid function i.e. k = 1 , x_0 = 0, L = 1
//
// https://en.wikipedia.org/wiki/Logistic_function
impl ActivationFunction for LogisticActivationFunction {
    fn activation_function(z: f32) -> f32 {
        use std::f32;
        1.0 / (1.0 + f32::consts::E.powf(z))
    }
}

struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    fn new<R: Rng>(input_count: usize,
                   output_count: usize,
                   hidden_layer_count: usize,
                   hidden_neuron_count_per_layer: usize,
                   learning_factor: f32,
                   error_boundary: f32,
                   rng: &mut R)
                   -> Self {
        const BIAS: f32 = 1.0;
        Self {
            layers: if hidden_layer_count == 0 {
                vec![
                    Layer::new(output_count, input_count, BIAS, rng)
                ]
            } else {
                let mut layers = Vec::with_capacity(hidden_layer_count + 2);
                layers.push(Layer::new(hidden_neuron_count_per_layer,
                                       input_count,
                                       BIAS,
                                       rng));
                for _ in 0..hidden_layer_count {
                    let hidden_layer =
                        Layer::new(hidden_neuron_count_per_layer,
                                   hidden_neuron_count_per_layer,
                                   BIAS,
                                   rng);
                    layers.push(hidden_layer);
                }

                layers.push(Layer::new(output_count,
                                       hidden_neuron_count_per_layer,
                                       BIAS,
                                       rng));
                layers
            }
        }
    }

    fn feed<A>(&self, mut input: &[f32]) -> Vec<f32>
        where A: ActivationFunction,
    {
        use std::mem;

        let mut layers = self.layers.iter();
        let mut output = vec![];

        // Done this way to avoid tripping the borrow checker while avoiding the
        // extra allocation of `input`.
        if let Some(layer) = layers.next() {
            output = Vec::with_capacity(layer.neurons.len());
            for neuron in &layer.neurons {
                output.push(neuron.output::<A>(input))
            }
        }

        while let Some(layer) = layers.next() {
            let mut this_layer_output = Vec::with_capacity(layer.neurons.len());
            for neuron in &layer.neurons {
                this_layer_output.push(neuron.output::<A>(&output))
            }
            mem::replace(&mut output, this_layer_output);
        }

        output
    }

    fn train<Data>(&mut self, training_data: &[Data])
        where Data: TrainingData,
    {
        for data in training_data {
            self.train_one(data);
        }
    }

    fn train_one<Data>(&mut self, training_data: &Data)
        where Data: TrainingData,
    {
        let input = data.input();
        let output = self.feed(input);
        let expected_output = data.output();

        assert_eq!(output.len(), expected_output.len());
        let mut total_error = Vec::with_capacity(output.len());
        for (actual, expected) in output.iter().zip(expected_output.iter()) {
            error.push(expected - actual);
        }

        // Update the weights in the output layer.
        for (i, neuron) in self.layers.last_mut().unwrap().neurons.iter_mut().enumerate() {
            for (j, weight) in neuron.weights.iter_mut().enumerate() {
                let partial_error = total_error[i] * neuron.last_input[j];
                *weight -= self.learning_rate * partial_error;
            }
        }

        for layer in self.layers.iter_mut().rev().skip(1) {
            let mut partial_error = Vec::with_capacity(layer.neurons.len());

            for neuron in &layer.neurons {
                // We need to calculate the derivative of the error with respect
                // to the output of each hidden layer neuron.
                let mut error_
            }
        }
    }
}


struct MNISTTestImageIterator {
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

impl MNISTTestImageIterator {
    fn new(mnist_path: &path::Path) -> io::Result<Self> {
        use std::io::Read;
        Self {
            images: fs::File::open(mnist_path.join("train-images-idx3-ubyte"))?.bytes(),
            labels: fs::File::open(mnist_path.join("train-labels-idx1-ubyte"))?.bytes(),
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

impl

impl Iterator for MNISTTestImageIterator {
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

        assert!(label >= 0 && label <= 9);
        let mut output = vec![0.0; 10];
        output[label] = 1.0;

        Some(Ok(MNISTLabeledImage {
            label: label,
            expected_output: output,
            pixels: pixels,
            rows: self.rows,
            columns: self.columns,
        }))
    }
}

fn main() {
    // http://yann.lecun.com/exdb/mnist/
    let mnist_path = path::Path::new("./mnist");
    let images = MNISTTestImageIterator::new(&mnist_path).unwrap();
    let mut count = 0;
    for _image in images {
        count += 1;
    }
}
