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
        for i in 0..neuron_count {
            neurons.push(Neuron::new(bias, input_count, rng))
        }
        Layer { neurons }
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

    fn train(&mut self, one_input: &[f32], expected_output: &[f32]) {

    }
}


struct MNISTTestImageIterator {
    images: fs::File,
    labels: fs::File,
}

impl MNISTTestImageIterator {
    fn new(mnist_path: &path::Path) -> io::Result<Self> {
        Ok(Self {
            images: fs::File::open(mnist_path.join("train-images-idx2-ubyte"))?,
            labels: fs::File::open(mnist_path.join("train-labels-idx1-ubyte"))?,
        })
    }
}

fn main() {
    // http://yann.lecun.com/exdb/mnist/
}
