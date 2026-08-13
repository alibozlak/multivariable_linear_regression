//! Multivariable linear regression trained with plain gradient descent, without any
//! feature scaling.
//!
//! The partial derivatives the training uses are written down under `math/`.

use std::io::{BufWriter, Write};

/// `n` : how many features one sample of the data set has.
pub const N: usize = 2;

/// The model `f(x) = a0*x0 + a1*x1 + ... + a(n-1)*x(n-1) + b`.
pub struct MultivariableLinearRegression {
    /// The `a` coefficients, one per feature.
    coefficients_array: [f64; N],
    /// The constant coefficient `b`.
    constant_coefficient: f64,
}

impl MultivariableLinearRegression {
    pub fn new(coefficients_array: [f64; N], constant_coefficient: f64) -> Self {
        Self {
            coefficients_array,
            constant_coefficient,
        }
    }

    /// The prediction of the model for a single sample.
    pub fn f(&self, x_vector: &[f64; N]) -> f64 {
        let result: f64 = self
            .coefficients_array
            .iter()
            .zip(x_vector)
            .map(|(a, x)| a * x)
            .sum();
        result + self.constant_coefficient
    }

    /// The cost function `J` : mean of the squared errors over the whole data set.
    pub fn j(&self, real_inputs: &[[f64; N]], real_outputs: &[f64]) -> f64 {
        let m = self.validate_data_set(real_inputs, real_outputs);

        let mut j = 0.0;
        for i in 0..m {
            j += (self.f(&real_inputs[i]) - real_outputs[i]).powi(2);
        }

        j / m as f64
    }

    /// Partial derivative of `J` with respect to the `j`-th coefficient `aj`.
    pub fn dj_daj(&self, real_inputs: &[[f64; N]], real_outputs: &[f64], j: usize) -> f64 {
        let m = self.validate_data_set(real_inputs, real_outputs);

        if j >= N {
            panic!(
                "Invalid j index : j must be in [0,{}] whole number!!",
                N - 1
            );
        }

        let mut result = 0.0;
        for i in 0..m {
            result += real_inputs[i][j] * (self.f(&real_inputs[i]) - real_outputs[i]);
        }

        (2.0 * result) / m as f64
    }

    /// Partial derivative of `J` with respect to the constant coefficient `b`.
    pub fn dj_db(&self, real_inputs: &[[f64; N]], real_outputs: &[f64]) -> f64 {
        let m = self.validate_data_set(real_inputs, real_outputs);

        let mut result = 0.0;
        for i in 0..m {
            result += self.f(&real_inputs[i]) - real_outputs[i];
        }

        (2.0 * result) / m as f64
    }

    /// Runs `loop_count_for_train` gradient descent steps with the `alpha` learning
    /// rate, updating every coefficient simultaneously on each step.
    pub fn train_data_set(
        &mut self,
        real_inputs: &[[f64; N]],
        real_outputs: &[f64],
        loop_count_for_train: usize,
        alpha: f64,
    ) {
        self.validate_data_set(real_inputs, real_outputs);

        // One line is printed per loop, so a buffered writer is used instead of
        // locking and flushing stdout a million times.
        let stdout = std::io::stdout();
        let mut out = BufWriter::new(stdout.lock());

        writeln!(
            out,
            "Before train, aArray = {}, b = {}, J = {}",
            self.to_string_a_array(),
            self.constant_coefficient,
            self.j(real_inputs, real_outputs)
        )
        .expect("writing to stdout failed");

        let mut temp_coefficient_array = [0.0; N];
        for i in 0..loop_count_for_train {
            for (j, temp_aj) in temp_coefficient_array.iter_mut().enumerate() {
                *temp_aj =
                    self.coefficients_array[j] - alpha * self.dj_daj(real_inputs, real_outputs, j);
            }

            let temp_b = self.constant_coefficient - alpha * self.dj_db(real_inputs, real_outputs);

            self.coefficients_array = temp_coefficient_array;
            self.constant_coefficient = temp_b;

            writeln!(
                out,
                "{}. loop : aArray = {}, b = {}, J = {}",
                i + 1,
                self.to_string_a_array(),
                self.constant_coefficient,
                self.j(real_inputs, real_outputs)
            )
            .expect("writing to stdout failed");
        }

        out.flush().expect("flushing stdout failed");
    }

    fn to_string_a_array(&self) -> String {
        let coefficients: Vec<String> = self
            .coefficients_array
            .iter()
            .map(|a| a.to_string())
            .collect();
        format!("[{}]", coefficients.join(", "))
    }

    /// Checks that the data set is usable and returns `m`, its sample count.
    ///
    /// That every sample carries exactly [`N`] features is already guaranteed by the
    /// `[f64; N]` type, so only the emptiness and the input / output count have to be
    /// checked here.
    fn validate_data_set(&self, real_inputs: &[[f64; N]], real_outputs: &[f64]) -> usize {
        let m = real_inputs.len();

        if m == 0 {
            panic!("Real Inputs array must not be empty!!");
        }

        if m != real_outputs.len() {
            panic!("Real Inputs array and Real Outputs array must be same size!!");
        }

        m
    }
}
