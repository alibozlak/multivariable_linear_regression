//! Multivariable linear regression trained with plain gradient descent, without
//! any feature scaling. Rust port of the Java implementation living in
//! <https://github.com/alibozlak/multivariable-linear-regression>.

mod data_set;

use std::io::{BufWriter, Write};

/// The model `f(x) = a0*x0 + a1*x1 + ... + a(n-1)*x(n-1) + b`.
///
/// `coefficients_array` holds the `a` coefficients, `constant_coefficient` is `b`.
struct MultivariableLinearRegressionWithoutFeatureScaling {
    coefficients_array: Vec<f64>,
    constant_coefficient: f64,
}

impl MultivariableLinearRegressionWithoutFeatureScaling {
    fn new(coefficients_array: Vec<f64>, constant_coefficient: f64) -> Self {
        Self {
            coefficients_array,
            constant_coefficient,
        }
    }

    /// `n` : how many features one sample has.
    fn n(&self) -> usize {
        self.coefficients_array.len()
    }

    /// The prediction of the model for a single sample.
    fn f(&self, x_vector: &[f64]) -> f64 {
        if self.n() != x_vector.len() {
            panic!("Coefficient array length and Data Set one sample array size must be equal!!");
        }

        let result: f64 = self
            .coefficients_array
            .iter()
            .zip(x_vector)
            .map(|(a, x)| a * x)
            .sum();
        result + self.constant_coefficient
    }

    /// The cost function `J` : mean of the squared errors over the whole data set.
    fn j(&self, real_inputs: &[Vec<f64>], real_outputs: &[f64]) -> f64 {
        let m = self.validate_data_set(real_inputs, real_outputs);

        let mut j = 0.0;
        for i in 0..m {
            j += (self.f(&real_inputs[i]) - real_outputs[i]).powi(2);
        }

        j / m as f64
    }

    /// Partial derivative of `J` with respect to the `j`-th coefficient `aj`.
    fn dj_daj(&self, real_inputs: &[Vec<f64>], real_outputs: &[f64], j: usize) -> f64 {
        let m = self.validate_data_set(real_inputs, real_outputs);

        if j >= self.n() {
            panic!(
                "Invalid j index : j must be in [0,{}] whole number!!",
                self.n() - 1
            );
        }

        let mut result = 0.0;
        for i in 0..m {
            result += real_inputs[i][j] * (self.f(&real_inputs[i]) - real_outputs[i]);
        }

        (2.0 * result) / m as f64
    }

    /// Partial derivative of `J` with respect to the constant coefficient `b`.
    fn dj_db(&self, real_inputs: &[Vec<f64>], real_outputs: &[f64]) -> f64 {
        let m = self.validate_data_set(real_inputs, real_outputs);

        let mut result = 0.0;
        for i in 0..m {
            result += self.f(&real_inputs[i]) - real_outputs[i];
        }

        (2.0 * result) / m as f64
    }

    /// Runs `loop_count_for_train` gradient descent steps with the `alpha` learning
    /// rate, updating every coefficient simultaneously on each step.
    fn train_data_set(
        &mut self,
        real_inputs: &[Vec<f64>],
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

        let mut temp_coefficient_array = vec![0.0; self.n()];
        for i in 0..loop_count_for_train {
            for (j, temp_aj) in temp_coefficient_array.iter_mut().enumerate() {
                *temp_aj =
                    self.coefficients_array[j] - alpha * self.dj_daj(real_inputs, real_outputs, j);
            }

            let temp_b = self.constant_coefficient - alpha * self.dj_db(real_inputs, real_outputs);

            self.coefficients_array
                .copy_from_slice(&temp_coefficient_array);
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

    /// Checks that the data set is consistent with the model and returns `m`, the
    /// sample count of the data set.
    fn validate_data_set(&self, real_inputs: &[Vec<f64>], real_outputs: &[f64]) -> usize {
        if real_inputs[0].len() != self.n() {
            panic!("Real Inputs Array one sample array not correct size (!= n) !!");
        }

        let m = real_inputs.len();
        if m != real_outputs.len() {
            panic!("Real Inputs array and Real Outputs array must be same size!!");
        }

        m
    }
}

fn main() {
    let m2_and_room_count_inputs_data_set = data_set::gemini_created_inputs_data_set();
    let rental_price_outputs = data_set::gemini_created_outputs();

    let learning_rate_alpha = 0.00003;
    let loop_count_for_train = 1_000_000;

    let mut model =
        MultivariableLinearRegressionWithoutFeatureScaling::new(vec![149.1959, 4.08469], 1.37315);

    model.train_data_set(
        &m2_and_room_count_inputs_data_set,
        &rental_price_outputs,
        loop_count_for_train,
        learning_rate_alpha,
    );

    // **** Learning Too Slow !! *****
}
