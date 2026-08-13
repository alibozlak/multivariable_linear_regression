//! Multivariable linear regression trained with plain gradient descent, without any
//! feature scaling.
//!
//! Every function works directly on the data set as it comes out of its JSON file :
//! an object with an `inputs` array, whose rows carry [`N`] features each, and an
//! `outputs` array carrying the expected value of the row at the same index.
//!
//! The partial derivatives the training uses are written down under `math/`.

use serde_json::Value;
use std::io::{BufWriter, Write};

/// `n` : how many features one sample of the data set has.
pub const N: usize = 2;

/// The prediction of the model for a single sample :
/// `f(x) = a0*x0 + a1*x1 + ... + a(n-1)*x(n-1) + b`.
pub fn f(coefficients_array: &[f64; N], constant_coefficient: f64, x_vector: &Value) -> f64 {
    let mut result = 0.0;
    for (i, a) in coefficients_array.iter().enumerate() {
        result += a * feature(x_vector, i);
    }
    result + constant_coefficient
}

/// The cost function `J` : mean of the squared errors over the whole data set.
pub fn j(coefficients_array: &[f64; N], constant_coefficient: f64, data_set: &Value) -> f64 {
    let (real_inputs, real_outputs) = validate_data_set(data_set);

    let mut j = 0.0;
    for (i, x_vector) in real_inputs.iter().enumerate() {
        j += (f(coefficients_array, constant_coefficient, x_vector) - output(real_outputs, i))
            .powi(2);
    }

    j / real_inputs.len() as f64
}

/// Partial derivative of `J` with respect to the `j`-th coefficient `aj`.
pub fn dj_daj(
    coefficients_array: &[f64; N],
    constant_coefficient: f64,
    data_set: &Value,
    j: usize,
) -> f64 {
    let (real_inputs, real_outputs) = validate_data_set(data_set);

    if j >= N {
        panic!(
            "Invalid j index : j must be in [0,{}] whole number!!",
            N - 1
        );
    }

    let mut result = 0.0;
    for (i, x_vector) in real_inputs.iter().enumerate() {
        result += feature(x_vector, j)
            * (f(coefficients_array, constant_coefficient, x_vector) - output(real_outputs, i));
    }

    (2.0 * result) / real_inputs.len() as f64
}

/// Partial derivative of `J` with respect to the constant coefficient `b`.
pub fn dj_db(coefficients_array: &[f64; N], constant_coefficient: f64, data_set: &Value) -> f64 {
    let (real_inputs, real_outputs) = validate_data_set(data_set);

    let mut result = 0.0;
    for (i, x_vector) in real_inputs.iter().enumerate() {
        result += f(coefficients_array, constant_coefficient, x_vector) - output(real_outputs, i);
    }

    (2.0 * result) / real_inputs.len() as f64
}

/// Runs `loop_count_for_train` gradient descent steps with the `alpha` learning rate,
/// updating every coefficient simultaneously on each step.
pub fn train_data_set(
    coefficients_array: &mut [f64; N],
    constant_coefficient: &mut f64,
    data_set: &Value,
    loop_count_for_train: usize,
    alpha: f64,
) {
    validate_data_set(data_set);

    // One line is printed per loop, so a buffered writer is used instead of locking
    // and flushing stdout a million times.
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    writeln!(
        out,
        "Before train, aArray = {}, b = {}, J = {}",
        to_string_a_array(coefficients_array),
        *constant_coefficient,
        j(coefficients_array, *constant_coefficient, data_set)
    )
    .expect("writing to stdout failed");

    let mut temp_coefficient_array = [0.0; N];
    for i in 0..loop_count_for_train {
        for (j, temp_aj) in temp_coefficient_array.iter_mut().enumerate() {
            *temp_aj = coefficients_array[j]
                - alpha * dj_daj(coefficients_array, *constant_coefficient, data_set, j);
        }

        let temp_b = *constant_coefficient
            - alpha * dj_db(coefficients_array, *constant_coefficient, data_set);

        *coefficients_array = temp_coefficient_array;
        *constant_coefficient = temp_b;

        writeln!(
            out,
            "{}. loop : aArray = {}, b = {}, J = {}",
            i + 1,
            to_string_a_array(coefficients_array),
            *constant_coefficient,
            j(coefficients_array, *constant_coefficient, data_set)
        )
        .expect("writing to stdout failed");
    }

    out.flush().expect("flushing stdout failed");
}

/// The `a` coefficients written the way the Java implementation prints them.
pub fn to_string_a_array(coefficients_array: &[f64; N]) -> String {
    let coefficients: Vec<String> = coefficients_array.iter().map(|a| a.to_string()).collect();
    format!("[{}]", coefficients.join(", "))
}

/// Checks that the data set has the shape the functions expect and hands back its
/// `inputs` and `outputs` arrays, so that the callers do not have to look them up in
/// the JSON object again on every sample.
pub fn validate_data_set(data_set: &Value) -> (&[Value], &[Value]) {
    let real_inputs = data_set["inputs"]
        .as_array()
        .unwrap_or_else(|| panic!("Data Set must have an \"inputs\" array!!"));

    let real_outputs = data_set["outputs"]
        .as_array()
        .unwrap_or_else(|| panic!("Data Set must have an \"outputs\" array!!"));

    if real_inputs.is_empty() {
        panic!("Real Inputs array must not be empty!!");
    }

    if real_inputs.len() != real_outputs.len() {
        panic!("Real Inputs array and Real Outputs array must be same size!!");
    }

    (real_inputs.as_slice(), real_outputs.as_slice())
}

/// The `i`-th feature of one sample of the `inputs` array.
fn feature(x_vector: &Value, i: usize) -> f64 {
    let features = x_vector
        .as_array()
        .unwrap_or_else(|| panic!("Real Inputs Array one sample must be an array!!"));

    if features.len() != N {
        panic!(
            "Real Inputs Array one sample array not correct size (!= {N}) !! : {}",
            features.len()
        );
    }

    features[i]
        .as_f64()
        .unwrap_or_else(|| panic!("Feature {i} of a Real Inputs Array sample is not a number!!"))
}

/// The `i`-th value of the `outputs` array.
fn output(real_outputs: &[Value], i: usize) -> f64 {
    real_outputs[i]
        .as_f64()
        .unwrap_or_else(|| panic!("Value {i} of the Real Outputs array is not a number!!"))
}
