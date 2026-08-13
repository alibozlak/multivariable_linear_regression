mod without_feature_scaling;

use serde_json::Value;
use without_feature_scaling::N;

/// The training data set, read relative to the project root.
const DATA_SET_PATH: &str = "data/gemini_created_data_set.json";

fn main() {
    let json = std::fs::read_to_string(DATA_SET_PATH).unwrap_or_else(|error| {
        panic!(
            "Data set file could not be read : {DATA_SET_PATH} ({error})\n\
             Hint : the path is relative to the project root, run the program with `cargo run`."
        )
    });

    let gemini_created_data_set: Value = serde_json::from_str(&json)
        .unwrap_or_else(|error| panic!("Data set file is not valid JSON : {error}"));

    let learning_rate_alpha = 0.00003;
    let loop_count_for_train = 1_000_000;

    let mut coefficients_array: [f64; N] = [149.1959, 4.08469];
    let mut constant_coefficient = 1.37315;

    without_feature_scaling::train_data_set(
        &mut coefficients_array,
        &mut constant_coefficient,
        &gemini_created_data_set,
        loop_count_for_train,
        learning_rate_alpha,
    );

    // **** Learning Too Slow !! *****
}
