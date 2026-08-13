//! Reading of the training data set from JSON.
//!
//! The expected shape is an object with two arrays :
//!
//! ```json
//! {
//!   "inputs":  [ [55.0, 1.0], [130.0, 4.0] ],
//!   "outputs": [ 23000.0, 48500.0 ]
//! }
//! ```
//!
//! Every row of `inputs` carries the features of one sample, and the value at the
//! same index of `outputs` is what the model is expected to predict for it.

use serde_json::Value;
use std::path::Path;

/// The default place the training data set is read from, relative to the project root.
pub const DEFAULT_PATH: &str = "data/gemini_created_data_set.json";

/// Reads the file at `path` and deserializes the data set out of it.
///
/// # Panics
///
/// Panics if the file cannot be read or if it does not hold valid JSON.
pub fn load(path: impl AsRef<Path>) -> Value {
    let path = path.as_ref();

    let json = std::fs::read_to_string(path).unwrap_or_else(|error| {
        panic!(
            "Data set file could not be read : {} ({error})\n\
             Hint : the path is relative to the project root, run the program with `cargo run`.",
            path.display()
        )
    });

    parse(&json)
}

/// Deserializes the data set out of the JSON text.
///
/// # Panics
///
/// Panics if the text does not hold valid JSON.
pub fn parse(json: &str) -> Value {
    serde_json::from_str(json)
        .unwrap_or_else(|error| panic!("Data set file is not valid JSON : {error}"))
}
