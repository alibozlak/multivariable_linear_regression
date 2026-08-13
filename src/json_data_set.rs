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

/// Reads the file at `path` and deserializes the data set out of it.
///
/// Which file that is, is the caller's business : this module knows how to turn JSON
/// into a data set, not where the data set of a given program lives.
///
/// # Panics
///
/// Panics if the file cannot be read or if it does not hold valid JSON.
pub fn load(path: impl AsRef<Path>) -> Value {
    let path = path.as_ref();

    let json = std::fs::read_to_string(path).unwrap_or_else(|error| {
        let working_directory = std::env::current_dir()
            .map(|directory| directory.display().to_string())
            .unwrap_or_else(|_| String::from("unknown"));

        panic!(
            "Data set file could not be read : {} ({error})\n\
             Hint : a relative path is resolved from the working directory, which is {}.",
            path.display(),
            working_directory
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
