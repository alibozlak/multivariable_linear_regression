//! Conversion between JSON payloads and the vector shapes the model works with.
//!
//! A training set is expected in this shape, where every entry of `inputs` is
//! one sample and the entry of `outputs` at the same index is that sample's
//! expected value:
//!
//! ```json
//! {
//!   "inputs":  [[55.0, 1.0], [130.0, 4.0]],
//!   "outputs": [23000.0, 48500.0]
//! }
//! ```
//!
//! and the result of a training run is written back as:
//!
//! ```json
//! { "last_coefficients": [375.13, -195.00, 1807.28] }
//! ```

use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

/// A training set in the JSON shape described at the module level.
///
/// Exposed for callers that want the parsed struct itself; the usual entry
/// point is [`training_data_from_json`], which hands back the two plain vectors
/// the model's constructor takes.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrainingData {
    /// One entry per sample, each holding that sample's `n` feature values.
    pub inputs : Vec<Vec<f64>>,
    /// The expected output of the sample at the same index in [`Self::inputs`].
    pub outputs : Vec<f64>
}

/// The outcome of a training run, serialised as
/// `{"last_coefficients": [a_1, ..., a_n, b]}`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrainingResult {
    /// The coefficients `train_model` returned, the bias `b` in the last slot.
    pub last_coefficients : Vec<f64>
}

/// Everything that can go wrong while converting between JSON and the model's
/// vectors.
///
/// The variants other than [`Self::Json`] all describe a payload that parses as
/// JSON but does not describe a usable training set. Catching them here means
/// the model's own constructor is never reached with data that would make it
/// panic — which matters when the JSON arrives from outside, e.g. as a request
/// body.
#[derive(Debug)]
pub enum JsonConverterError {
    /// The text was not valid JSON, or did not match the expected shape.
    Json(serde_json::Error),
    /// `inputs` was empty: there is nothing to train on, and no feature count
    /// can be derived.
    EmptyDataSet,
    /// There are not as many outputs as there are input samples.
    SampleCountMismatch { inputs : usize, outputs : usize },
    /// One sample carries a different number of features than the first one.
    RaggedSample { index : usize, expected : usize, found : usize }
}

impl fmt::Display for JsonConverterError {
    fn fmt(&self, f : &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(error) => write!(f, "invalid JSON payload: {error}"),
            Self::EmptyDataSet => write!(f, "\"inputs\" is empty, there is nothing to train on"),
            Self::SampleCountMismatch { inputs, outputs } => write!(
                f,
                "sample count mismatch: {inputs} input sample(s) but {outputs} output(s)"
            ),
            Self::RaggedSample { index, expected, found } => write!(
                f,
                "sample {index} has {found} feature(s) while the first sample has {expected}"
            )
        }
    }
}

impl Error for JsonConverterError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            _ => None
        }
    }
}

impl From<serde_json::Error> for JsonConverterError {
    fn from(error : serde_json::Error) -> Self {
        Self::Json(error)
    }
}

/// Parses a training set and hands back the `(inputs, outputs)` pair that
/// `WithoutFeatureScaling::new` takes.
///
/// The data is validated first, so a successful return guarantees a non-empty
/// set, one output per sample, and the same feature count on every sample.
///
/// ```
/// use multivariable_linear_regression::json_converter;
///
/// let json = r#"{"inputs": [[55.0, 1.0], [130.0, 4.0]], "outputs": [23000.0, 48500.0]}"#;
/// let (inputs, outputs) = json_converter::training_data_from_json(json).unwrap();
///
/// assert_eq!(inputs, vec![vec![55.0, 1.0], vec![130.0, 4.0]]);
/// assert_eq!(outputs, vec![23000.0, 48500.0]);
/// ```
pub fn training_data_from_json(
    json : &str
) -> Result<(Vec<Vec<f64>>, Vec<f64>), JsonConverterError> {
    let data : TrainingData = serde_json::from_str(json)?;
    validate(&data)?;

    Ok((data.inputs, data.outputs))
}

/// Serialises the coefficients a training run ended with into a single JSON
/// line.
///
/// Accepts a slice, so both the `Vec<f64>` returned by `train_model` and a
/// plain array can be passed without cloning at the call site.
pub fn coefficients_to_json(last_coefficients : &[f64]) -> Result<String, JsonConverterError> {
    let result = TrainingResult { last_coefficients : last_coefficients.to_vec() };

    Ok(serde_json::to_string(&result)?)
}

/// Same as [`coefficients_to_json`], but indented for output a human reads or
/// for a file kept under version control.
pub fn coefficients_to_json_pretty(
    last_coefficients : &[f64]
) -> Result<String, JsonConverterError> {
    let result = TrainingResult { last_coefficients : last_coefficients.to_vec() };

    Ok(serde_json::to_string_pretty(&result)?)
}

/// Rejects payloads that parse as JSON but cannot describe a training set.
///
/// The feature count is taken from the first sample and every other sample is
/// checked against it, which is the check the model's own `validate_data_set`
/// does not perform.
fn validate(data : &TrainingData) -> Result<(), JsonConverterError> {
    let Some(first_sample) = data.inputs.first() else {
        return Err(JsonConverterError::EmptyDataSet);
    };

    if data.inputs.len() != data.outputs.len() {
        return Err(JsonConverterError::SampleCountMismatch {
            inputs : data.inputs.len(),
            outputs : data.outputs.len()
        });
    }

    let n : usize = first_sample.len();
    for (index, sample) in data.inputs.iter().enumerate() {
        if sample.len() != n {
            return Err(JsonConverterError::RaggedSample {
                index,
                expected : n,
                found : sample.len()
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID : &str = r#"{
        "inputs":  [[55.0, 1.0], [130.0, 4.0], [85.0, 2.0]],
        "outputs": [23000.0, 48500.0, 32000.0]
    }"#;

    #[test]
    fn parses_a_valid_training_set() {
        let (inputs, outputs) = training_data_from_json(VALID).unwrap();

        assert_eq!(inputs, vec![vec![55.0, 1.0], vec![130.0, 4.0], vec![85.0, 2.0]]);
        assert_eq!(outputs, vec![23000.0, 48500.0, 32000.0]);
    }

    #[test]
    fn accepts_integer_and_exponent_notation() {
        let json = r#"{"inputs": [[55, 1]], "outputs": [2.3e4]}"#;
        let (inputs, outputs) = training_data_from_json(json).unwrap();

        assert_eq!(inputs, vec![vec![55.0, 1.0]]);
        assert_eq!(outputs, vec![23000.0]);
    }

    #[test]
    fn rejects_an_empty_data_set() {
        let json = r#"{"inputs": [], "outputs": []}"#;

        assert!(matches!(
            training_data_from_json(json),
            Err(JsonConverterError::EmptyDataSet)
        ));
    }

    #[test]
    fn rejects_a_sample_count_mismatch() {
        let json = r#"{"inputs": [[55.0, 1.0], [130.0, 4.0]], "outputs": [23000.0]}"#;

        assert!(matches!(
            training_data_from_json(json),
            Err(JsonConverterError::SampleCountMismatch { inputs : 2, outputs : 1 })
        ));
    }

    #[test]
    fn rejects_a_ragged_sample() {
        let json = r#"{"inputs": [[55.0, 1.0], [130.0]], "outputs": [23000.0, 48500.0]}"#;

        assert!(matches!(
            training_data_from_json(json),
            Err(JsonConverterError::RaggedSample { index : 1, expected : 2, found : 1 })
        ));
    }

    #[test]
    fn rejects_malformed_json() {
        assert!(matches!(
            training_data_from_json("{\"inputs\": ["),
            Err(JsonConverterError::Json(_))
        ));
    }

    #[test]
    fn rejects_a_missing_field() {
        assert!(matches!(
            training_data_from_json(r#"{"inputs": [[55.0]]}"#),
            Err(JsonConverterError::Json(_))
        ));
    }

    #[test]
    fn serialises_coefficients() {
        let json = coefficients_to_json(&[375.1373157216981, -195.00662490954502]).unwrap();

        assert_eq!(
            json,
            r#"{"last_coefficients":[375.1373157216981,-195.00662490954502]}"#
        );
    }

    #[test]
    fn coefficients_survive_a_round_trip() {
        let coefficients = vec![375.1373157216981, -195.00662490954502, 1807.2879939508762];
        let json = coefficients_to_json_pretty(&coefficients).unwrap();
        let parsed : TrainingResult = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.last_coefficients, coefficients);
    }
}
