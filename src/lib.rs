//! Multivariable linear regression trained with batch gradient descent,
//! implemented from scratch.
//!
//! The model function is `f(x) = a_1 * x_1 + ... + a_n * x_n + b` and the cost
//! it minimises is the mean squared error. Neither the cost nor the derivatives
//! come from a library — they are hand-derived, and the derivations live as PDFs
//! in the `math/` directory of the repository.
//!
//! # Modules
//!
//! * [`without_feature_scaling`] — the model itself: cost, partial derivatives
//!   and the descent loop. Depends on nothing outside `std`.
//! * [`json_converter`] — turns JSON payloads into the vectors the model takes,
//!   and a finished run's coefficients back into JSON.
//! * [`dataset`] — the synthetic rental data set the binary trains on, useful
//!   for trying the crate out without bringing your own data.
//!
//! # Example
//!
//! ```
//! use multivariable_linear_regression::json_converter;
//! use multivariable_linear_regression::without_feature_scaling::WithoutFeatureScaling;
//!
//! let json = r#"{
//!     "inputs":  [[55.0, 1.0], [130.0, 4.0], [85.0, 2.0]],
//!     "outputs": [23000.0, 48500.0, 32000.0]
//! }"#;
//!
//! let (inputs, outputs) = json_converter::training_data_from_json(json).unwrap();
//! let mut model = WithoutFeatureScaling::new(inputs, outputs, vec![0.0; 3]);
//!
//! let cost_before = model.J();
//! let last_coefficients = model.train_model(0.000003, 1_000);
//!
//! assert!(model.J() < cost_before);
//! assert_eq!(last_coefficients.len(), 3);
//! ```

pub mod without_feature_scaling;
pub mod json_converter;
pub mod dataset;
