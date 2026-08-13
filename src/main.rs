mod data_set;
mod without_feature_scaling;

use data_set::DataSet;
use without_feature_scaling::MultivariableLinearRegression;

fn main() {
    let gemini_created_data_set = DataSet::load(data_set::DEFAULT_PATH);

    let learning_rate_alpha = 0.00003;
    let loop_count_for_train = 1_000_000;

    let mut model = MultivariableLinearRegression::new([149.1959, 4.08469], 1.37315);

    model.train_data_set(
        &gemini_created_data_set.inputs,
        &gemini_created_data_set.outputs,
        loop_count_for_train,
        learning_rate_alpha,
    );

    // **** Learning Too Slow !! *****
}
