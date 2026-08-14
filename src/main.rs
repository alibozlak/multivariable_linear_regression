use crate::without_feature_scaling::WithoutFeatureScaling;

mod without_feature_scaling;
mod dataset;
mod json_converter;

/// Trains the model on the rental data set and prints the cost before and after.
///
/// Every sample is `[square meters, room count]` and its output is the rental
/// price. Because the features are not scaled, the learning rate has to stay
/// this small — a larger step makes the cost diverge instead of fall — and the
/// price of that is a high iteration count.
///
/// The third argument of [`WithoutFeatureScaling::new`] is the starting point
/// `[a_1, a_2, b]`. Instead of zeros it holds the coefficients of the previous
/// run, so this run picks training up where the last one stopped. The block at
/// the end of the function is the log of the runs so far: `J` keeps falling but
/// ever more slowly, which is the typical picture of gradient descent on
/// unscaled features.
fn main() {
    let m2_and_room_count_datas :Vec<Vec<f64>> = dataset::gemini_created_inputs_data_set();
    let prices : Vec<f64> = dataset::gemini_created_outputs();

    let learning_rate : f64 = 0.000003;
    let loop_count : usize = 1_000_000;

    let mut without_feature_scaling : WithoutFeatureScaling = WithoutFeatureScaling::new(
        m2_and_room_count_datas,
        prices,
        vec![378.4231259725358, -240.75627615723258, 1583.8807455185147]
    );

    println!("\n\nJ (Before model training) = {}", without_feature_scaling.J());
    let last_coefficients = without_feature_scaling.train_model(learning_rate, loop_count );
    println!("\nJ (After model training) = {}", without_feature_scaling.J());
    println!("\nLast coefficients: {:?}\n", last_coefficients);

    //1_000_000 iterations, initial_coefficients = [0.0;3], alpha = 0.000003 :
    //Last coefficients: [382.42810776151464, -226.37674378439436, 1100.1919909133364]
    //J (Before model training) = 1567635000
    //J (After model training) =  1465777

    //1_000_000 iterations, initial_coefficients = [382.42810776151464, -226.37674378439436, 1100.1919909133364], alpha = 0.000003 :
    //Last coefficients: [378.4231259725358, -240.75627615723258, 1583.8807455185147]
    //J (Before model training) = 1465777
    //J (After model training) =  1383285

    //1_000_000 iterations, initial_coefficients = [378.4231259725358, -240.75627615723258, 1583.8807455185147], alpha = 0.000003 :
    //Last coefficients: [375.1373157216981, -195.00662490954502, 1807.2879939508762]
    //J (Before model training) = 1383285
    //J (After model training) =  1365171
}
