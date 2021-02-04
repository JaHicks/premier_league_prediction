# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import ParameterGrid


@click.command()
@click.argument('interim_filepath', type=click.Path(exists=True))
@click.argument('models_filepath', type=click.Path())
def main(interim_filepath, models_filepath):
    """ Runs model training and selection for championship match transformation
    """
    logger = logging.getLogger(__name__)

    # generates {interim_filepath}/season_summaries.csv
    calculate_season_summaries(interim_filepath)

    # generates {interim_filepath}/promoted_and_relegated_season_summaries.csv
    separate_promoted_and_relegated_season_summaries(interim_filepath)

    # train all models with tuning options
    train_models(interim_filepath, models_filepath)

    # select best tuned model per feature per season
    select_best_models(models_filepath)

    logger.info("Trained and selected best championship transformation models")


def set_column_constant_lists():
    """ Return column lists so they only need to be calculated once
    """
    feature_bases = ["goals", "shots", "shotsOnTarget", "corners", "fouls",
                     "yellowCards", "redCards"]
    home_columns = []
    away_columns = []
    for for_against in ["for", "against"]:
        for feature_base in feature_bases:
            home_column = "home_{}_{}".format(feature_base, for_against)
            away_column = "away_{}_{}".format(feature_base, for_against)
            home_columns.append(home_column)
            away_columns.append(away_column)

    mean_columns = ["{}_mean".format(col)
                    for col in home_columns + away_columns]

    premier_columns = ["{}_premier".format(col) for col in mean_columns]

    return (feature_bases,
            home_columns,
            away_columns,
            mean_columns,
            premier_columns)


FEATURE_BASES, HOME_COLUMNS, AWAY_COLUMNS, MEAN_COLUMNS, PREMIER_COLUMNS =\
    set_column_constant_lists()


def calculate_season_summaries(interim_filepath):
    """ Calculate mean feature values per team per season
    """
    logger = logging.getLogger(__name__)
    logger.info("Calculating season summaries...")

    try:
        matches = pd.read_csv("{}/matches_expanded_features.csv".format(
            interim_filepath))
    except FileNotFoundError as e:
        msg = "Make sure the previous steps have been run using: make data"
        raise Exception(msg) from e

    # Get mean feature values per team per season for home matches
    home_index_columns = ["home_team", "season", "division"]
    home_season_summary_columns = HOME_COLUMNS + home_index_columns

    home_season_summaries = matches[home_season_summary_columns].pivot_table(
        index=home_index_columns, aggfunc=np.mean)
    home_season_summaries = home_season_summaries.add_suffix("_mean")

    # Make sure indexes are the same for home and away
    home_season_summaries.index.rename("team", level=0, inplace=True)

    # Get mean feature values per team per season for away matches
    away_index_columns = ["away_team", "season", "division"]
    away_season_summary_columns = AWAY_COLUMNS + away_index_columns

    away_season_summaries = matches[away_season_summary_columns].pivot_table(
        index=away_index_columns, aggfunc=np.mean)
    away_season_summaries = away_season_summaries.add_suffix("_mean")

    # Make sure indexes are the same for home and away
    away_season_summaries.index.rename("team", level=0, inplace=True)

    season_summaries = home_season_summaries.merge(away_season_summaries,
                                                   left_index=True,
                                                   right_index=True)

    # Separate multiIndex in to individual columns for easier grouping later
    season_summaries = pd.DataFrame(season_summaries.reset_index())

    # Populate values per team for the previous season
    columns = MEAN_COLUMNS + ["season", "division"]

    for column in columns:
        column_previous = "{}_previous".format(column)
        season_summaries[column_previous] = season_summaries.groupby(
            ["team"])[column].shift(1)

    filename = "{}/season_summaries.csv".format(interim_filepath)
    season_summaries.to_csv(filename, index=False)
    logger.info("Saved file: {}".format(filename))

    logger.info("Calculated season summaries!")


def separate_promoted_and_relegated_season_summaries(interim_filepath):
    """ Isolate promoted and relegated season summaries with
         mean feature values for championship and premier league seasons
    """
    logger = logging.getLogger(__name__)
    logger.info("Separating promoted & relegated season summaries...")

    season_summaries = pd.read_csv("{}/season_summaries.csv".format(
        interim_filepath))

    # Gather together the promoted teams and rename columns:
    #  *_mean renamed to *_premier
    #   (i.e. first season in premier league after promotion)
    #  *_mean_previous renamed to *_championship
    #   (i.e. last season in championship before promotion)
    promoted_season_summaries = season_summaries[
        (season_summaries.division == "premier") &
        (season_summaries.division_previous == "championship")
    ]
    rename_cols = {}
    for mean_column in MEAN_COLUMNS:
        rename_cols[mean_column] = "{}_premier".format(mean_column)
        previous_column = "{}_previous".format(mean_column)
        rename_cols[previous_column] = "{}_championship".format(mean_column)
    promoted_season_summaries = promoted_season_summaries.rename(
        columns=rename_cols)

    # Gather together the relegated teams and rename columns:
    #  *_mean renamed to *_championship
    #   (i.e. first season in championship after relegation)
    #  *_mean_previous renamed to *_premier
    #   (i.e. last season in premier league before relegation)
    relegated_season_summaries = season_summaries[
        (season_summaries.division == "championship") &
        (season_summaries.division_previous == "premier")
    ]
    rename_cols = {}
    for mean_column in MEAN_COLUMNS:
        rename_cols[mean_column] = "{}_championship".format(mean_column)
        previous_column = "{}_previous".format(mean_column)
        rename_cols[previous_column] = "{}_premier".format(mean_column)
    relegated_season_summaries = relegated_season_summaries.rename(
        columns=rename_cols)

    # Prepare promoted teams in same way as promoted_and_relegated
    promoted_season_summaries = promoted_season_summaries.sort_values(
        by=["season", "division", "team"], ignore_index=True)
    filename = "{}/promoted_season_summaries.csv".format(interim_filepath)
    promoted_season_summaries.to_csv(filename, index=False)
    logger.info("Saved file: {}".format(filename))

    promoted_and_relegated_season_summaries = promoted_season_summaries.append(
        relegated_season_summaries)
    promoted_and_relegated_season_summaries.sort_values(
        by=["season", "division", "team"], inplace=True, ignore_index=True)

    filename = "{}/promoted_and_relegated_season_summaries.csv".format(
        interim_filepath)
    promoted_and_relegated_season_summaries.to_csv(filename, index=False)
    logger.info("Saved file: {}".format(filename))

    logger.info("Separated promoted,"
                " and promoted & relegated season summaries!")


def setup_sample_weighting_columns(df):
    """
    """
    # samples will be weighted twice as highly compared to 1 season before
    df["1_season_half_life_weight"] = (2**1) ** (df["season"] - 2001)

    # samples will be weighted twice as highly compared to 2 seasons before
    df["2_season_half_life_weight"] = (2**(1/2)) ** (df["season"] - 2001)

    # samples will be weighted twice as highly compared to 4 seasons before
    df["4_season_half_life_weight"] = (2**(1/4)) ** (df["season"] - 2001)

    # samples will be weighted twice as highly compared to 8 seasons before
    df["8_season_half_life_weight"] = (2**(1/8)) ** (df["season"] - 2001)

    # samples will be weighted twice as highly compared to 16 seasons before
    df["16_season_half_life_weight"] = (2**(1/16)) ** (df["season"] - 2001)

    # samples will be weighted twice as highly compared to 1 season before
    df["unweighted"] = 1


def get_intercept_and_coefficient(model):
    if model.__class__ == DummyRegressor:
        intercept = model.constant_[0][0]
        coefficient = 0
    elif model.__class__ == Ridge:
        intercept = model.intercept_
        coefficient = model.coef_[0]
    return intercept, coefficient


def train_and_test_model_for_seasons(
        interim_filepath,
        model_name,
        model,
        parameter_grid=[{}],
        training_samples=["Promoted", "Promoted & Relegated"],
        sample_weight_columns=["unweighted"],
        target_variables=PREMIER_COLUMNS,
        seasons=range(2002, 2021)):
    """ Trains models for each feature for each season
         using only data from earlier seasons
        Returns fitted model details for each feature/season/model variation
         along with the mean squared error on the test season
    """
    logger = logging.getLogger(__name__)
    logger.info("Training championship transformation model: {}".format(
        model_name))

    fitted_models = []

    for training_sample in training_samples:
        log_message = "Training for {} teams".format(training_sample)
        logger.info(log_message)

        if training_sample == "Promoted":
            df_filename = "promoted_season_summaries.csv"
        elif training_sample == "Promoted & Relegated":
            df_filename = "promoted_and_relegated_season_summaries.csv"
        else:
            raise "training_sample not recognised: {}"

        df = pd.read_csv("{}/{}".format(interim_filepath, df_filename))
        setup_sample_weighting_columns(df)

        for season in seasons:
            log_message = "Training {} {}/{}".format(model_name, season,
                                                     seasons[-1])
            logger.info(log_message)

            # Train with all previous records
            train_records = df[
                (df.season < season)
            ]

            # Test with promoted teams from current season
            test_records = df[
                (df.season == season) &
                (df.division == "premier") &
                (df.division_previous == "championship")
            ]

            # Train each target feature separately
            for target_variable in target_variables:

                # Only use championship version of feature
                training_features = [target_variable.replace("_premier",
                                                             "_championship")]

                X_train = train_records[training_features].values
                y_train = train_records[target_variable].values

                X_test = test_records[training_features].values
                y_test = test_records[target_variable].values

                for sample_weight_column in sample_weight_columns:
                    sample_weight = train_records[sample_weight_column]

                    for model_parameters in parameter_grid:
                        model.set_params(**model_parameters)

                        model.fit(X_train, y_train,
                                  sample_weight=sample_weight)

                        intercept, coefficient =\
                            get_intercept_and_coefficient(model)

                        y_pred = model.predict(X_test)

                        mse = mean_squared_error(y_test, y_pred)

                        fitted_model = {
                            "target_variable": target_variable,
                            "season": season,
                            "model_name": model_name,
                            "training_sample": training_sample,
                            "weighting_column": sample_weight_column,
                            "parameters": str(model_parameters),
                            "mse": mse,
                            "intercept": intercept,
                            "coefficient": coefficient
                        }

                        fitted_models.append(fitted_model)

    log_message = "Training {} Completed!".format(model_name)
    logger.info(log_message)

    return fitted_models


def train_models(interim_filepath, models_filepath):
    """
    """
    logger = logging.getLogger(__name__)
    model_scores_per_season = pd.DataFrame(columns=[
        "target_variable",
        "season",
        "model_name",
        "training_sample",
        "weighting_column",
        "parameters",
        "mse",
        "intercept",
        "coefficient"
    ])

    # Fit baseline model: unweighted mean of all previous promoted teams
    baseline_models = train_and_test_model_for_seasons(
        interim_filepath,
        model_name="Baseline",
        model=DummyRegressor(strategy="mean"),
        sample_weight_columns=["unweighted"],
        training_samples=["Promoted"]
    )

    # Fit average model with range of sample weightings and
    #  try median as well as mean strategy
    param_options = {
        "strategy": ["mean", "median"]
    }
    average_models = train_and_test_model_for_seasons(
        interim_filepath,
        model_name="Average",
        model=DummyRegressor(),
        parameter_grid=ParameterGrid(param_options),
        sample_weight_columns=[
            "unweighted",
            "16_season_half_life_weight",
            "8_season_half_life_weight",
            "4_season_half_life_weight",
            "2_season_half_life_weight",
            "1_season_half_life_weight"
        ]
    )

    # Fit ridge regression model with range of regularisation parameters
    param_options = {
        "alpha": [float('%.1g' % x) for x in np.logspace(3, -4, 15)]
    }
    ridge_models = train_and_test_model_for_seasons(
        interim_filepath,
        model_name="Ridge",
        model=Ridge(),
        parameter_grid=ParameterGrid(param_options),
        sample_weight_columns=[
            "unweighted",
            "16_season_half_life_weight",
            "8_season_half_life_weight",
            "4_season_half_life_weight",
            "2_season_half_life_weight",
            "1_season_half_life_weight"
        ]
    )

    fitted_models = baseline_models + average_models + ridge_models
    model_scores_per_season = pd.DataFrame(fitted_models)

    filename = "{}/championship_transformation_models.csv".format(
        models_filepath)
    model_scores_per_season.to_csv(filename, index=False)
    logger.info("Saved file: {}".format(filename))


def select_tuned_models(df):
    # First tune models by selecting the best performing parameters
    #  from previous seasons
    df_group = df.groupby(["target_variable",
                           "model_name",
                           "training_sample",
                           "weighting_column",
                           "parameters"])

    df["base_predicted_mse"] = df_group["mse"].transform(
                                lambda x: x.expanding().mean().shift(1))
    df["base_predicted_mse"].fillna(10, inplace=True)

    df_group = df.groupby(["target_variable", "model_name", "season"])
    tuned_model_ids = df_group["base_predicted_mse"].idxmin().values
    df["tuned_model"] = False
    df.loc[df.index.isin(tuned_model_ids), "tuned_model"] = True

    # Then select best tuned model based on how the tuned models
    #  have performed previously
    tuned_models_df = df[(df.tuned_model)]

    df_group = tuned_models_df.groupby(["target_variable", "model_name"])

    df["model_predicted_mse"] = df_group["mse"].transform(
                                lambda x: x.expanding().mean().shift(1))
    df["model_predicted_mse"].fillna(10, inplace=True)

    df_group = df.groupby(["target_variable", "season"])
    best_model_ids = df_group["model_predicted_mse"].idxmin().values
    df["best_model"] = False
    df.loc[df.index.isin(best_model_ids), "best_model"] = True


def select_best_models(models_filepath):
    logger = logging.getLogger(__name__)
    logger.info("Selecting best models...")

    filepath = "{}/championship_transformation_models.csv".format(
        models_filepath)
    model_scores_per_season = pd.read_csv(filepath)

    # Make sure they are sorted by season, but keep the order of models
    model_scores_per_season['model_name'] = pd.Categorical(
        model_scores_per_season['model_name'],
        ["Baseline", "Average", "Ridge"])
    model_scores_per_season.sort_values(["season", "model_name"], inplace=True)

    select_tuned_models(model_scores_per_season)

    # Save each of the models along with actual and predicted MSE
    #  for analysing model selection process
    tuned_models_df = model_scores_per_season[
        ["target_variable", "season", "model_name", "training_sample",
         "weighting_column", "parameters", "intercept", "coefficient",
         "mse", "model_predicted_mse", "base_predicted_mse",
         "tuned_model", "best_model"]
         ].sort_values(["target_variable", "season"])

    filename = "{}/championship_transformation_selected_models.csv".format(
        models_filepath)
    tuned_models_df.to_csv(filename, index=False)
    logger.info("Saved file: {}".format(filename))

    # Save selected best model per feature per season
    #  for transforming championship matches in to premier leage matches
    best_models_df = model_scores_per_season[
        (model_scores_per_season.best_model)
    ][["target_variable", "season", "model_name", "training_sample",
       "weighting_column", "parameters", "intercept", "coefficient"
       ]].sort_values(["target_variable", "season"])

    filename = "{}/championship_transformation_best_models.csv".format(
        models_filepath)
    best_models_df.to_csv(filename, index=False)
    logger.info("Saved file: {}".format(filename))

    logger.info("Best models selected!")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
