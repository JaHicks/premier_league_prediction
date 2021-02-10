# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
from src.models.train_championship_transformation import\
    set_column_constant_lists


FEATURE_BASES, HOME_COLUMNS, AWAY_COLUMNS, MEAN_COLUMNS, PREMIER_COLUMNS =\
    set_column_constant_lists()


@click.command()
@click.argument('interim_filepath', type=click.Path(exists=True))
@click.argument('models_filepath', type=click.Path())
@click.argument('processed_filepath', type=click.Path())
def main(interim_filepath, models_filepath, processed_filepath):
    """ Generate combined dataset of premier & transformed championship matches

    1. split premier & championship matches
    2. split home & away championship matches (double number of matches)
    3. remove bookmaker features from championship matches
    4. transform home & away championship features
    5. update opponent values to match
    6. append championship matches to premier league, sort & save to single CSV
    """
    transform_championship_matches(interim_filepath, models_filepath,
                                   processed_filepath)


def transform_match_feature(df, column, models_df):
    # TODO: consider refactoring without looping over features & seasons
    #  but this runs fast enough and may be more readable
    prediction_feature = "{}_mean_premier".format(column)

    for season in models_df.season.unique():
        transformation_model = models_df[
            (models_df.season == season) &
            (models_df.target_variable == prediction_feature)
        ]

        intercept = transformation_model.intercept.values[0]
        coefficient = transformation_model.coefficient.values[0]

        # Apply linear transformation on feature for specified season
        df.loc[(df.season == season), column] =\
            coefficient * df[column] + intercept


def transform_championship_matches(interim_filepath,
                                   models_filepath,
                                   processed_filepath):
    logger = logging.getLogger(__name__)
    logger.info("Transforming championship matches...")

    logger.info("Preprocessing ready for transformation...")
    all_matches = pd.read_csv("{}/matches_expanded_features.csv".format(
            interim_filepath))

    premier_matches = all_matches[
        (all_matches.division == "premier")]

    # Not enough data to transform 2001 championship matches, so exclude
    championship_matches = all_matches[
        (all_matches.division == "championship") &
        (all_matches.season > 2001)]

    # Bookmaker odds will no longer be relevant on transformed matches
    bookmaker_columns = [
        "odds_interwetten_homeWin",
        "odds_interwetten_draw",
        "odds_interwetten_awayWin",
        "odds_williamHill_homeWin",
        "odds_williamHill_draw",
        "odds_williamHill_awayWin"]
    championship_matches = championship_matches.drop(columns=bookmaker_columns)

    # Duplicate home/away matches with opponents renamed
    championship_home_matches = championship_matches.copy()
    premier_away_teams = championship_home_matches["away_team"] + " (Premier)"
    championship_home_matches["away_team"] = premier_away_teams

    championship_away_matches = championship_matches.copy()
    premier_home_teams = championship_home_matches["home_team"] + " (Premier)"
    championship_away_matches["home_team"] = premier_home_teams

    filename = "{}/championship_transformation_best_models.csv".format(
        models_filepath)
    transformation_models = pd.read_csv(filename)

    logger.info("Transforming...")
    for home_column in HOME_COLUMNS:
        transform_match_feature(championship_home_matches, home_column,
                                transformation_models)

    for away_column in AWAY_COLUMNS:
        transform_match_feature(championship_away_matches, away_column,
                                transformation_models)

    # Update opponent values to match
    for feature_base in FEATURE_BASES:
        home_for_feature = "home_{}_for".format(feature_base)
        home_against_feature = "home_{}_against".format(feature_base)
        away_for_feature = "away_{}_for".format(feature_base)
        away_against_feature = "away_{}_against".format(feature_base)

        championship_away_matches[home_for_feature] =\
            championship_away_matches[away_against_feature]

        championship_away_matches[home_against_feature] =\
            championship_away_matches[away_for_feature]

        championship_home_matches[away_for_feature] =\
            championship_home_matches[home_against_feature]

        championship_home_matches[away_against_feature] =\
            championship_home_matches[home_for_feature]

    premier_equivalent_matches = premier_matches.append(
        championship_home_matches, ignore_index=True, verify_integrity=True)

    premier_equivalent_matches = premier_equivalent_matches.append(
        championship_away_matches, ignore_index=True, verify_integrity=True)

    premier_equivalent_matches = premier_equivalent_matches.sort_values(
        "date", ignore_index=True)

    filename = "{}/premier_equivalent_matches.csv".format(processed_filepath)
    premier_equivalent_matches.to_csv(filename, index=False)
    logger.info("Saved file: {}".format(filename))

    logger.info("Transformed championship matches!")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
